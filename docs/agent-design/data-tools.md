# Agent 数据侧支撑设计（只读工具草案 + 语料版本 + 合规边界）

本文档由 **Agent A（数据 / Ingestion）** 维护，描述 Agent 可用的**只读数据工具**设计草案、与 `data/raw` / 评估集的关系，以及敏感与合规边界。  
**不写具体实现**；切块主逻辑不在本文范围内。

与生成侧拒答字段对齐见 `src/agent/core/generation.py` 中 `GenerationResult.reason` 的取值习惯。

---

## 1. Agent 可用「只读工具」清单草案

以下工具均为**只读**：不修改索引、不写库、不改 `data/raw`。实现时可挂在 Agent 循环或独立 HTTP 内部 API，由协调 Agent 与 **D（接口）** 定路由形态。

| 工具名（草案） | 用途 | 建议入参 | 建议出参（摘要） | 备注 |
|----------------|------|----------|------------------|------|
| `list_docs` | 列出当前可检索语料范围内的文档清单（逻辑文档，非强制等于单个 PDF） | 可选：`company` 过滤、`doc_type` 过滤、`limit` | `doc_id`、`source`、`metadata` 摘要（若有）、可选 `chunk_count` 估计 | 与入库时 `doc_id`/`source` 约定一致；内存索引场景下仅反映**当前进程已入库**内容 |
| `read_chunk_meta` | 按 `chunk_id` 读取**元数据与短预览**，不返回全文 | `chunk_id`（必填） | `chunk_id`、`doc_id`、`source`、`metadata`、`text_preview`（长度上限固定） | 与 `RetrievalHit` / `coordination.md` §3.9 字段对齐 |
| `search_doc_meta` | 按 `source` 前缀或 `company` 反查相关 chunk 的 meta 列表（分页） | `prefix` 或 `company`、`page`、`page_size` | `items[]`：每项含 `chunk_id`、`source`、`metadata` | 用于 Agent 自检「是否引用了同公司错误报告」 |
| `get_corpus_manifest` | 返回**当前评测/对话绑定的语料版本指纹**（见 §2） | 无或 `session_id` | `manifest_id`、`generated_at`、`doc_fingerprints[]` | 评测前由编排层注入，工具只读返回 |
| `list_eval_sets` | 列出仓库内评估集路径与行数（只读扫描 `data/eval/`） | 可选：`glob` | `path`、`line_count`、`sha256`（可选） | 不解析题目内容以外的敏感信息 |

**明确不做为只读工具的能力（避免范围膨胀）**

- 不暴露「任意路径读文件」（防止路径穿越）；若需读原始文件，应限定在 `data/raw/finance/**` 且经白名单校验。
- 不提供「删除 / 重建索引 / 改 chunk 文本」类工具（属运维或专用 ingestion 流程）。

---

## 2. 与 `data/raw`、`data/eval` 的关系及语料版本固定

### 2.1 目录角色

| 路径 | 角色 |
|------|------|
| `data/raw/finance/<公司>/` | 原始与派生文本（`.pdf`、脚本生成的 `.md`/`.txt`）；**评测语料的物理来源** |
| `data/eval/*.jsonl` | 离线评估问题与期望答案；`source` 字段应与入库时使用的 `source` 字符串一致（如 `同花顺/年报_2025.md`） |

### 2.2 Agent 评测时如何固定语料版本

目标：同一套 `eval` 跑分可复现，避免因「换了一批入库文件」导致指标不可比。

1. **入库前冻结清单**  
   - 记录本次评测依赖的 `(相对路径, 文件大小, 可选 sha256)` 列表，写入 `runtime/` 下某次运行的 manifest（如 `runtime/eval_manifest_<id>.json`），或由 CI 注入环境变量 `EVAL_MANIFEST_PATH`。

2. **`get_corpus_manifest` 与 trace 对齐**  
   - 服务启动并完成入库后，编排层生成 `manifest_id`（如各文件 hash 拼接的短摘要），写入 trace / 评测 JSON 的顶层字段，便于事后核对「当时索引里到底是哪些文档」。

3. **eval 行内 `source` 与 `run_ingest` 一致**  
   - 批量入库脚本使用 `source = "<公司目录名>/<文件名>"`；评估集 `source` 必须与之逐字一致，否则会出现「有数据但检索不到」的假阴性。

4. **重启服务**  
   - 当前索引为内存态时，每次评测 run 应固定顺序：`启动服务 → 按 manifest 入库 → 跑 eval`，避免中途混入库外文件。

---

## 3. 敏感与合规边界（与 C 侧 `reason` 对齐）

生成侧已使用的可观测 `reason`（见 `GroundedGenerator.generate` / `_postprocess_answer`）包括：

| `reason` | 典型含义 |
|----------|----------|
| `no_retrieval_hit` | 检索无命中 |
| `insufficient_evidence` | 证据不足、模型显式拒答、锚点覆盖过低、或谨慎模板退回拒答等 |
| `citation_missing` | 严格策略下缺少有效引用编号 |

以下 **查询类型或回答方式** 建议由 **C（生成 / Agent 循环）** 在 prompt 或前置策略中处理，并在拒答时**优先映射到上表已有 `reason`**，避免引入过多新枚举；若必须区分，可在 trace `payload` 中加 `sub_reason`（需协调 Agent 更新契约）。

### 3.1 建议必须拒答或不得仅凭库内证据断言的场景

| 类别 | 说明 | 建议映射 `reason` | 说明文（给用户）方向 |
|------|------|-------------------|----------------------|
| 非公开内幕 / 未披露重大信息 | 要求预测未公告业绩、未公开重组等 | `insufficient_evidence` | 明确「语料中无可靠披露，不提供猜测」 |
| 个股「买卖 / 加仓 / 清仓」类投资建议 | 属于投资建议监管敏感区 | `insufficient_evidence` | 可答「不构成投资建议」，不提供具体买卖指令 |
| 要求绕过披露、伪造引用 | 如「编造一段年报原文」 | `insufficient_evidence` 或 `citation_missing`（若试图生成却无引用） | 坚持可追溯引用 |
| 个人隐私与非上市公司敏感信息 | 与当前财报语料无关的隐私挖掘 | `insufficient_evidence` | 超出语料范围 |
| 违法违规指令 | 操纵市场、洗钱等 | `insufficient_evidence` | 拒绝执行 |

### 3.2 与 RAG 证据链的关系

- **有命中但不应强答**：仍可能落入 `insufficient_evidence`（锚点不足、目录型片段等），与 C 侧 `evaluate_anchor_coverage` 逻辑一致。
- **无命中**：统一 `no_retrieval_hit`，数据侧不通过「伪造工具结果」补命中。

### 3.3 数据侧责任边界

- Agent A 提供的只读工具**不得**返回未经验证的「结论」；仅返回**事实片段、元数据、清单**。
- 合规话术与最终拒答文案以 **C** 为准；本文仅规定**数据工具不越权**与 **`reason` 对齐原则**。

---

## 4. 后续交接

| 角色 | 建议动作 |
|------|----------|
| **协调 Agent** | 将本文工具名纳入 `coordination.md` 或 Day 计划「待实现 API」列表 |
| **D（接口）** | 若工具以 HTTP 暴露，定义路径、鉴权与速率限制 |
| **B（检索）** | `read_chunk_meta` 的字段与 `RetrievalHit` 保持一致 |
| **C（生成）** | 拒答策略与上表映射；如需 `sub_reason` 再开契约变更 |

---

*文档版本：与仓库内 `generation.py` 拒答 reason 一致；若 C 侧扩展 reason，请同步更新本节表格。*
