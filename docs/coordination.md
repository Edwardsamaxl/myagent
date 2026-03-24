# 多 Agent 协调说明（事实与接口）

本文档由**协调型 Agent**维护：把周计划拆成可验收任务、记录**真实存在**的交付物路径与接口约定，并在并行开发时标明**易冲突文件**与合并策略。

- **不写未在仓库中验证的路径**；下列路径均可在本仓库内检索到。
- 若与 [week1-finance-agent-plan.md](week1-finance-agent-plan.md) 字面不一致，**以本文「与计划差异」及代码为准**，并应回写周计划或本表。

---

## 1. 文档索引（当前 `docs/`）

| 文件 | 用途 |
|------|------|
| [week1-finance-agent-plan.md](week1-finance-agent-plan.md) | 第 1 周总目标与 Day 1～7 模板 |
| [day1-summary.md](day1-summary.md) | Day 1 已落地范围、数据与脚本清单 |
| [day2-daily-plan.md](day2-daily-plan.md) | Day 2 当日任务、依赖与验收 |
| [day2-final-summary.md](day2-final-summary.md) | Day 2 最终统一总结（替代分散 delivery 文档） |
| [day3-daily-plan.md](day3-daily-plan.md) | Day 3 开工任务、依赖与验收 |
| [coordination.md](coordination.md) | 本文：接口约定、任务依赖、冲突策略 |
| [agent-roles.md](agent-roles.md) | 各 Agent 职责分工（可随时改） |
| [explain-agent-prompt.md](explain-agent-prompt.md) | 解释与学习助手 Prompt（可直接复制） |

---

## 2. 与周计划差异（避免误读）

| 计划原文 | 仓库事实 |
|----------|----------|
| Day 1 产出 `docs/day1-requirements.md` | **未创建**；需求与落地总结见 [day1-summary.md](day1-summary.md) |
| 约 30 条评估样例 | 当前 [data/eval/week1_eval_set.jsonl](data/eval/week1_eval_set.jsonl) 为 **25** 条（可后续扩充至 30+） |
| Day 1 曾列 `docs/day1-ingestion-explained.md` | **不在仓库中**（勿引用该路径） |

---

## 3. 接口约定（契约）

### 3.1 `DocumentChunk`（`src/agent/core/schemas.py`）

- **字段**：`chunk_id: str`，`doc_id: str`，`text: str`，`source: str`，`metadata: dict[str, str]`（入库时由 `source` 推导并合并可选请求体 `metadata`）。
- **metadata 键（贯通约定）**：周计划与 [day2-daily-plan.md](day2-daily-plan.md) 统一使用以下键（值均为字符串）；推导规则见 [data/README.md](../data/README.md)「文档级元数据」。

| 键 | 含义 | 状态 |
|----|------|------|
| `doc_type` | 文档类型（如年报/半年报） | 已实现（由文件名启发式 + 扩展名） |
| `company` | 公司或主体名 | 已实现（`finance/` 下目录或 `公司/文件`） |
| `date` | 报告期或日期（当前为文件名中的四位年份） | 已实现 |
| `source` | 与 `chunk.source` 一致，便于在 `metadata` 内一并查看 | 已实现 |

可选键由调用方通过 `POST /api/ingest` 的 `metadata` 传入（覆盖同名字段）。

### 3.2 `POST /api/ingest`（`src/agent/interfaces/web_app.py`）

- **请求 JSON（必填）**：`doc_id`，`source`，`content`（均为非空字符串，否则 400）。
- **请求 JSON（可选）**：`metadata`（对象；键值写入前转为字符串，与 `derive_doc_metadata_from_source(source)` 结果合并，**同名字段以请求体为准**）。
- **请求 JSON（可选）**：`chunk_size`（正整数）、`chunk_overlap`（非负整数，且 `< chunk_size`）。
- **请求 JSON（可选）**：`dedup_across_docs`（布尔，默认 `false`；`true` 时启用跨文档内容去重）。
- **分块参数语义**：若本次请求携带 `chunk_size/chunk_overlap`，仅覆盖**本次入库**；未携带时仍使用 `AgentConfig`（`CHUNK_SIZE`/`CHUNK_OVERLAP`）默认值。
- **校验失败**：返回 400，`code=invalid_chunk_params`。

### 3.3 `ingest` 成功响应（`src/agent/application/rag_agent_service.py`）

返回字典包含：`trace_id`，`doc_id`，`source`，`total_chunks`，`deduplicated_chunks`，`dropped_duplicates`。  
**去重口径**：
- `total_chunks`：原始分块数（去重前）
- `deduplicated_chunks`：去重后实际写入数
- `dropped_duplicates`：被内容哈希去重丢弃的块数
- 默认仅对**同一次 ingest 内**重复块去重；跨文档去重需显式传 `dedup_across_docs=true`。

### 3.4 切块默认参数（`src/agent/config.py`）

- 环境变量 `CHUNK_SIZE`（默认 **500**）、`CHUNK_OVERLAP`（默认 **80**），由 `RagAgentService` 构造 `DocumentIngestionPipeline` 时使用。
- `POST /api/ingest` 可在单次请求里覆盖上述参数；覆盖值仅作用于该次 `ingest_text`，不会修改进程内默认配置。

### 3.5 相关脚本（`scripts/`）

| 路径 | 作用 |
|------|------|
| [scripts/run_ingest.py](../scripts/run_ingest.py) | 扫描 `data/raw/finance/**` 下 `.txt`/`.md`，对运行中的服务 `POST /api/ingest` |
| [scripts/dump_chunks.py](../scripts/dump_chunks.py) | 本地切块预览，输出到 `data/debug/` |
| [scripts/extract_pdf_to_md.py](../scripts/extract_pdf_to_md.py) | PDF → Markdown |
| [scripts/extract_pdf.py](../scripts/extract_pdf.py) | PDF → 纯文本 |

### 3.6 HTTP 错误响应（通用）

- **形状**：`{"error": "<可读说明>", "code": "<机器可读标识>"}`，HTTP 4xx/5xx 与语义一致。
- **常见 `code`**：`empty_body`（请求体为空）、`invalid_json`（非合法 JSON 对象）、`validation_error`（业务必填项缺失）、`invalid_top_k`（`/api/rag` 的 `top_k` 非整数）、`invalid_chunk_params`（`/api/ingest` 分块参数非法）、`model_update_failed`（模型切换校验失败）、`rag_upstream_http_error`（模型服务返回 4xx/5xx，HTTP 502）、`rag_model_unavailable`（模型服务连接失败或超时，HTTP 503）、`rag_upstream_request_error`（其他请求层失败，HTTP 502）、`rag_internal_error`（`/api/rag` 未分类异常，HTTP 500）。

### 3.7 可选 CORS

- 环境变量 **`WEB_CORS_ORIGINS`**：未设置时**不**添加 CORS 头（同站默认行为）。
- 设为 `*` 时：`Access-Control-Allow-Origin: *`。
- 设为逗号分隔的多源时：若请求带 `Origin` 且在列表中则回显该源；若仅一项则始终使用该源（便于本地固定前端端口）。

### 3.8 `POST /api/chat`（`src/agent/interfaces/web_app.py`）

- **请求 JSON**：`message`（必填，非空字符串）、`session_id`（可选，默认 `default`）。
- **响应**：由 `AgentService.chat` 返回，包含 `answer`、`steps_used`、`tool_calls`、`session_id`；若 `RAG_ENABLED=true`，另有 `rag`（与 §3.9 同结构的 RAG 结果或检索为空时的占位）。

### 3.9 `POST /api/rag`（`src/agent/interfaces/web_app.py`）

- **请求 JSON**：`question`（必填）；`top_k`（可选整数，缺省用 `RETRIEVAL_TOP_K`）。
- **成功响应**：与 `RagAgentService.answer` 一致，含 `trace_id`、`answer`、`refusal`、`reason`、`citations`、`retrieval_hits`。
- **`retrieval_hits` 每项**：含 `chunk_id`、`score`、`source`、`metadata`、`text_preview`（与 `RetrievalHit` 对齐，便于调试文档级元数据）。
- **字段稳定性要求（Day3）**：上述 5 个 `retrieval_hits` 字段名不变；如需扩展解释字段（如 `hit_reason`、`score_breakdown`），必须先在本节文档批准，再落实现。

### 3.10 Retrieval/Embedding 配置契约（Day3 新增）

- **新增配置组（允许通过 env 或配置对象注入）**：
  - `EMBEDDING_PROVIDER`（如 `ollama` / `openai` / `local`）
  - `EMBEDDING_MODEL`
  - `EMBEDDING_DIM`
  - `RETRIEVAL_FUSION_STRATEGY`（如 `weighted_sum` / `rrf`）
  - `RETRIEVAL_LEXICAL_TOP_K`
  - `RETRIEVAL_SEMANTIC_TOP_K`
  - `RETRIEVAL_HYBRID_TOP_K`
  - `RETRIEVAL_WEIGHT_LEXICAL`
  - `RETRIEVAL_WEIGHT_SEMANTIC`
- **兼容要求**：
  - 保留现有 lexical/TF-IDF 通道。
  - 新增 embedding 语义通道为可开关能力，默认值需可回退到 baseline。
  - 不要求修改 ingestion 主切块逻辑；向量可在检索侧按需生成/缓存。

### 3.11 评估记录口径（Day3 新增）

- Day3 离线评估输出（`runtime/*.json`）需新增以下字段：
  - `strategy_version`（例如 `baseline_v1` / `hybrid_v1`）
  - `fusion_strategy`
  - `weight_lexical`、`weight_semantic`
  - `retrieved_zero_rate`
  - `top1_hit_rate`
  - `refusal_rate`
  - `avg_latency_ms`
- 单样本建议保留：
  - `retrieval_hit_count`
  - `top3`（含 `source` 与可读预览）
  - `diagnosis_tag`（如 `tokenization_gap` / `expression_gap` / `year_number_miss`）。

### 3.12 契约变更放行规则（硬门禁）

- 触发范围：任何 **错误码集合**、**HTTP 状态语义**、**接口字段名/结构**（请求或响应）变化。
- 放行顺序：**先更新 `coordination.md` §3 对应小节，再合并实现代码**；未更新文档视为未对齐，不放行。
- 执行责任：PM 负责把关；D 负责 API 与错误码落地；A/B/C 在改动前先引用本节确认字段口径。

### 3.13 已知检索现象（2026-03-23）

- 现象：在当前语料下，`/api/rag` 对自然问句（如“贵州茅台2024年营业收入是多少？”）存在高概率 `no_retrieval_hit`；同题经 query rewrite 后可稳定命中。
- 证据文件：
  - `runtime/day2_query_rewrite_compare.json`
  - `runtime/day2_extended_eval_compare.json`
- 现阶段判定：
  - 摄入链路可用（最小复现实验中，新摄入文本可被命中）。
  - 主要矛盾在检索鲁棒性与排序质量，不在 API 契约字段。
- 责任分工：
  - Agent A：可提供临时 query rewrite 方案用于应急评测。
  - Agent B：作为根因责任方，优先优化检索切词与召回排序，减少“财务题召回到股东大会块”。

### 3.14 生成模块契约（证据选取 + 回答生成）

- **目标**：在检索/重排已有命中的前提下，保证回答“有证据、可引用、可拒答”。
- **证据选取输入**：
  - 来源：`retrieval_hits`（含 `chunk_id/score/source/metadata/text_preview`）。
  - 允许使用 rerank 后顺序作为证据优先级；不改 `retrieval_hits` 主字段名。
- **回答生成输出约束**：
  - 回答结构至少包含：`结论`、`关键依据`、`引用编号`。
  - 证据不足时必须拒答，且 reason code 与现有口径一致（`no_retrieval_hit` / `insufficient_evidence`）。
  - 不允许“无引用硬答”。
- **可扩展解释字段（可选）**：
  - 如需新增 `selected_evidence_ids`、`citation_spans`、`generation_confidence` 等字段，必须先更新本节文档再实现。
- **实现边界**：
  - Agent C 主责 `src/agent/core/generation.py`、`src/agent/core/evidence_format.py` 及相关编排调用；
  - 不在路由层实现证据筛选逻辑，不改 `/api/rag` 主字段契约。

---

## 4. 第 1 周任务依赖顺序（高层）

```text
需求与数据（Day 1）
    → Ingestion：元数据 + 入库可观测（Day 2）
        → Retrieval 可解释 / 混合分数（Day 3）
            → Rerank + 证据选取 + 回答生成约束（Day3+ / 原 Day4）
                → 评估闭环 + 指标（Day 5）
                    → 工具链与权限（Day 6）
                        → 周复盘与演示（Day 7）
```

- **硬依赖**：检索/重排/评估依赖 chunk 与 `source`/`metadata` 是否可追溯；扩展 `POST /api/ingest` 须在编排层与 UI 契约上一致。
- **软依赖**：评估集可在 Day 1 后持续追加，不阻塞 ingestion 改造，但阻塞「指标对比叙事」闭环。

---

## 5. 易冲突文件与合并策略

| 文件 | 风险 | 建议 |
|------|------|------|
| `src/agent/interfaces/web_app.py` | UI + API 同文件 | 一人改 **ingest/rag 路由**，另一人改 **聊天/技能** 时优先拆分 PR；同改路由则先统一 JSON 契约再合 |
| `src/agent/application/rag_agent_service.py` | RAG 编排中枢 | 按阶段拆：`ingest_document` 与 `rag_answer` 分支分开提交；共享 `TraceLogger` 时避免重复改 `payload` 结构 |
| `src/agent/core/ingestion.py` | 分块逻辑 | 先合「元数据/签名」再合「算法」；禁止并行改 `_split_fixed_then_sentence_end` 与重叠逻辑，除非一人只加测试/脚本 |
| `scripts/run_ingest.py` | 批量入库 | 与 `web_app` 契约绑定；API 变更必须同步此脚本与 [coordination.md](coordination.md) 第 3 节 |
| `src/agent/core/generation.py` + `src/agent/core/evidence_format.py` | 证据与回答格式耦合高 | C 先改生成模板，B 仅提供 rerank/证据顺序；若字段扩展，先更新 §3.14 再合并 |

**边界拆分示例**：Agent A 只增加 `DocumentChunk.metadata` 填充与单元级行为；Agent B 只增加 `run_ingest.py` 汇总打印与错误码——二者通过 **§3.1 键名约定**对接，减少同一 diff 内改 `ingestion`+`web_app`。

---

## 6. 子任务模板（复制到 Day 文档或 issue）

每项任务应包含：

1. **验收标准**：可检查的行为或产出（含路径或 API 字段）。
2. **不要做什么**：范围上限（见各 Day 文档「刻意不做」节）。
3. **依赖**：必须先完成的契约（见 §4）。

---

## 7. 维护职责（协调型 Agent）

1. 周计划与 `day*-*.md` 与代码/数据**事实一致**；发现漂移则改文档或列在 §2。
2. 新增或调整对外契约（metadata 键、API 字段、错误码/状态码）时，严格遵守 **§3.12 先文档后实现**。
3. 不臆造路径；新增文件若在评审中尚未合并，不得写入本节为「已存在」。
