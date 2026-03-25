# Agent 与 RAG 桥接设计（检索在 Agent 中的位置）

本文档定义 **Agent 层如何调用 RAG** 的契约与路由原则，**不扩展检索算法**。实现以 [coordination.md](../coordination.md) §3 与当前代码为准；契约变更须先更新 `coordination.md` §3.12。

---

## 1. 调用关系（检索在链路中的位置）

```text
用户消息
  → AgentService.chat（编排）
       → [可选] 路由决策：是否调用 RAG、传何参数
       → RagAgentService.answer（检索 → 重排 → 同路径内 grounded 生成）
            → 写 trace（stage=rag_answer）；响应中的 answer/refusal 供调试与 /api/rag，不作为 chat 终答来源
       → 若有 retrieval_hits：将证据块注入本轮 user 消息（[检索证据]），再走 SimpleAgent（工具循环或自然语言终答）
       → 若无命中：仍走 SimpleAgent（无证据块或仅空检索信息），由对话策略与工具兜底
```

- **直连 RAG**（不经 Agent 工具循环）：`POST /api/rag` → `AgentService.rag_answer` → `RagAgentService.answer`。
- **对话 Agent**：`POST /api/chat` → `AgentService.chat`，内部仍调用同一套 `RagAgentService.answer`（当 `RAG_ENABLED=true`）。

---

## 2. Agent 调 RAG 的契约

### 2.1 输入（设计口径）

| 字段 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `question` | 是 | `str` | 用户问题全文；与当前 `RagAgentService.answer(question=...)` 一致。 |
| `top_k` | 否 | `int \| None` | **召回**阶段从索引取回的候选条数；缺省为环境变量 `RETRIEVAL_TOP_K`。重排条数仍由 `RERANK_TOP_K` 决定，与 `top_k` 独立。 |
| `filters` | 否 | `object` | **可选元数据过滤**（契约预留）。键建议与 `DocumentChunk.metadata` 对齐，见下表。 |

**`filters` 建议键（与 §3.1 metadata 一致，见 [coordination.md](../coordination.md) §3.1）**

| 键 | 含义 | 备注 |
|----|------|------|
| `company` | 公司/主体 | 实现时可做等值或子串匹配策略，需单独立文档。 |
| `doc_type` | 文档类型（如年报/半年报） | 与入库推导的 `doc_type` 一致。 |
| `date` | 报告期年份（当前库内多为四位年） | 可与 query 中年份合并策略由 B 定义。 |
| `source` | 精确或前缀匹配 `chunk.source` | 可选，用于多公司语料收窄。 |

**当前实现状态（事实）**

- `question`：已实现。
- `top_k`：`RagAgentService.answer` 与 `POST /api/rag` 支持；`AgentService.chat` **未向 RAG 传入自定义 `top_k`**，始终用配置默认。
- `filters`：**检索内核尚未消费**；Agent 层若提前传参，须在无实现前忽略或记录到 trace，避免 silently 误导用户。落地过滤需先改 `coordination.md` §3 再实现。

### 2.2 输出（沿用现有 RAG 结构）

与 `RagAgentService.answer` / [coordination.md](../coordination.md) §3.9 一致，**主字段名与形状不变**：

| 字段 | 说明 |
|------|------|
| `trace_id` | 本次 RAG 问答追踪 ID（12 位 hex）。 |
| `latency_ms` | 端到端耗时（检索+重排+生成）。 |
| `answer` | 模型输出正文。 |
| `refusal` | 是否拒答。 |
| `reason` | 如 `no_retrieval_hit`、`insufficient_evidence`（与生成侧对齐，见 §3.14）。 |
| `citations` | 引用行列表，与 `retrieval_hits` 序号一致。 |
| `retrieval_hits` | 列表；每项含 `chunk_id`、`score`、`source`、`metadata`、`text_preview`。 |

**说明（给 C）**：`retrieval_hits[].score` 为 **重排后仍保留的检索融合分**（`hit.score`），**不是** rerank 的 `final`；排序依据以 rerank 结果顺序为准。细粒度分解见 §4 trace 中的 `rerank_score_breakdown`。

---

## 3. 路由规则：何时必须检索、何时禁止检索

本节是 **Agent 编排层** 的路由设计，用于与 **C（生成/证据策略）** 的「是否依赖语料」判断对齐。  
[C 侧契约摘要](../coordination.md) §3.14：证据来自 `retrieval_hits`；拒答 reason 与现有口径一致；不在路由层做证据筛选算法。

### 3.1 决策表（目标口径，供 C 联调）

| 类别 | 条件（示例） | 动作 | C 侧关注点 |
|------|----------------|------|------------|
| **必须检索** | 问题依赖 **已入库财报/公告/内部文档** 中的事实、数字、表格、公司基本信息（与当前评测集同类） | 调用 `RagAgentService.answer`；**必须**将 `retrieval_hits`（若非空）交给生成或注入对话上下文 | 仅允许基于证据作答；无引用不硬答 |
| **禁止检索** | 产品侧关闭语料（`RAG_ENABLED=false` 或单次 `use_rag=false`） | **不调用** `RagAgentService.answer` | 由配置/请求统一控制；chat 路径**不再**按「工具/闲聊」意图二次跳过 RAG |
| **条件检索** | 多意图：部分需文档、部分需工具 | 先检索；**有 hits 即带证据进 Agent**（与 RAG 路径内 `refusal` 无关）；工具步不重复检索（见 `agent_loop` 约定） | 证据块与工具结果分区可读 |

### 3.2 与当前实现的 **分歧点**（须后续收敛）

| 分歧 | 目标（上表） | 当前代码事实 |
|------|----------------|----------------|
| D1 | 关闭 RAG 时不调检索 | 由 `rag_enabled` / `use_rag` 控制；开启时对非澄清轮**统一**调用 `rag.answer(rewrite_for_rag(...))`，意图仅用于澄清与计划摘要。 |
| D2 | `top_k` 可由 Agent 按任务调节 | `chat` 路径 **未传** `top_k`，仅用 `RETRIEVAL_TOP_K`。 |
| D3 | `filters` 收窄公司与报告类型 | **未实现**；检索为全库混合召回 + rerank 软约束。 |
| D4 | trace 中带 `session_id` / 路由原因 | `rag_answer` trace **无** `session_id`、`route_decision`（见 §4 建议字段）。 |

**建议收敛顺序（不改算法前提下）**：先 D1+D4（路由 + trace），再 D2；`filters` 待 B 与 §3 契约批准后实现。

---

## 4. Trace 中检索相关字段清单（与 E 对齐）

**载体**：`runtime/traces.jsonl`（或 `AgentConfig.trace_file`），`stage = rag_answer`（见 [`src/agent/core/observability.py`](../src/agent/core/observability.py)）。

**约束**：`payload` 内键值均为 **字符串**（JSON 对象需 `json.dumps` 成字符串）。

### 4.1 当前已写入字段（RAG 一次问答）

| payload 键 | 含义 | 消费方 E 用途 |
|--------------|------|----------------|
| `retrieved` | 召回候选条数（字符串化的整数） | 检索是否为空 |
| `reranked` | 重排后条数 | 与 `RERANK_TOP_K` 对齐检查 |
| `refusal` | 生成是否拒答 | 与 `reason` 联合统计 |
| `latency_ms` | 毫秒耗时 | SLA / 回归对比 |
| `query_tokens` | 检索侧分词（空格拼接） | 分词/召回诊断 |
| `top_scores` | 召回融合 top 分数字符串（逗号分隔） | 分数分布 |
| `zero_score_reasons` | 各 chunk 零分原因统计（JSON 字符串） | 索引健康度 |
| `fusion_mode` | 如 `weighted_sum` / `rrf` | 实验复现 |
| `score_breakdown` | 检索侧每路分数（JSON 数组字符串） | lexical / sparse / embedding 等 |
| `rerank_score_breakdown` | 重排可解释分（JSON 数组字符串） | `lexical` / `semantic` / `final` / `rank` / `base` 等 |

### 4.2 建议 E 与 Agent 编排后续补齐（契约级，未实现）

| 建议键 | 含义 |
|--------|------|
| `session_id` | 对话会话 ID（仅 `chat` 路径有意义） |
| `route_decision` | 如 `rag_required` / `rag_skipped` / `rag_hits_injected_to_agent`（chat 终答不经 RAG 字段直出） |
| `rag_top_k_requested` | Agent 传入的 `top_k`（若有） |
| `filters_json` | `filters` 序列化（若有） |

以上扩展若落地，须先更新 [coordination.md](../coordination.md) §3 与 §3.12。

---

## 5. 文档维护

- **Agent B**：检索与 rerank 行为、trace 字段与 `score_breakdown` 含义。
- **Agent C**：§3 决策表与拒答/引用策略一致性。
- **Agent E**：§4 仪表盘与离线关联 `trace_id` ↔ `eval_records.jsonl`。

---

## 6. 参考路径

| 路径 | 说明 |
|------|------|
| `src/agent/application/agent_service.py` | `chat` 中 RAG 调用与分支 |
| `src/agent/application/rag_agent_service.py` | `answer`、trace 写入 |
| `src/agent/core/observability.py` | Trace 格式说明 |
| `coordination.md` §3.9 / §3.14 | HTTP 与生成契约 |
