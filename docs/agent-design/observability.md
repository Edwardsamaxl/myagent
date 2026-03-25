# Agent MVP：可观测与评估口径设计

本文档由「评估与可观测」Agent E 输出，约定 **Agent 循环**（多步推理、工具调用）下的 trace 事件形态与 MVP 指标，并说明与现有 `runtime/traces.jsonl`、`runtime/eval_records.jsonl` 的兼容方式。

---

## 1. Trace 事件 Schema（每步记录什么）

### 1.1 顶层行结构（与现有 `TraceLogger` 一致）

每条 trace 仍为 **JSONL 一行**，顶层字段与 `src/agent/core/observability.py` 中 `TraceLogger.log` 写入格式对齐：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `trace_id` | string | 是 | 与单次「用户任务 / 会话轮次」绑定；建议沿用 12 位 hex，与同轮 `eval_records.jsonl` 可关联。 |
| `stage` | string | 是 | 事件类别，见下文「Agent 扩展 stage」。RAG 侧已有：`ingestion_start`、`ingestion_done`、`rag_answer`。 |
| `message` | string | 是 | 人类可读摘要，便于 grep 与排障。 |
| `payload` | object | 否 | **键值均为字符串**（与当前 `TraceEvent` 一致）；复杂结构用 JSON 字符串放入单一 key。 |
| `timestamp` | string | 是 | UTC ISO8601，与现有一致。 |

### 1.2 Agent 步级语义（建议在 `payload` 中统一）

为支持「每步」统计，在 **Agent 相关 `stage`** 的 `payload` 中约定下列键（值均为 string；数字、布尔、嵌套对象请 JSON 序列化后写入）：

| payload 键 | 含义 | 示例值 |
|------------|------|--------|
| `step` | 从 0 或 1 起的步序号（全任务内单调递增） | `"3"` |
| `step_name` | 逻辑步名（plan / think / act / observe 等） | `"act"` |
| `tool_name` | 工具标识；非工具步可为空字符串 | `"search"`、`""` |
| `tool_call_id` | 可选，同一工具多次调用时区分 | `"tc_01"` |
| `latency_ms` | 该步耗时（毫秒），端到端见 1.3 | `"120"` |
| `error_code` | 机器可读错误码；成功建议 `""` 或 `"ok"` | `"tool_timeout"`、`""` |
| `error_message` | 可选，简短错误说明 | `"deadline exceeded"` |
| `recovery` | 是否算一次「恢复/重试后继续」（见 2.3） | `"true"`、`"false"` |
| `input_digest` | 可选，输入摘要或 hash，避免日志过大 | `"sha256:..."` |
| `output_digest` | 可选，输出摘要或 hash | `"n_hits=3"` |

**`step` 与 `stage` 分工**：`stage` 用于日志分区与仪表盘聚合；`step` 用于单任务内顺序与「平均步数」统计。二者可同时存在。

### 1.3 建议新增的 `stage` 常量（实现时写入 `TraceStage` 或等价枚举）

| `stage` | 用途 |
|---------|------|
| `agent_session_start` | 任务开始：可带 `run_id`、`user_task_id`（字符串） |
| `agent_step` | 单步完成：必须带 `step`、`latency_ms`；有工具则带 `tool_name` |
| `agent_tool_start` | 可选：工具调用开始前打点，便于拆分排队与实际执行时间 |
| `agent_tool_end` | 可选：与 `agent_tool_start` 成对，计算工具纯耗时 |
| `agent_final` | 任务结束：建议带 `outcome`（`success` / `failure` / `aborted`）、`total_steps`、`e2e_latency_ms` |
| `agent_error` | 未捕获异常或致命失败：带 `error_code`、`error_message` |

**端到端延迟**：以同 `trace_id` 下 `agent_session_start.timestamp` 与 `agent_final.timestamp` 之差为准；或在 `agent_final.payload.e2e_latency_ms` 中显式写入字符串毫秒数，便于校验。

---

## 2. Agent MVP 指标（定义与聚合口径）

以下指标默认在 **单次用户任务** 粒度上先定义，再在评估集上取平均或比例。

### 2.1 任务成功率（task_success_rate）

- **分母**：评估集中有效任务数（排除跳过/损坏样本）。
- **分子**：`agent_final.payload.outcome == "success"`（或等价布尔字段）的任务数。
- **说明**：成功条件需与产品一致（例如：达到正确最终答案、或完成规定子目标）；评估脚本应在 `expected_outcome` 中写死判定规则，避免口径漂移。

### 2.2 平均步数（avg_steps）

- 对单次任务：`steps = max(step)` 或 `agent_final.payload.total_steps`（二者实现时只选一种，文档与代码保持一致）。
- 全集：`avg_steps = mean(steps)`。

### 2.3 工具失败率（tool_failure_rate）

- 单次任务：统计 `stage in {agent_step, agent_tool_end}` 且 `error_code` 非空且非 `ok` 的 **工具相关** 事件数，记为 `tool_fail_events`；分母为工具调用次数 `tool_calls`（`tool_name` 非空的步数）。
- 全集：`tool_failure_rate = sum(tool_fail_events) / max(sum(tool_calls), 1)`，或先按任务算比率再平均，**二选一写进评估脚本并固定**。

### 2.4 恢复次数（recovery_count）

- 单次任务：对 `payload.recovery == "true"` 的事件计数（或专用 `stage=agent_recovery` 计数）。
- 全集：可报告 `avg_recovery_count` 或 `recovery_rate`（至少任务占比：`recovery_count >= 1` 的任务比例）。

### 2.5 端到端延迟（e2e_latency_ms）

- 单次任务：见 1.3；单位毫秒。
- 全集：`avg_e2e_latency_ms`、`p50` / `p95`（若评估流水线支持分位数）。

---

## 3. 与现有 `eval_records.jsonl` / `traces.jsonl` 的兼容策略

### 3.1 设计原则

- **同一用户可见轮次** 仍用 **`trace_id` 串联** `traces.jsonl` 与 `eval_records.jsonl`，与 `runtime/README.md` 描述一致。
- **不破坏现有解析**：旧代码只认 `stage` + 字符串 `payload` 的，应仍能读 RAG 三类事件；Agent 事件通过 **新 `stage` 值** 区分，旧消费者可忽略未知 `stage`。

### 3.2 `traces.jsonl`：优先「新增 stage + payload 键」，必要时「新文件」

| 方案 | 做法 | 适用 |
|------|------|------|
| **A. 兼容扩展（推荐默认）** | 仍在 `runtime/traces.jsonl` 追加行；新增 `agent_*` 类 `stage`；步级字段全部放在 `payload`，且遵守 **键值均为字符串**。 | Agent 与 RAG 同进程、日志量可接受、希望一条 `trace_id` 全链路可查。 |
| **B. 新文件** | 新增例如 `runtime/agent_traces.jsonl`，行结构可与 `traces.jsonl` 相同或增加 `schema_version`。 | Agent 步事件极多、需单独轮转与采样；或希望 RAG trace 保持精简。 |

**推荐**：MVP 阶段采用 **方案 A**；当日志量或合规要求上升时，再切 **方案 B**，并在 `runtime/README.md` 增补文件说明。

### 3.3 `eval_records.jsonl`：以「可选字段 + 专用离线文件」为主

当前 `EvalRecord`（`src/agent/core/schemas.py`）面向 **单次问答结果行**，字段以 `question`、`answer`、`latency_ms` 等为主。

| 方案 | 做法 | 适用 |
|------|------|------|
| **A. 可选扩展字段** | 在 **同文件** 行尾增加可选键，例如 `agent_steps`、`task_outcome`、`tool_failure_count`（JSONL 灵活，旧聚合若按固定列需兼容未知键）。 | 希望在线与 Agent 共用一条 eval 行、且步数较少。 |
| **B. 新文件（推荐与复杂 Agent 评估配合）** | 例如 `runtime/agent_eval_records.jsonl`，每行包含 `trace_id`、`task_id`、`metrics`（JSON 字符串）及原始判定字段。 | Agent 任务与 RAG 一问一答模型差异大，避免污染现有 `aggregate_eval_rows` 口径。 |

**推荐**：MVP 若仍以「一轮对话 = 一条 eval」为主，可用 **方案 A** 增加少量可选字段；若引入多轮任务与复杂子目标，采用 **方案 B**，并在 `src/agent/core/evaluation.py` 或独立脚本中提供 `aggregate_agent_eval_rows`，与 RAG 指标函数并列，避免混算。

### 3.4 版本与迁移

- 在首行或文档中约定 `payload` 内可选 `schema_version`（如 `"1"`），便于日后收紧字段。
- 离线评估产物（如 `offline_eval_week1.jsonl`）继续通过 `trace_id` 与 `traces.jsonl` 关联；Agent 评估集建议单独 JSONL，字段在评测脚本 README 中写清。

---

## 4. 交付检查清单（实现侧对照）

- [ ] `TraceLogger` 写入仍满足：**顶层字段不变**；新增 Agent 事件不修改 RAG 既有 `stage` 语义。
- [ ] `payload` 值均为字符串；结构化内容 JSON 编码后写入单 key。
- [ ] 评估脚本固定：`avg_steps`、`tool_failure_rate` 的分子分母定义各一处。
- [ ] `runtime/README.md` 在落地 Agent 日志后同步增加文件/字段说明（若采用新文件则必更）。

---

## 5. 参考

- 实现参考：`src/agent/core/observability.py`、`runtime/README.md`
- 评估行参考：`src/agent/core/schemas.py` 中 `EvalRecord`、`src/agent/core/evaluation.py` 中聚合逻辑
