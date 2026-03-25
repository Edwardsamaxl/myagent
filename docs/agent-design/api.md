# Agent D：HTTP/API 与前后端契约设计（草案）

本文档描述 **`POST /api/chat` 扩展**、**流式策略（MVP）**、**错误码与 HTTP 映射**、**Web UI 最小暴露项**。  
**实现状态**：本文为设计稿；合并代码前须按 [coordination.md](../coordination.md) **§3.12** 先更新正式契约小节。

---

## 1. `POST /api/chat` 扩展设计

### 1.1 兼容原则（对现有客户端）

- 现有请求体 `{ "session_id"?, "message" }` **继续有效**，行为与当前实现一致。
- 新增字段均为**可选**；未传时不改变默认行为。

### 1.2 请求 JSON（扩展后）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `session_id` | string | 否 | 会话标识，默认 `default`。 |
| `message` | string | **是** | 本轮用户输入（非空）。与现有校验一致。 |
| `messages` | array | 否 | **OpenAI 风格**多轮消息：`{ "role": "user"\|"assistant"\|"system", "content": string }[]`。用于「单次请求携带上文」或与 session 持久化组合（见 1.4）。 |
| `use_rag` | boolean \| null | 否 | `null`/省略：遵循服务端 `RAG_ENABLED` 与既有编排；`true`：强制本请求走 RAG 子链路（若服务全局关闭 RAG，返回 `400` + `validation_error` 或专用 `code`，见 §3）；`false`：本请求跳过 RAG，仅 Agent 循环。 |
| `client_trace_id` | string | 否 | 调用方关联 ID（如前端 UUID）；服务端**原样回显**到响应，便于与日志/埋点对齐。 |
| `return_trace_id` | boolean | 否 | 默认 `false`。为 `true` 时，响应中尽量携带服务端 `trace_id`（来自 RAG 子调用或后续统一 trace，见 1.3）。 |

**约束建议（MVP）**

- `messages` 单请求条数上限（如 50）与单条 `content` 长度上限，由实现时在 `coordination.md` 写明，避免滥用。
- `messages` 中 `role` 仅允许上述三值；非法则 `400` + `validation_error`。

### 1.3 响应 JSON（扩展后）

在现有字段基础上**增量**（主字段名不变）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `answer` | string | 不变。 |
| `steps_used` | int | 不变。 |
| `tool_calls` | string[] 或 object[] | 不变（与现实现一致）；若未来结构化，须先更新契约。 |
| `session_id` | string | 不变。 |
| `rag` | object \| null | 不变；结构同 `§3.9` 成功体子集（与现 `AgentService.chat` 一致）。 |
| `client_trace_id` | string \| null | **新增**：请求传入则回显，否则 `null` 或省略（二选一在正式 § 中固定）。 |
| `trace_id` | string \| null | **新增**：当 `return_trace_id=true` 且本请求产生可关联的 trace（例如走了 RAG 且 `rag.trace_id` 存在）时返回；否则 `null` 或省略。 |

**与 PM 对齐点**

- `trace_id` 是否仅在 RAG 路径返回，还是 Agent 循环也统一挂一条 trace：建议 **Phase1** 仅 RAG 路径透传 `rag.trace_id`；Phase2 由编排层统一 `chat_trace_id`。

### 1.4 `messages` 与 `session` 合并策略（MVP 建议）

为避免与持久化 session 语义冲突，建议 **MVP 采用「显式模式」**（实现时可二选一，须在正式 § 写死）：

- **模式 A（推荐）**：仅传 `message` + `session_id`，历史仍来自 `SessionStore`；**不传 `messages`**。
- **模式 B**：传 `messages` 时表示「**仅本次推理**使用的临时上文」，**不写回** session；推理完成后仍将 `(user, assistant)` 按现有逻辑写入 session。  
  - 若同时传 `message`：将 `message` 视为**最后一条 user**，接在 `messages` 之后参与本次 prompt 拼装。

若 PM 选择「`messages` 全量替换 session 视图」，须单独开契约版本号，不建议作为 MVP 默认。

---

## 2. 流式 / 非流式：MVP 选择

### 2.1 MVP：**非流式**（保持 `POST /api/chat` 单次 JSON 响应）

**理由**

- 与当前 Flask 实现、`SimpleAgent` 循环、工具调用聚合方式一致，改动面最小。
- 前端与脚本（eval、curl）调试成本最低；错误码与 body 一次到位。
- Day3+ 生成与证据链仍在演进，流式协议（分块、取消、中途 tool 事件）易与契约锁死过早。

### 2.2 降级与演进

| 阶段 | 行为 |
|------|------|
| MVP | 仅 `POST /api/chat`，完整 JSON；长耗时由客户端超时与服务端 `timeout` 配置承担。 |
| 降级 | 模型不可用时返回 **JSON 错误**（503/502 + `code`），**不**退回 HTML；UI 展示 `error` 文案。 |
| 后续 | 新增 `GET`/`POST` ` /api/chat/stream`（如 SSE），与 JSON 路由并存；流式事件 schema 单独成章，**不**改变非流式响应字段名。 |

---

## 3. 错误码与 HTTP 状态映射

### 3.1 与 `coordination.md` §3.6 **现有**条目对齐（不重命名）

以下已在 §3.6 列出，**`/api/chat` 复用**（请求体/JSON 解析类）：

| HTTP | `code` | 适用端点 |
|------|--------|----------|
| 400 | `empty_body` | 通用 POST |
| 400 | `invalid_json` | 通用 POST |
| 400 | `validation_error` | 含 `message` 为空、`messages` 非法、`use_rag` 与全局配置冲突等（细则见正式 §） |
| 400 | `model_update_failed` | `/api/model`（chat 不用） |

### 3.2 **拟新增**（相对 §3.6 的 diff，合并前写入正式 §3.6）

| HTTP | `code` | 语义 | 适用 |
|------|--------|------|------|
| 502 | `chat_upstream_http_error` | 聊天链路调用的模型服务返回 4xx/5xx | `/api/chat` |
| 503 | `chat_model_unavailable` | 连接失败、超时、DNS 等 | `/api/chat` |
| 502 | `chat_upstream_request_error` | 其它 `requests` 级错误（非 HTTPError） | `/api/chat` |
| 500 | `chat_internal_error` | 未分类异常 | `/api/chat` |

**说明**

- 与 `/api/rag` 已存在的 `rag_*` **平行命名**，避免混用 `rag_model_unavailable` 表示 chat 失败（客户端难以区分来源）。
- **替代方案（不新增 code）**：在 §3.6 声明「模型上游失败统一使用 `llm_model_unavailable`」并废弃 `rag_*` 前缀——牵涉已有客户端，**不作为本草案默认**。

### 3.3 映射表（含 RAG 子路径业务成功但「拒答」）

以下 **HTTP 200**，**非** §3.6 错误体：

| 场景 | HTTP | 响应形状 |
|------|------|----------|
| RAG 证据不足 / 无命中 | 200 | `rag.refusal=true`，`rag.reason` 如 `no_retrieval_hit` / `insufficient_evidence` |
| 模型可用且 Agent 正常结束 | 200 | 现有 `answer` + `steps_used` + `tool_calls` |

---

## 4. Web UI 最小改动清单

仅列**需要暴露的 Agent 状态**（不大改布局前提下，侧边栏或消息下方一行即可）：

| 暴露项 | 数据来源 | 展示建议 |
|--------|----------|----------|
| 当前步数 | 响应 `steps_used` | 文案：`步数: {steps_used}` |
| 最近工具调用 | 响应 `tool_calls` | 展示最近 1～3 条（字符串则原样；若为结构化则 `name` + 摘要） |
| （可选）RAG 是否介入 | 响应 `rag` 非空且 `refusal` | 小标签：`RAG: 已引用` / `RAG: 拒答` / 无 |

**不在 MVP 强制暴露**：`trace_id` / `client_trace_id`（可在「高级调试」折叠区或仅控制台日志）。

---

## 5. 契约一致性结论（供 PM）

- **主字段兼容**：`answer`、`steps_used`、`tool_calls`、`session_id`、`rag` 名称与语义保持不变；仅**追加**可选响应字段 `client_trace_id`、`trace_id`（受 `return_trace_id` 控制）。
- **错误码**：在 §3.6 **增补** `chat_*` 四条后，`/api/chat` 模型失败与 `/api/rag` 对称且可区分。
- **流式**：MVP 不引入流式端点；后续独立路径扩展。

---

## 附录 A：`coordination.md` 拟增 § 草稿（可粘贴为 PR 评论或合并进正文）

> **状态**：拟稿；评审通过后：将下述 **§3.15** 并入 `coordination.md`，并视情况**修订 §3.8** 与 **§3.6**。

### 拟增 §3.15 `POST /api/chat` 扩展（请求/响应/合并策略）

- **请求 JSON（必填）**：`message`（非空字符串）。
- **请求 JSON（可选）**：
  - `session_id`（默认 `default`）
  - `messages`：`{role, content}[]`，`role ∈ {user, assistant, system}`；条数与长度上限见实现注释或本节补充。
  - `use_rag`：`boolean`；省略时遵循 `RAG_ENABLED`；与全局冲突时返回 `400` + `validation_error`（或专用子 code，若 §3.6 增补）。
  - `client_trace_id`：字符串；响应原样回显。
  - `return_trace_id`：布尔，默认 `false`；为 `true` 时响应可含 `trace_id`（优先来自 RAG 子调用）。
- **响应 JSON**：在既有字段上增加可选 `client_trace_id`、`trace_id`（定义见 [api.md](api.md) §1.3）。
- **`messages` 与 session**：MVP 默认 **模式 B**：`messages` 为本次推理临时上文，不替代 `SessionStore` 全量历史；最后一条 user 为 `message`。若变更合并策略须升版本节。

### 拟修订 §3.6「常见 code」增补一行

在现有列表末尾追加（实现 `/api/chat` 上游异常映射后生效）：

- `chat_upstream_http_error`（`/api/chat` 模型服务 HTTP 非成功，502）
- `chat_model_unavailable`（`/api/chat` 模型连接失败或超时，503）
- `chat_upstream_request_error`（`/api/chat` 其它请求层失败，502）
- `chat_internal_error`（`/api/chat` 未分类异常，500）

---

## 附录 B：与 `coordination.md` §3.6 的 diff 摘要（新增前先评审）

| 操作 | `code` | HTTP |
|------|--------|------|
| 新增 | `chat_upstream_http_error` | 502 |
| 新增 | `chat_model_unavailable` | 503 |
| 新增 | `chat_upstream_request_error` | 502 |
| 新增 | `chat_internal_error` | 500 |

**不变**：`empty_body`、`invalid_json`、`validation_error`、`invalid_top_k`、`invalid_chunk_params`、`model_update_failed`、`rag_*` 全套。
