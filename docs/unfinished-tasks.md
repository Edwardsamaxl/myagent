# 未完成功能清单

> 生成时间：2026-04-06（更新）
> 分析范围：coordinator-design.md、agent-design/*.md、coordinator.py、synthesizer.py、worker_result.py、plan_schema.py、registry.py 与实际代码的差距

---

## 高优先级

### 1. [并行执行实际并行化]
- **描述**：`_execute_plan` 设计为"并行调度 ready 步骤"，但实际用 `for step in ready: result = execute(...)` 串行执行，非真正的并行。
- **当前状态**：骨架实现（逻辑正确但非真正并行）
- **设计文档**：`docs/coordinator-design.md` §4.4 「并行执行所有就绪的步骤」
- **代码位置**：`src/agent/core/planning/coordinator.py` 第 286-290 行
- **建议完成方式**：使用 `asyncio` 或 `concurrent.futures.ThreadPoolExecutor` 改造为真正并行

### 2. [工具重试策略未实现]
- **描述**：`loop.md` §4.3 定义了差异化重试策略（`TOOL_NOT_FOUND` 不重试，`TOOL_VALIDATION_ERROR`/`TOOL_TIMEOUT` 最多 1 次），当前 `WorkerExecutor.execute` 无任何重试逻辑。
- **当前状态**：未实现
- **设计文档**：`docs/agent-design/loop.md` §4.3
- **代码位置**：`src/agent/core/planning/coordinator.py` 第 135-173 行（WorkerExecutor.execute）
- **建议完成方式**：根据异常类型实现差异化重试；第二次失败后必须换策略或终答

### 3. [Agent 步级 Trace 缺失]
- **描述**：`observability.md` 定义的 `agent_session_start`、`agent_step`、`agent_final`、`agent_error` 等 stage 未写入 traces.jsonl。
- **当前状态**：未实现
- **设计文档**：`docs/agent-design/observability.md` §1.2-§1.3
- **代码位置**：`src/agent/core/observability.py`
- **建议完成方式**：在 `Coordinator.run` 和 `_execute_plan` 中调用 `TraceLogger.log` 写入新 stage

### 4. [Max Steps 限制未实装]
- **描述**：`loop.md` 定义 `max_steps` 为单轮工具调用上限，但 `_execute_plan` 仅以 `max_iterations` 作为死锁保护，无真正的步数限制。
- **当前状态**：未实现
- **设计文档**：`docs/agent-design/loop.md` §1.3
- **代码位置**：`src/agent/core/planning/coordinator.py` 第 251-292 行
- **建议完成方式**：在 `Coordinator.run` 中增加 `max_steps` 参数，超限后跳过剩余步骤直接进入 synthesize

### 5. [RAG 与 Agent 协同路由（rag-bridge D1）]
- **描述**：RAG 与 Agent 协作职责边界不清晰。当前 `AgentService.chat` 在开启 RAG 时统一调用 `rag.answer`，但未按意图（knowledge_corpus/tool_only/mixed/chitchat）做路由决策。
- **当前状态**：部分实现
- **设计文档**：`docs/agent-design/rag-bridge.md` §3.1 决策表
- **代码位置**：`src/agent/application/agent_service.py` 第 95 行附近
- **差距**：`_should_use_coordinator` 仅判断是否走 Coordinator，未区分「必须 RAG/禁止 RAG/条件检索」三类场景
- **建议完成方式**：按 rag-bridge §3.1 决策表，在 `chat` 方法中增加意图路由逻辑；D2（top_k 可调节）、D3（filters）可后续迭代

---

## 中优先级

### 7. [意图分类与澄清规则（dialogue-planning Phase 1）]
- **描述**：`classify_intent` 已实现，但返回的 `IntentResult` 仅用于判断 `AMBIGUOUS` 分支；其他意图（knowledge_corpus/tool_only/mixed/chitchat）未用于 RAG 路由决策。
- **当前状态**：骨架实现
- **设计文档**：`docs/agent-design/dialogue-planning.md` §1.1 Intent 枚举、§1.2 澄清策略
- **代码位置**：`src/agent/core/dialogue/intent_classifier.py` 或 `dialogue.py`
- **建议完成方式**：Phase 1 规则+关键词方案落地，将 `IntentKind` 扩展与 RAG 路由联动

### 8. [Evaluator/Reranker 骨架实现（待完善）]
- **描述**：`SimpleReranker` 在 `src/agent/core/rerank.py` 为骨架实现，仅做简单的分数排序，未实现真正的重排模型（如 BGE-Reranker）。
- **当前状态**：骨架实现
- **设计文档**：`docs/coordinator-design.md` §4.3 Worker 执行器
- **代码位置**：`src/agent/core/rerank.py`
- **建议完成方式**：接入 `BGE-Reranker` 或 `Cohere Rerank API`；当前 fallback 为按原始分数排序

### 9. [RAG trace 字段不完整（session_id/route_decision）]
- **描述**：rag-bridge.md §4.2 建议的 `session_id`、`route_decision` 等字段未写入 trace。
- **当前状态**：未实现
- **设计文档**：`docs/agent-design/rag-bridge.md` §4.2
- **代码位置**：`src/agent/core/observability.py` 或 `rag_agent_service.py`
- **建议完成方式**：在 `RagAgentService.answer` 返回的 trace payload 中增加这些字段

### 10. [top_k / filters 参数未传递给 RAG（rag-bridge D2/D3）]
- **描述**：`AgentService.chat` 未将意图分类出的 `top_k`（召回阶段）和 `filters`（元数据过滤）传递给 `rag.answer`。
- **当前状态**：未实现
- **设计文档**：`docs/agent-design/rag-bridge.md` §2.1 输入表、§3.2 分歧点 D2/D3
- **代码位置**：`src/agent/application/agent_service.py` 第 103-125 行
- **建议完成方式**：D2 简单（chat 路径传 top_k），D3 需先改 coordination.md §3

### 11. [`remember` 工具未映射到 MEMORY action]
- **描述**：`WorkerExecutor.action_to_tool` 仅将 `MEMORY` 映射到 `read_memory`，`remember`（写记忆）工具存在但未接入。
- **当前状态**：部分实现
- **设计文档**：`docs/coordinator-design.md` §4.3 action_to_tool
- **代码位置**：`src/agent/core/planning/coordinator.py` 第 138-143 行
- **建议完成方式**：在 `action_to_tool` 中增加 `MEMORY: "remember"` 或增加独立的 `WRITE_MEMORY` action

### 12. [工具结果格式未规范化]
- **描述**：`loop.md` §4.2 定义了 `[TOOL_RESULT]status: ok|error\ntool: ...\nmessage: ...\npayload: ...\n[/TOOL_RESULT]` 格式，当前 `WorkerExecutor` 直接返回 `str(result)`。
- **当前状态**：未实现
- **设计文档**：`docs/agent-design/loop.md` §4.2
- **代码位置**：`src/agent/core/planning/coordinator.py` 第 158-173 行
- **建议完成方式**：定义 `format_tool_result()` 统一输出格式

### 13. [记忆/技能长期上下文节流]
- **描述**：`build_long_context` 将 memory + skills 全文注入每次调用，长上下文场景可能超出模型窗口。
- **当前状态**：已实现（直接拼接）
- **设计文档**：`docs/agent-design/loop.md` 或对话规划相关
- **代码位置**：`src/agent/application/agent_service.py` 第 60-63 行
- **建议完成方式**：周期性摘要写入 `last_summary`；或按 token 预算截断

### 14. [MCP 工具扩展接口]
- **描述**：`coordinator-design.md` §8.2 定义了 `extend_tools` 方法用于运行时扩展 MCP 工具，当前 `Coordinator` 未实现。
- **当前状态**：未实现
- **设计文档**：`docs/coordinator-design.md` §8.2
- **代码位置**：`src/agent/core/planning/coordinator.py`
- **建议完成方式**：在 `Coordinator.__init__` 或 `AgentService` 中增加 `extend_tools(mcp_tools)` 方法

### 15. [数据工具均未实现]
- **描述**：`data-tools.md` 定义了 5 个只读工具草案（`list_docs`、`read_chunk_meta`、`search_doc_meta`、`get_corpus_manifest`、`list_eval_sets`），均为未实现状态。
- **当前状态**：未实现（草案）
- **设计文档**：`docs/agent-design/data-tools.md`
- **建议完成方式**：按需实现（取决于 Agent 是否需要主动查询语料状态）

### 16. [重规划机制缺失]
- **描述**：Planner 失败后未实现 replan 逻辑（`dialogue-planning.md` §2.4 定义），Coordinator 仅依赖 `_fallback_plan` 兜底，无重试计数。
- **当前状态**：未实现
- **设计文档**：`docs/agent-design/dialogue-planning.md` §2.4
- **代码位置**：`src/agent/core/planning/coordinator.py`
- **建议完成方式**：在 `Planner` 增加 `max_replan` 参数；失败时触发 replan 循环

---

## 已完成（骨架/部分功能）

| 功能 | 状态 | 代码位置 |
|------|------|----------|
| Coordinator/Planner/WorkerExecutor/Synthesizer 实装 | 已完成 | `coordinator.py` |
| PlanStepAction 枚举扩展（RAG/CALC/WEB/MEMORY/SYNTHESIZE） | 已完成 | `plan_schema.py` |
| WorkerResult 数据类 | 已完成 | `worker_result.py` |
| `use_coordinator` 配置项 | 已完成 | `config.py` |
| `search_knowledge_base`/`calculate`/`web_search`/`read_memory`/`remember` 工具 | 已完成 | `tools/registry.py` |
| `_should_use_coordinator` 启发式判断 | 已完成 | `agent_service.py` 第 73-85 行 |
| RAG 健康检查 + embedding 降级 | 已完成 | `rag_agent_service.py` |
| Synthesizer 汇总生成（带降级逻辑） | 已完成 | `synthesizer.py` |
| 工具依赖关系处理（_execute_plan 依赖拓扑排序） | 已完成（串行执行中） | `coordinator.py` |
| _fallback_plan 降级（JSON 解析失败时直接 synthesize） | 已完成 | `coordinator.py` |

---

## 附录：设计文档与代码路径索引

| 文档 | 路径 |
|------|------|
| Coordinator 设计 | `docs/coordinator-design.md` |
| Agent 设计索引 | `docs/agent-design/README.md` |
| MVP 范围 | `docs/agent-design/mvp-scope.md` |
| 状态机与工具循环 | `docs/agent-design/loop.md` |
| RAG 与 Agent 桥接 | `docs/agent-design/rag-bridge.md` |
| 可观测性与评估 | `docs/agent-design/observability.md` |
| 数据侧只读工具 | `docs/agent-design/data-tools.md` |
| 意图与对话规划 | `docs/agent-design/dialogue-planning.md` |
| 评审门禁 | `docs/agent-design/review-gate.md` |
| 工具注册 | `src/agent/tools/registry.py` |
| Coordinator 主调度 | `src/agent/core/planning/coordinator.py` |
| Planner 任务分解 | `src/agent/core/planning/planner.py`（在 coordinator.py 内） |
| Synthesizer 汇总生成 | `src/agent/core/planning/synthesizer.py` |
| WorkerResult | `src/agent/core/planning/worker_result.py` |
| PlanSchema | `src/agent/core/planning/plan_schema.py` |
| SimpleAgent 工具循环 | `src/agent/core/agent_loop.py` |
| AgentService 编排 | `src/agent/application/agent_service.py` |
| RagAgentService | `src/agent/application/rag_agent_service.py` |
| 可观测性 TraceLogger | `src/agent/core/observability.py` |
