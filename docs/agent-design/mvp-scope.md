# Agent MVP 范围一页纸（两天内可交付边界）

> 角色：PM 定稿。本文描述「Agent 全链路」在 **约两天冲刺** 内**必须交付**与**明确不做**的范围；实现路径以仓库现有代码为准。

---

## 1. 必须做什么（MVP 内）

1. **对话入口稳定可用**  
   - 对外契约以 [coordination.md §3.8](../coordination.md) 为准：`POST /api/chat` 请求体 `message`（必填）、`session_id`（可选）；响应含 `answer`、`steps_used`、`tool_calls`、`session_id`；`RAG_ENABLED=true` 时另有 `rag` 字段，形状与 §3.9 成功响应语义对齐（见 [coordination.md §3.8](../coordination.md)）。  
   - 实现参考：`src/agent/interfaces/web_app.py`（路由）、`src/agent/application/agent_service.py`（`AgentService.chat`）。

2. **多步工具循环可跑通**  
   - `SimpleAgent` 完成「模型输出工具 JSON → 执行 → 结果写回 → 再生成」闭环，步数受 `AgentConfig.max_steps` 约束。  
   - 实现参考：`src/agent/core/agent_loop.py`、工具注册 `src/agent/tools/registry.py` 及 `src/agent/tools/` 下具体工具。

3. **与 RAG 的职责边界清晰（可测、可讲）**  
   - 纯检索问答走 `RagAgentService` / `POST /api/rag`；带工具的对话走 `AgentService` + `SimpleAgent`。  
   - 当上层在 user 消息中注入「检索证据」类上下文时，循环侧**不**用工具替代检索或重复拉取同一批证据（见 `src/agent/core/agent_loop.py` 模块说明）。  
   - RAG 编排与生成约束见 [coordination.md §3.14](../coordination.md)；生成实现主路径：`src/agent/core/generation.py`、`src/agent/core/evidence_format.py`。

4. **拒答与引用口径一致**  
   - 证据不足或无效检索时的 reason 与现有生成契约一致（如 `no_retrieval_hit`、`insufficient_evidence`），见 [coordination.md §3.14](../coordination.md)。不允许无引用硬答。

5. **契约变更可追踪**  
   - 任何对外字段名、错误码、HTTP 语义变更，须先更新 [coordination.md](../coordination.md) §3 再合代码（见 [coordination.md §3.12](../coordination.md) 与 [review-gate.md](review-gate.md)）。

---

## 2. 明确不做什么（MVP 外）

| 类别 | 说明 |
|------|------|
| 未批契约的 API 扩展 | 不在未更新 §3 的情况下新增/改名 `retrieval_hits` 主字段（§3.9 Day3 稳定性要求）、§3.8 响应顶栏位等。 |
| 检索算法大重构 | 混合检索、重排、embedding 开关等以 §3.10 与现有代码为界；MVP 不承诺「一次迭代解决所有召回问题」（已知现象见 [coordination.md §3.13](../coordination.md)）。 |
| 复杂 Agent 形态 | 多角色扮演、任意 DAG 规划器、开放互联网浏览等，不在两天 MVP 范围。 |
| Ingestion 主链路大改 | PDF/切块主逻辑、批量入库脚本的大改归属数据侧计划；Agent MVP 仅要求**消费**现有 chunk 与 metadata 契约（§3.1）。 |
| 完整可观测产品化 | trace 与评估字段可在后续迭代增强；MVP 以现有 `src/agent/core/observability.py`、`src/agent/core/evaluation.py` 与 `runtime/` 产出为基线，不强制一日内上新平台。 |

---

## 3. 验收（两天结束时 PM 可勾选）

- [ ] `POST /api/chat` 行为与 [coordination.md §3.8](../coordination.md) 描述一致（含 `RAG_ENABLED` 开/关可各验一条路径）。  
- [ ] 至少一条会话路径可稳定触发工具调用且最终返回自然语言 `answer`。  
- [ ] RAG 相关响应/拒答 reason 与 §3.14 不冲突；无「无引用硬答」回归。  
- [ ] 若改动了对外 JSON 或错误码，§3 已先更新且 [merge-decisions.md](merge-decisions.md) 无未决冲突（若有，已裁决）。

---

## 4. 相关文档与代码索引（均已存在于仓库）

| 类型 | 路径 |
|------|------|
| 契约总览 | [coordination.md](../coordination.md) §3 |
| 角色分工 | [agent-roles.md](../agent-roles.md) |
| Agent 循环 | `src/agent/core/agent_loop.py` |
| 对话服务 | `src/agent/application/agent_service.py` |
| RAG 服务 | `src/agent/application/rag_agent_service.py` |
| HTTP 路由 | `src/agent/interfaces/web_app.py` |
| 协作规则 | `.cursor/rules/multi-agent-collaboration.mdc` |

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-03-25 | PM 初版：MVP 边界与验收清单 |
