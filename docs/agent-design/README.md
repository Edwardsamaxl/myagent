# Agent 全链路设计稿索引

本目录集中存放 **Agent 行为、API、RAG 桥接、数据工具、可观测** 等设计文档，与 [../coordination.md](../coordination.md)（契约与事实）及 [../agent-operation-manual.md](../agent-operation-manual.md)（每日怎么干）分工如下：

| 读什么 | 解决什么问题 |
|--------|----------------|
| [../coordination.md](../coordination.md) | **对外契约**：HTTP 字段、错误码、metadata、`retrieval_hits` 形状；**谁先谁后**（§3.12）；易冲突文件（§5）。 |
| [../agent-operation-manual.md](../agent-operation-manual.md) | **开工顺序**：先读 coordination、再读当日 day 计划、再动代码。 |
| 本目录各稿 | **怎么做更合理**：状态机、Prompt 分层、chat 扩展草案、评估 trace 等；**落地前**须与 §3 对齐并经 PM 门禁。 |

## 阅读顺序建议

1. [mvp-scope.md](mvp-scope.md) — MVP 必须做 / 不做（两天边界）  
2. [review-gate.md](review-gate.md) — 契约变更先改 coordination §3 再放行  
3. [merge-decisions.md](merge-decisions.md) — 多稿冲突时听谁的  
4. 按角色选读：[loop.md](loop.md)（C）、[rag-bridge.md](rag-bridge.md)（B/C）、[api.md](api.md)（D）、[data-tools.md](data-tools.md)（A）、[observability.md](observability.md)（E）、[dialogue-planning.md](dialogue-planning.md)（C/D，进阶）

## 文件一览

| 文件 | 说明 |
|------|------|
| [mvp-scope.md](mvp-scope.md) | MVP 范围一页纸 |
| [review-gate.md](review-gate.md) | 设计评审门禁（对齐 §3.12） |
| [merge-decisions.md](merge-decisions.md) | 合并决策表 |
| [loop.md](loop.md) | 状态机、工具循环、Prompt 分层 |
| [dialogue-planning.md](dialogue-planning.md) | 意图、澄清、显式规划 |
| [rag-bridge.md](rag-bridge.md) | Agent 与 RAG 编排路由 |
| [api.md](api.md) | HTTP/chat 扩展与错误码（草案，见 coordination §3.15） |
| [data-tools.md](data-tools.md) | 数据侧只读工具与路径约定 |
| [observability.md](observability.md) | trace / 评估记录设计 |

---

修订：2026-03-25 自 `docs/agent-design-*.md` 平铺迁入本目录并统一索引。
