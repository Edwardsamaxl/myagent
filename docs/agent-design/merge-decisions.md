# Agent 设计稿合并决策表

> 角色：PM 维护。当 C/D/B/A/E 等并行产出 `docs/agent-design/` 下各稿或 PR 中出现冲突时，在此记录**听谁的**与**原因**。  
> **约束**：不引用仓库中不存在的路径。

---

## 1. 当前并稿状态（事实）

| 设计稿 | 路径 | 状态 |
|--------|------|------|
| MVP 范围 | [mvp-scope.md](mvp-scope.md) | 已落盘（PM） |
| 评审门禁 | [review-gate.md](review-gate.md) | 已落盘（PM） |
| 合并决策表 | [merge-decisions.md](merge-decisions.md) | 本文（PM） |
| 循环 / 工具 / Prompt | [loop.md](loop.md) | 已落盘 |
| API / Web | [api.md](api.md) | 草案（与 coordination §3.15 同步） |
| RAG 桥接 | [rag-bridge.md](rag-bridge.md) | 已落盘 |
| 数据与只读工具 | [data-tools.md](data-tools.md) | 已落盘 |
| 可观测与评估 | [observability.md](observability.md) | 已落盘 |
| 对话与规划 | [dialogue-planning.md](dialogue-planning.md) | 已落盘 |

**结论**：子稿均已存在；若出现字面冲突，**以本表新增裁决行为准**；与 [coordination.md](../coordination.md) §3 冲突时 **以 §3 为对外契约真理源**，并回写设计稿或本表。

---

## 2. 预置裁决原则（已实现契约优先）

| 主题 | 裁决 | 原因 / 依据 |
|------|------|-------------|
| HTTP / JSON 字段名与错误码 | **听 `coordination.md` §3 + D 落地** | §3.12 规定先文档后实现；D 主责 `src/agent/interfaces/web_app.py`（见 [agent-roles.md](../agent-roles.md)）。 |
| `retrieval_hits` 五字段主结构 | **听 §3.9** | Day3 字段稳定性要求；扩展解释字段须先批文档（§3.9 原文）。 |
| 生成与拒答、证据格式 | **听 §3.14 + C 主责实现** | C 主责 `generation.py` / `evidence_format.py`；不在路由层做证据筛选（§3.14）。 |
| 检索分数/融合/重排细节 | **听 B** | 与 §3.10、§3.13 责任分工一致；但**若**改变对外 `retrieval_hits` 形状 → 升格为 §3.9 变更，PM 门禁。 |
| Ingestion / `DocumentChunk.metadata` 键 | **听 §3.1 + A** | 元数据键名以契约表为准；A 主责 `ingestion.py` 与入库链。 |
| Agent 循环内是否用工具「代替检索」 | **听 `src/agent/core/agent_loop.py` 模块约定 + 编排层** | 模块 docstring：证据块为只读上下文，工具用于时间/计算/记忆/技能等，不替代检索。编排以 `src/agent/application/agent_service.py` / `rag_agent_service.py` 为准。 |
| 评估 trace 字段集合 | **听 §3.11 + E 提案** | 指标与记录口径变更须写入 §3 后再实现（与 [review-gate.md](review-gate.md) 一致）。 |

---

## 3. 待填：并行设计稿冲突行（有则追加）

> 若 [loop.md](loop.md)、[dialogue-planning.md](dialogue-planning.md)、[api.md](api.md) 等与另一稿或 §3 冲突，在下方追加一行；解决后勿删，可标「已关闭」。

| ID | 冲突简述 | 涉及文档 / § | 裁决（听谁） | 原因 | 状态 |
|----|----------|----------------|--------------|------|------|
| — | （暂无） | — | — | — | — |

---

## 4. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-03-25 | PM 初版：并稿状态事实 + 预置裁决表 + 待填冲突模板 |
| 2026-03-25 | 迁入 `docs/agent-design/` 并更新并稿状态为「均已存在」 |
