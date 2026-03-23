# 各 Agent 职责说明

> 本文供**随时查阅与修改**：谁负责哪类改动、主要动哪些路径、与谁交接。  
> **接口与契约**以 [coordination.md](coordination.md) 为准；实现后请同步更新该文档 §3。  
> 仓库级协作底线见 [.cursor/rules/multi-agent-collaboration.mdc](../.cursor/rules/multi-agent-collaboration.mdc)。

---

## 角色总览

| 代号 | 角色 | 一句话 |
|------|------|--------|
| **PM** | 协调 / 项目总结与任务布置 | 拆任务、对齐周计划、维护 `coordination.md` 契约、裁决冲突 |
| **A** | 数据与 Ingestion | PDF/文本、切块、元数据、批量入库脚本、`ingestion.py` |
| **B** | 检索与重排 | 召回、混合、重排、检索可解释字段与配置 |
| **C** | 生成与 Agent 循环 | 生成模板、拒答/引用、`agent_loop`、工具与 RAG 编排中的「模型侧」 |
| **D** | 接口与 Web | Flask 路由、`main.py`、API 契约、内嵌前端 |
| **E**（可选） | 评估与可观测 | `evaluation` / `observability`、评估集脚本、trace 口径 |
| **F**（可选） | 解释与学习助手 | 针对用户不懂点给“结论/原理/取舍/验证/下一步”结构化讲解 |

---

## PM｜协调型 Agent

| 项目 | 内容 |
|------|------|
| **职责** | 维护 `docs/` 下 day 计划与周计划的事实一致性；把需求拆成可验收子任务；记录 API/metadata 约定；标注易冲突文件与合并顺序。 |
| **常改路径** | `docs/week1-finance-agent-plan.md`、`docs/day*-*.md`、`docs/coordination.md` |
| **避免** | 大包大揽改业务代码（可只做小修正）；不编造仓库中不存在的路径。 |
| **交接** | 契约变更**先**更新 `coordination.md` §3，再通知 A/D 实现。 |

---

## A｜数据与 Ingestion

| 项目 | 内容 |
|------|------|
| **职责** | 数据管线：`scripts/extract_pdf*.py`、`dump_chunks.py`、`run_ingest.py`；`data/raw/`、`data/README.md`；`src/agent/core/ingestion.py`；`DocumentChunk.metadata` 填充规则与 `ingest_text` / 入库调用链对齐。 |
| **常改路径** | `scripts/`、`src/agent/core/ingestion.py`、`src/agent/core/schemas.py`（字段说明）、`src/agent/application/rag_agent_service.py` 中与 **ingest_document** 相关的部分 |
| **避免** | 修改 `retrieval.py` / `rerank.py` 核心算法（除非任务仅为「适配 chunk 新字段」）。 |
| **交接** | 元数据键名、入库请求体变更 → 同步 **PM** 更新 `coordination.md`；若扩展 HTTP API → 同步 **D**。 |

---

## B｜检索与重排

| 项目 | 内容 |
|------|------|
| **职责** | `src/agent/core/retrieval.py`、`rerank.py`；检索相关环境变量与配置；命中结果的可解释信息（分数、通道等）。 |
| **常改路径** | `src/agent/core/retrieval.py`、`src/agent/core/rerank.py`、`src/agent/config.py`（仅检索相关项） |
| **避免** | 大改 `ingestion.py` 切块主逻辑；大改 `generation.py`（可提需求给 **C**）。 |
| **交接** | 依赖 `DocumentChunk` 字段时以 `schemas.py` 与 **A** 的产出为准；**先对齐字段，再动评分逻辑**。Day 3 第一优先级固定为「字段契约稳定 + 可解释输出」。 |

---

## C｜生成与 Agent 循环

| 项目 | 内容 |
|------|------|
| **职责** | `src/agent/core/generation.py`、`src/agent/core/agent_loop.py`；`src/agent/tools/`；`rag_agent_service` 中与 **生成、工具调用、prompt 策略**相关的编排（不与 ingest 抢同一 diff 时优先）。 |
| **常改路径** | `src/agent/core/generation.py`、`src/agent/core/agent_loop.py`、`src/agent/application/rag_agent_service.py`（answer/工具链）、`src/agent/tools/` |
| **避免** | 修改 PDF 脚本与数据目录结构；避免在无契约更新时改 `POST /api/ingest`。 |
| **交接** | 检索返回结构变化时先与 **B**、**PM** 对齐。 |

---

## D｜接口与 Web

| 项目 | 内容 |
|------|------|
| **职责** | `src/agent/interfaces/web_app.py`、`main.py`；HTTP API、错误码、与前端/脚本共享的 JSON 契约；内嵌 HTML/JS。 |
| **常改路径** | `src/agent/interfaces/web_app.py`、`main.py` |
| **避免** | 在路由里实现复杂检索数学；未经 **PM** 契约更新擅自删改 RAG/ingest 端点。 |
| **交接** | API 变更必须同步 **PM**（`coordination.md`）与 **A**（`run_ingest.py`）。 |

---

## E｜评估与可观测（可选）

| 项目 | 内容 |
|------|------|
| **职责** | `src/agent/core/evaluation.py`、`observability.py`；`data/eval/`、评估脚本；trace 格式与指标口径文档。 |
| **常改路径** | `src/agent/core/evaluation.py`、`src/agent/core/observability.py`、`runtime/` 相关说明（不写密钥） |
| **避免** | 改动核心业务分块与检索实现（除非任务仅为「多记字段」）。 |
| **交接** | 指标定义变更通知 **PM** 写入文档。 |

---

## F｜解释与学习助手（可选）

| 项目 | 内容 |
|------|------|
| **职责** | 解释用户在项目里不懂的概念/代码/链路；帮助用户形成可复用理解框架（结论、原理、取舍、验证、下一步）。 |
| **常改路径** | 仅解释与建议；不改代码（除非用户明确要求修改）。 |
| **避免** | 不代替 A/B/C/D 进行实现；不擅自“猜”不存在的文件或字段。 |
| **交接** | 当解释涉及接口契约（metadata keys、API 字段）时，提醒用户与 `coordination.md §3` 一致。 |

### Prompt

完整 Prompt 见 `docs/explain-agent-prompt.md`。

---

## 合并与冲突（摘要）

详细策略见 [coordination.md §5](coordination.md)。原则：**先契约、后代码**；同一文件两人改时按函数/PR 拆分。

---

## 修订记录

| 日期 | 修订说明 |
|------|----------|
| （填日期） | 初版：PM / A / B / C / D / E 职责表 |
