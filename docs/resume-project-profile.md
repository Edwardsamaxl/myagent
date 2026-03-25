# 项目简历版总结（考研复试可用）

## 项目命名（建议）

**FinRAG-Agent：面向金融财报问答的可解释多阶段智能体系统**

备选名称：
- **FinDoc Copilot（RAG + Agent）**
- **A股财报智能问答 Agent（可追溯引用版）**

---

## 一句话介绍（简历可直接使用）

独立构建了一个面向金融财报场景的 RAG + Agent 系统，实现了文档摄入、混合检索、重排、证据约束生成与评估闭环，支持引用可追溯与拒答可解释，并完成多轮对话的意图路由与澄清机制。

---

## 项目定位与目标

- 场景：上市公司年报/半年报问答（财务指标、公司基本面、重大事项等）
- 目标：让回答“可检索、可引用、可解释、可复现评估”
- 方法：RAG 主链路 + Agent 编排层（意图识别、澄清、工具循环、会话状态）

---

## 我的核心工作（模块化）

### 1) 数据与 Ingestion（文档进入系统）
- 实现 PDF 转 Markdown / 文本抽取脚本，建立 `data/raw/finance/` 数据目录规范。
- 完成分块策略升级：结构分节（`##/###`）+ 句末对齐 + 重叠控制 + 参数化（`chunk_size/chunk_overlap`）。
- 设计并落地文档级元数据（`company/doc_type/date/source`）自动推导与覆盖规则。
- 完成入库统计口径统一：`total_chunks` / `deduplicated_chunks` / `dropped_duplicates`。
- 编写批量入库与切块检查脚本（`run_ingest` / `dump_chunks`），支持可复现排查。

### 2) 检索与重排（证据选取）
- 将检索从单路匹配升级为混合检索：`lexical + sparse(TF-IDF/BM25) + embedding`。
- 改造中文 tokenization（2/3-gram + 数字锚点），显著缓解自然问句 0-hit 问题。
- 引入融合策略与调参能力（`weighted_sum` / `RRF`、多路权重可配置）。
- 实现 Rerank v2：在基础分上叠加 keyword/length/metadata/numeric 规则，提升证据排序质量。
- 增强检索可解释性：输出 `score_breakdown`、zero-score 原因、trace 观测字段。

### 3) 生成与回答策略（答案质量）
- 构建“证据约束生成”流程：仅基于选中证据回答，引用编号与证据一一对应。
- 设计结构化输出模板（结论/关键依据/引用编号），并加入后处理校验。
- 实现拒答策略与 reason code 统一（`no_retrieval_hit` / `insufficient_evidence` / `citation_missing`）。
- 增加锚点覆盖、低信息片段识别、谨慎回答模板，降低“无依据硬答”风险。

### 4) Agent 编排与对话能力
- 设计并实现意图分类与澄清机制（`knowledge_corpus/tool_only/chitchat/ambiguous`），并在澄清阶段进入 `awaiting_clarification`。
- 实现 RAG 前 query 改写：基于 `rewrite_for_rag`（支持 `rule/llm/hybrid`），先进行规则归一化与轻量指代拼接，再可选 LLM 生成 `rewrite_query`，作为检索输入提升命中稳定性。
- 多轮会话记忆拆分：`sessions.json` 记录最近会话历史用于意图/上下文；`sessions_meta.json` 记录任务态与 `pending_context`，实现“待澄清 -> 补充后继续”（下一轮会把 `pending_context + user_message` 拼成 `turn_text`）。
- 实现路由策略：按 `RAG_ENABLED/use_rag` 决定是否走检索；走 RAG 时仅将 `retrieval_hits` 格式化注入用户消息以支持 grounded 生成，而最终答复仍由 `SimpleAgent` 统一产出，便于工具循环与策略兜底。
- 保留工具循环能力（`SimpleAgent` JSON tool call 协议）：解析工具请求、执行 `default_tools`（`read_memory/remember/read_skill/save_skill/read_workspace_file/get_time/calculate`），并将“工具执行结果”回写到消息流支持多步迭代。

### 5) 评估与工程化
- 建立离线评估脚本与报告体系（embedding A/B、token 对比、generation policy A/B）。
- 统一评估与追踪输出：在线用 `runtime/traces.jsonl` 按 `trace_id` 记录各阶段（如 `ingestion_*`、`rag_answer`），离线用 `runtime/day3/*`（如 query rewrite 消融、generation policy A/B 报告）量化改动效果。
- 会话与状态可复现：将消息历史落到 `runtime/sessions.json`、将澄清任务态落到 `runtime/sessions_meta.json`，支持失败用例回放与策略调参。
- 形成多 Agent 协作文档体系（职责分工、接口契约、冲突策略、阶段计划）。

---

## 阶段性结果（可在复试中陈述）

- 完成从“可跑通骨架”到“可解释 RAG Agent”的系统化升级，主链路端到端可运行。
- 在检索层通过 tokenization 与混合检索改造，解决了早期自然问句大面积 `no_retrieval_hit` 问题（10 题对照样本由全 0 命中提升为稳定命中）。
- 在生成层通过证据约束策略显著降低幻觉回答风险，并建立了可观测拒答机制与可复现实验报告。
- 完成了从文档摄入、检索重排到回答生成的完整工程闭环，具备后续产品化迭代基础。

> 注：最终简历可只保留“趋势结果 + 代表性指标”，口述时补充具体实验文件（如 `runtime/day3/*`）。

---

## 技术栈（简历写法）

- Python、Flask、JSONL、Requests
- RAG：Ingestion / Retrieval / Rerank / Grounded Generation
- 检索：Lexical、TF-IDF、BM25、Embedding、Weighted Sum、RRF
- Agent：Intent Routing、Clarification、Tool Loop、Session/Memory Store
- 工程化：可观测 Trace、离线评估 A/B、多 Agent 协作规范

---

## 简历条目（可直接粘贴）

**FinRAG-Agent（金融财报智能问答）**  
*个人项目 / Python, Flask, RAG, Agent*

- 设计并实现面向财报问答的 RAG + Agent 架构，打通“文档摄入-检索重排-证据生成-评估闭环”全流程。  
- 完成 Ingestion 模块工程化：结构化分块、元数据自动推导、去重统计、批量入库与可复现排查脚本。  
- 构建混合检索与重排策略（Lexical + TF-IDF/BM25 + Embedding + RRF/加权融合），并通过 tokenization 改造显著提升自然问句命中稳定性。  
- 实现证据约束生成与拒答机制（引用校验、锚点覆盖、低信息过滤），降低幻觉回答风险，增强答案可追溯性。  
- 搭建意图路由与澄清机制，支持“RAG/工具/闲聊”分流；建立 trace 与 A/B 评估体系，支撑迭代决策。  

---

## 复试口述（30 秒版本）

我做了一个金融财报问答 Agent。不是只做了一个聊天模型，而是完整实现了文档入库、混合检索、重排、证据约束生成和评估闭环。项目里我重点解决了两个问题：一是自然问句检索命中差，二是回答容易无依据。我通过中文 tokenization + 混合检索提高召回，再通过引用校验和拒答策略控制生成质量，最后用 A/B 报告把改动效果量化出来。

