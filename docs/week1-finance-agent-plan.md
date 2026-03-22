# 第1周执行计划（金融领域 RAG + Agent）

本文档目标：把你当前项目拆成“每天可完成、可验证、可讲述”的小步任务。  
原则：每天都有**代码产出 + 指标记录 + 3分钟口述复盘**。

---

## 0. 你当前已经有的东西（现状盘点）

你已经有一套很好的基础骨架，重点能力如下：

- 基础 Agent 循环：`model -> tool call -> tool result -> final answer`
- Web UI + API：可聊天、切换模型、编辑 `MEMORY.md`、读写 `skills`
- 会话与长期记忆：`runtime/sessions.json`、`workspace/MEMORY.md`
- RAG 骨架模块（可运行占位版）：
  - `core/ingestion.py`
  - `core/retrieval.py`
  - `core/rerank.py`
  - `core/generation.py`
  - `core/evaluation.py`
  - `core/observability.py`
  - `application/rag_agent_service.py`
- 新增了 RAG 相关接口：
  - `POST /api/ingest`
  - `POST /api/rag`
  - `GET /api/metrics`

结论：你不是从 0 开始，而是从“可跑通框架”开始进入“质量提升阶段”。

---

## 1. 第1周总体目标（只做一件事）

只聚焦一个场景：**金融文档问答（带引用）**。  
第1周完成标准（MVP）：

1. 能导入 20~50 篇文档（公告/研报摘录）；
2. 问答输出必须有引用来源；
3. 有最小评估表（准确/引用正确/拒答是否合理/耗时）；
4. 能说明每次优化改了什么、为什么有效。

---

## 2. 第1周每日计划（Day 1 ~ Day 7）

## Day 1：需求冻结 + 数据集初始化

### 目标
- 把任务边界固定，避免边做边漂移。

### 任务
- 定义问题类型（例如：公司基本面、财务指标、风险提示、事件时间线）。
- 建立一个小型数据目录（建议 `data/raw/finance/`）。
- 准备 20~50 份文本（先用 txt/md，后续再接 pdf/html）。
- 写 30 条问答样例（问题 + 参考答案 + 证据来源）。

### 当日产出
- `docs/day1-requirements.md`
- `data/raw/finance/*`
- `data/eval/week1_eval_set.jsonl`（建议格式：`question/expected_answer/source`）

### 验收标准
- 你能用 3 分钟解释：这个 Agent 解决什么，不解决什么。

---

## Day 2：Ingestion 模块改造（清洗/切块/元数据）

### 目标
- 让文档进入系统时结构稳定，便于后续检索。

### 任务
- 在 `src/agent/core/ingestion.py` 增加：
  - 文档级元数据：`doc_type/company/date/source`
  - 切块策略参数化：`chunk_size/chunk_overlap`
  - 去重日志（哪些块被去重）
- 新增一个离线脚本（建议 `scripts/run_ingest.py`）批量入库。

### 当日产出
- ingestion 版本说明（参数、输入输出）
- 一次入库结果统计（文档数、块数、去重率）

### 验收标准
- 你能回答：为什么 `chunk_size=500, overlap=80` 是当前选择。

---

## Day 3：Retrieval 从占位走向可解释

### 目标
- 把“检索命中”变得可看、可调、可对比。

### 任务
- 在 `src/agent/core/retrieval.py` 明确拆分两路分数：
  - lexical（关键词）
  - semantic（语义）
- 增加融合策略参数（例如权重或 RRF）。
- 为每条检索结果输出解释字段：
  - 命中关键词
  - 各路分数
  - 最终融合分数

### 当日产出
- 检索调试输出（TopK 命中明细）
- 一张小表：只 lexical / 只 semantic / hybrid 对比

### 验收标准
- 你能给出“为什么混合检索比单路稳”的本项目证据。

---

## Day 4：Rerank + 回答约束

### 目标
- 降低“看起来合理但没有依据”的回答。

### 任务
- 在 `src/agent/core/rerank.py` 加更清晰的 rerank 输入输出结构。
- 在 `src/agent/core/generation.py` 强制回答模板：
  - 结论
  - 关键依据
  - 引用编号
  - 证据不足时拒答
- 给拒答添加固定 reason code（如 `no_hit/low_conflict/insufficient_evidence`）。

### 当日产出
- 10 条问题的“改造前后”回答对比
- 拒答案例 3 条（并解释为何拒答）

### 验收标准
- 不允许无引用硬答；拒答要可解释。

---

## Day 5：评估闭环（离线 + 线上记录）

### 目标
- 所有优化都有数据支撑，而不是感觉。

### 任务
- 在 `src/agent/core/evaluation.py` 增加最小指标：
  - `answer_match`（可先人工 0/1）
  - `citation_ok`（引用是否对应）
  - `refusal_ok`（该拒答时是否拒答）
  - `latency_ms`
- 在 `src/agent/core/observability.py` 规范 trace 字段：
  - `trace_id/session_id/query/model/retrieval_count/latency`
- 增加一个简易评估脚本（建议 `scripts/eval_week1.py`）。

### 当日产出
- `docs/week1-metrics.md`（第一次基线数据）

### 验收标准
- 能复现一次评估并得到同口径指标。

---

## Day 6：工具链和权限边界（Agent 可靠性）

### 目标
- 保证 Agent 调工具“可控、可审计、可回放”。

### 任务
- 梳理 `src/agent/tools/registry.py`：
  - 给每个工具加风险级别（read_only / write_limited）
  - 统一参数校验与错误码
- 在 `src/agent/application/agent_service.py` 增加工具调用日志聚合（每轮执行记录）。
- 增加“禁用高风险工具”配置开关（例如 env）。

### 当日产出
- 工具白名单清单（文档）
- 5 条工具调用回放日志样例

### 验收标准
- 你能清楚说出：哪些工具默认不开，为什么。

---

## Day 7：收敛整理 + 面试表达材料

### 目标
- 把这周成果变成“可展示、可解释、可复用”的资产。

### 任务
- 整理架构图（当前版）
- 汇总实验表（改动、指标变化、结论）
- 写“失败案例复盘”至少 3 条
- 更新 README 的 Week1 章节

### 当日产出
- `docs/week1-review.md`
- 演示脚本（5 分钟）
- 口述提纲（3 分钟）

### 验收标准
- 脱离代码能完整讲清：链路、指标、取舍、下一步。

---

## 3. 你可以逐个改的“模块学习地图”

下面是“当前实现 -> 可以改进”的对照，你可以按顺序改，难度递增。

### A. Ingestion（`src/agent/core/ingestion.py`）
- 当前：文本清洗 + 固定切块 + hash 去重（MVP）
- 改进方向：
  - 支持 PDF/HTML/表格抽取
  - 按标题层级切块（语义切块）
  - 元数据标准化（公司、日期、报告类型）

### B. Retrieval（`src/agent/core/retrieval.py`）
- 当前：内存版混合检索（简化 lexical + tfidf）
- 改进方向：
  - 接入真实向量库（FAISS/Milvus/pgvector）
  - BM25 独立通道 + RRF 融合
  - 可学习权重与 query 分类路由

### C. Rerank（`src/agent/core/rerank.py`）
- 当前：规则式重排
- 改进方向：
  - cross-encoder reranker
  - 动态 top_n（按 query 难度）
  - rerank 置信分作为拒答依据

### D. Generation（`src/agent/core/generation.py`）
- 当前：证据拼接 + 引用约束提示
- 改进方向：
  - 严格结构化输出（JSON schema）
  - 引用位置对齐（句子级）
  - 数值核对（财务数字一致性检查）

### E. Evaluation（`src/agent/core/evaluation.py`）
- 当前：在线记录与均值汇总
- 改进方向：
  - 离线评估任务流水线
  - 自动错误分类（检索错/生成错/工具错）
  - 指标看板（按问题类型拆分）

### F. Observability（`src/agent/core/observability.py`）
- 当前：jsonl trace
- 改进方向：
  - Prompt/version 追踪
  - 每阶段耗时占比
  - 接 OpenTelemetry / Langfuse

### G. Orchestration（`src/agent/application/rag_agent_service.py`）
- 当前：串行链路编排
- 改进方向：
  - 路由策略（简单问答直答，复杂问题走检索）
  - 并发检索 + 超时回退
  - 多模型分级（便宜模型先筛选）

---

## 4. 每天固定执行模板（防止 vibe coding）

每天开始前先写 10 分钟：

1. 今天改哪个模块？
2. 输入输出是什么？
3. 成功指标是什么？
4. 失败场景是什么？

每天结束后写 10 分钟：

1. 改了什么（最多 3 条）  
2. 指标变了什么（必须有数）  
3. 为什么会这样（你的解释）  
4. 明天先做什么

---

## 5. 本周不做（避免分散）

- 不做多智能体编排
- 不做复杂前端重构
- 不做云端分布式部署
- 不做高风险自动执行工具

先把单机场景打透，收益最高。

