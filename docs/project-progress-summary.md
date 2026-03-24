# 项目进度总览（截至当前）

本文用于统一回答三个问题：
- 现在做到哪一步了？
- 每个模块具体改进了什么？
- 下一步的主矛盾是什么？

对应事实来源：
- 计划与契约：`docs/week1-finance-agent-plan.md`、`docs/coordination.md`
- Day1/Day2/Day3记录：`docs/day1-summary.md`、`docs/day2-daily-plan.md`、`docs/day2-final-summary.md`、`docs/day3-daily-plan.md`
- 评估产物：`runtime/day3/summary.json`、`runtime/day3/generation_policy_ab.summary.json`

---

## 1) 总体阶段判断

- **已完成阶段**：Day1（需求冻结与数据就绪）、Day2（Ingestion 收口）
- **进行中阶段**：Day3+（检索增强 + rerank + 生成约束前移）
- **当前主矛盾**：检索命中已明显改善，但生成策略偏保守，出现高拒答率，需要在“引用合规”与“可回答率”之间重新平衡。

---

## 2) 模块级进度（细化到改进点）

### A. 数据与 Ingestion（`src/agent/core/ingestion.py`、`scripts/run_ingest.py`、`scripts/dump_chunks.py`）

**已完成**
- 文本清洗 + 结构感知切块（`##/###` 分节，句末对齐，重叠控制）。
- 入库参数化：`chunk_size`、`chunk_overlap` 支持默认值与单次覆盖。
- metadata 贯通：`company/doc_type/date/source` 自动推导，支持请求体覆盖。
- 去重机制升级为可观测口径：
  - 同次 ingest 去重默认开启；
  - 跨文档去重由 `dedup_across_docs=true` 控制；
  - 输出 `total_chunks/deduplicated_chunks/dropped_duplicates`。
- 批量脚本完成可复现汇总（文档数、成功/失败、块数、失败列表）。
- 本地沙盒验证链路完善：`dump_chunks` 产出 txt/jsonl 双格式调试文件。

**状态**
- 可交接完成，契约稳定（见 `docs/coordination.md`）。

---

### B. 检索 Retrieval（`src/agent/core/retrieval.py`、`src/agent/llm/embeddings.py`）

**已完成**
- tokenization 从“粗粒度长中文串”升级为“中文 2/3-gram + 数字子 token + 英文数字 token 保留”。
- sparse 通道增强：
  - 支持 `tfidf`、`bm25`、`tfidf_bm25`（`SPARSE_MODE` 控制）。
- embedding 通道接入：
  - provider 工厂支持 `ollama/openai_compatible/mock`；
  - query/chunk 向量化与 dense cosine 评分。
- 混合融合可配置：
  - `weighted_sum` 与 `rrf`；
  - lexical/tfidf/embedding 权重可调。
- 检索可解释性增强：
  - `score_breakdown` 输出 `lexical/tfidf/bm25/sparse/embedding/fused`；
  - trace 记录 query tokens、top scores、zero-score 原因。

**阶段结果（来自 Day3 报告）**
- 命中空集问题已显著下降，top1 命中率提升（具体见 `runtime/day3/summary.json`）。

**状态**
- 主链可用，处于参数与策略持续调优阶段。

---

### C. 重排 Rerank（`src/agent/core/rerank.py`）

**已完成**
- 从早期简版升级为 Rerank v2：
  - 基础分：继承 retrieval 融合分；
  - `keyword_bonus`；
  - `length_penalty`；
  - `metadata_bonus`（year/doc_type/company 软约束）；
  - `numeric_bonus`（问题数字与文本数字对齐）。
- 通过环境变量支持规则开关，便于 A/B 排查。

**状态**
- 可用，但仍属于规则式重排，后续可评估 cross-encoder 精排。

---

### D. 生成 Generation（`src/agent/core/generation.py`、`src/agent/core/evidence_format.py`）

**已完成**
- 生成提示词强化“只基于证据回答”。
- 输出结构标准化：
  - `结论`、`关键依据`、`引用编号`。
- 引用合规校验：
  - 命中证据会分配 `[1]...[n]`；
  - 若回答无引用标记则触发拒答保护。
- 拒答语义统一：
  - `no_retrieval_hit`（无命中）
  - `insufficient_evidence`（有命中但证据不足）

**阶段结果（来自 Day3 生成策略评估）**
- 幻觉率下降明显，但拒答率偏高，出现“过度保守”（见 `runtime/day3/generation_policy_ab.summary.json`）。

**状态**
- 规则严谨但过保守，需在 Day4 做“可答性”与“可信性”平衡。

---

### E. 评估与可观测（`src/agent/core/evaluation.py`、`src/agent/core/observability.py`、`runtime/day3/*`）

**已完成**
- 在线评估记录与聚合口径统一（`total_requests/avg_latency/avg_cost/substring_match_rate`）。
- trace 事件统一化（ingestion_start/done、rag_answer）。
- Day2/Day3 评估产物体系化，支持回放与对比：
  - token 改造对比
  - embedding A/B
  - generation policy A/B
  - metadata coverage 报告

**状态**
- 指标闭环初步可用，下一步需补充更贴近任务目标的语义正确性指标（如 top1_hit、citation_ok、false_refusal）。

---

### F. Agent 基础能力（`src/agent/core/agent_loop.py`、`memory_store.py`、`session_store.py`、`skill_store.py`）

**已完成**
- 工具循环主链可运行（模型 -> 工具 -> 结果回填 -> 最终回答）。
- Memory/Session/Skill 存储能力可用。
- 与 RAG 解耦边界已写入注释，避免在工具链里重复做检索。

**状态**
- 基础可用，后续属于 Day6 的权限/风险分级与审计增强。

---

## 3) 里程碑完成度（按周计划）

- Day1：完成
- Day2：完成并收口
- Day3：核心目标完成，已前移部分 Day4
- Day4：已开工（生成约束与拒答策略），需继续优化
- Day5~Day7：待系统化推进

---

## 4) 当前风险与下一步优先级

### 主要风险
- 生成侧过度拒答（高 refusal，影响可用性）。
- 检索命中虽提升，但个别问题仍有语义漂移。
- 本地模型服务可用性依赖运行环境（端口/模型实例）。

### 下一步优先级（建议）
1. **Day4收口**：生成策略从“硬拒答”调整为“证据充分性分层回答”。
2. **Day5闭环**：统一离线评估脚本和指标门槛，固定回归流程。
3. **Day6准备**：工具权限分级、错误码统一与调用审计。

---

## 5) 一句话总结

项目已从“骨架可跑”进入“质量优化阶段”：Ingestion 与检索链路已具备工程可用性，当前核心任务是提升生成阶段的可答性与稳定性，并用统一评估口径证明改进有效。

