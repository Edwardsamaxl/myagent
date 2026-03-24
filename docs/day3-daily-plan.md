# Day 3 当日计划（Day3+：并入原 Day4 的 Rerank 与回答约束）

本文档是 Day 3 当天执行单，按阶段 0->3 推进。  
依赖文档：`docs/coordination.md`、`docs/day2-daily-plan.md`、`docs/day2-ingestion-notes.md`。

---

## 1. 今日目标（一句话）

在不破坏现有 API 契约前提下，完成 Day3 检索增强收口，并前移执行原 Day4 的 `rerank + 回答约束` 核心任务。

---

## 2. 执行顺序（今天按此推进）

### 阶段 0（PM，30-45 分钟）：先更新契约

- [x] 在 `docs/coordination.md` §3 增加 embedding/融合配置契约、评估记录口径。
- [x] 明确 `retrieval_hits` 既有字段不改名；解释字段扩展需先文档批准。
- [x] 明确“先文档、后实现”放行规则仍生效。

验收标准：
- 任何 Agent 在编码前能从 `coordination.md` 读到完整 Day3 契约。

不要做什么：
- 不在该阶段修改检索算法实现。

### 阶段 1（B，1-1.5 小时）：检索增强收口（快速封板）

- [ ] 固化 embedding + lexical 双通道配置（provider/model/fusion/top_k/weights）。
- [ ] 对照 `runtime/day3/summary.json` 与 `runtime/day3/day3_embedding_ab_report.full.summary.json` 回填“当前最佳配置”。
- [ ] 保留 baseline 回退开关，避免后续 rerank 联调受阻。

验收标准：
- `retrieval_hits` 主字段保持不变：`chunk_id/score/source/metadata/text_preview`。
- 在至少一批评估题上，出现“baseline 无命中，hybrid 有命中”的样本。

不要做什么：
- 不改路由层做检索计算。
- 不引入大范围索引重构或向量库迁移。

### 阶段 2（B + C，2 小时）：前移原 Day4（Rerank + 回答约束）

- [ ] 在 `src/agent/core/rerank.py` 明确 rerank 输入输出结构（可解释、可调试）。
- [ ] 在 `src/agent/core/generation.py` 强制回答模板：
  - 结论
  - 关键依据
  - 引用编号
  - 证据不足时拒答
- [ ] 统一拒答 reason code 口径：`no_retrieval_hit` / `insufficient_evidence`（保留现有兼容语义）。

验收标准：
- 不允许无引用硬答；拒答可解释且 reason code 稳定。
- `retrieval_hits` 主字段与 `/api/rag` 顶层字段保持兼容。

不要做什么：
- 不重写检索算法主链路（只做必要 rerank/生成侧约束）。
- 不扩展 API 主字段名。

### 阶段 3（E，1-1.5 小时）：离线评估对比（含 rerank/生成差异）

- [ ] 跑 `data/eval/week1_eval_set.jsonl`，比较 baseline vs hybrid（可附带 rerank/模板约束前后）。
- [ ] 输出核心指标：`retrieved_zero_rate`、`top1_hit_rate`、`refusal_rate`、`avg_latency_ms`、`citation_ok`。
- [ ] 至少 5 条失败案例诊断（表达差异/分词/年份数字等）。

验收标准：
- 生成可复现评估文件（`runtime/*.json`）与命令记录。

不要做什么：
- 不改检索实现，只做评估与诊断。

### 阶段 4（C + D，并行约 1 小时）：输出语义与接口稳定

- [ ] C：区分“检索为空”与“有命中但证据不足”的拒答路径展示。
- [ ] D：校验 `/api/rag` 契约稳定与错误码语义，不改主字段名。

验收标准：
- 用户可从响应/日志区分两类拒答原因。
- 接口字段保持向后兼容。

不要做什么：
- C 不改 retrieval/rerank 算法。
- D 不在路由层写检索算法。

### 发布顺序（串并行视图，按此派发）

1. **PM 先发（串行）**：先锁定 `coordination.md` 契约与验收口径。  
2. **B 第二个发（串行主线）**：先做检索增强收口，再进入 rerank 结构化。  
3. **A 与 B 并行**：A 提供 metadata 覆盖率与表达差异词，喂给 B。  
4. **C 在 B 给出 rerank 输入输出后接入（串行依赖）**：落地回答模板与拒答约束。  
5. **E 在 B+C 可跑后发（串行依赖）**：跑对比评估并出失败诊断。  
6. **D 最后并行回归**：接口字段与错误码稳定性检查。

---

## 3. 角色分工（今日）

- **PM（协调）**：维护 `coordination.md` 契约、收口验收标准。
- **A（数据）**：输出 metadata 覆盖率检查；提供表达差异词样本给 B。
- **B（检索）**：收口 embedding 混合召回 + 承接 rerank 输入输出结构。
- **C（生成）**：回答模板与拒答约束（并保持契约兼容）。
- **D（接口）**：契约稳定与错误语义一致性。
- **E（评估）**：A/B 对比报告与失败诊断固化。

---

## 4. 今日验收标准（晚上收口按这 4 条）

1. `retrieved=0` 相比 baseline 明显下降（建议目标：下降 30%+）。
2. 至少 5 条评估题从“检索为空”变为“有命中”。
3. 不允许无引用硬答；`citation_ok` 有记录且可抽样核查。
4. API 主字段契约不变（`chunk_id/score/source/metadata/text_preview`）。
5. 有可复现对比结果（命令、配置、指标、样例、trace）。

---

## 5. 今日验收记录（执行后回填）

- 执行时间：
- baseline 策略版本：
- hybrid 策略版本：
- 融合策略与权重：
- 指标对比（`retrieved_zero_rate` / `top1_hit_rate` / `refusal_rate` / `avg_latency_ms`）：
- 引用指标（`citation_ok`）：
- “空检索 -> 有命中”样例（至少 5 条）：
- 失败案例诊断（至少 5 条）：
- 遗留风险（最多 3 条）：

---

## 6. 风险与责任人（B/D/E）

- **B（检索主责）**：embedding 通道可用但效果不稳定；风险是语义召回引入噪声导致 top1 下降。  
  - 缓解：先保留 baseline 回退开关，按 `fusion_strategy/weight` 做小步调参并记录版本。
- **B/C（联合风险）**：rerank 与生成模板联调时，可能出现“有命中但引用弱关联”。
  - 缓解：先固定 rerank 输出结构，再让 C 消费；评估中强制抽样 `citation_ok`。
- **D（接口主责）**：实现期若误改 `/api/rag` 主字段或错误码语义，会破坏联调。  
  - 缓解：任何接口字段/错误码变化先改 `coordination.md` §3，再放行代码。
- **E（评估主责）**：评估口径不一致会导致“优化有效性”不可比较。  
  - 缓解：统一输出 `retrieved_zero_rate/top1_hit_rate/refusal_rate/avg_latency_ms` 与策略版本字段。

