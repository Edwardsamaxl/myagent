# Day 3 当日计划（Embedding + 双通道混合召回）

本文档是 Day 3 当天执行单，按阶段 0->3 推进。  
依赖文档：`docs/coordination.md`、`docs/day2-daily-plan.md`、`docs/day2-ingestion-notes.md`。

---

## 1. 今日目标（一句话）

在不破坏现有 API 契约前提下，引入 embedding 语义召回通道，并与 lexical/TF-IDF 形成可配置的双通道混合召回。

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

### 阶段 1（B，2-3 小时）：最小可用 embedding 通道

- [ ] 新增 embedding 客户端封装（本地模型或 API）。
- [ ] 入库后检索侧可消费 chunk 向量（可按需生成/缓存，不改 ingestion 主切块逻辑）。
- [ ] 增加语义相似度 top_k，并与 lexical 通道融合（先 `weighted_sum` 或 `RRF`）。
- [ ] 输出可解释日志：各通道分数 + 融合分数 + 命中来源。

验收标准：
- `retrieval_hits` 主字段保持不变：`chunk_id/score/source/metadata/text_preview`。
- 在至少一批评估题上，出现“baseline 无命中，hybrid 有命中”的样本。

不要做什么：
- 不改路由层做检索计算。
- 不引入大范围索引重构或向量库迁移。

### 阶段 2（E，1-1.5 小时）：离线评估对比

- [ ] 跑 `data/eval/week1_eval_set.jsonl`，比较 baseline vs hybrid。
- [ ] 输出核心指标：`retrieved_zero_rate`、`top1_hit_rate`、`refusal_rate`、`avg_latency_ms`。
- [ ] 至少 5 条失败案例诊断（表达差异/分词/年份数字等）。

验收标准：
- 生成可复现评估文件（`runtime/*.json`）与命令记录。

不要做什么：
- 不改检索实现，只做评估与诊断。

### 阶段 3（C + D，并行约 1 小时）：输出语义与接口稳定

- [ ] C：区分“检索为空”与“有命中但证据不足”的拒答路径展示。
- [ ] D：校验 `/api/rag` 契约稳定与错误码语义，不改主字段名。

验收标准：
- 用户可从响应/日志区分两类拒答原因。
- 接口字段保持向后兼容。

不要做什么：
- C 不改 retrieval/rerank 算法。
- D 不在路由层写检索算法。

---

## 3. 角色分工（今日）

- **PM（协调）**：维护 `coordination.md` 契约、收口验收标准。
- **A（数据）**：输出 metadata 覆盖率检查；提供表达差异词样本给 B。
- **B（检索）**：实现 embedding 通道与混合召回（最小可用）。
- **C（生成）**：拒答语义分流展示。
- **D（接口）**：契约稳定与错误语义一致性。
- **E（评估）**：A/B 对比报告与失败诊断固化。

---

## 4. 今日验收标准（晚上收口按这 4 条）

1. `retrieved=0` 相比 baseline 明显下降（建议目标：下降 30%+）。
2. 至少 5 条评估题从“检索为空”变为“有命中”。
3. API 主字段契约不变（`chunk_id/score/source/metadata/text_preview`）。
4. 有可复现对比结果（命令、配置、指标、样例）。

---

## 5. 今日验收记录（执行后回填）

- 执行时间：
- baseline 策略版本：
- hybrid 策略版本：
- 融合策略与权重：
- 指标对比（`retrieved_zero_rate` / `top1_hit_rate` / `refusal_rate` / `avg_latency_ms`）：
- “空检索 -> 有命中”样例（至少 5 条）：
- 失败案例诊断（至少 5 条）：
- 遗留风险（最多 3 条）：

