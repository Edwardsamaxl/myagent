# Day 2 最终工作总结（统一口径版）

本文是 Day 2 的唯一汇总版本，用于替代多份 `*delivery*.md` 分散记录。  
目标是把“当天完成、证据产物、遗留风险、次日交接”放到同一口径，避免 Day 标注混乱。

---

## 1. Day 2 当天目标与完成结论

Day 2 目标：完成 Ingestion 可交接改造（元数据、参数化、入库可观测），并为 Day 3 检索优化提供可复现实验基线。  
最终结论：**Day 2 主任务已完成，可收口；保留 1 项 P1 遗留风险（模型端点可用性）。**

---

## 2. 当天已完成（按模块归并）

### A. 数据与 Ingestion（主线完成）

- `DocumentChunk.metadata` 贯通，至少包含：`company/doc_type/date/source`。
- `POST /api/ingest` 支持可选：`metadata`、`chunk_size`、`chunk_overlap`、`dedup_across_docs`。
- 入库统计口径统一为：
  - `total_chunks`（去重前）
  - `deduplicated_chunks`（去重后）
  - `dropped_duplicates`（去重丢弃）
- `scripts/run_ingest.py` 可输出可复现汇总（文档数、成功/失败、块数统计、失败列表）。

### D. 接口与错误语义（完成）

- `/api/ingest` 参数校验完整，非法参数返回 `code=invalid_chunk_params`。
- `/api/rag` 上游异常统一返回 JSON 错误体（不再是 HTML 错误页），并映射到机器可读错误码。

### E. 评估与可观测（完成）

- 完成 `run_ingest -> /api/rag -> trace/eval` 的可复现闭环。
- 产出 Day 2 对照评估：
  - `runtime/day2_query_rewrite_compare.json`
  - `runtime/day2_extended_eval_compare.json`
- 结论明确：摄入链路可用；主要问题在检索问法鲁棒性与召回排序质量。

---

## 3. 超前完成（Day 3 预支成果，已落地）

> 以下内容不作为 Day 2 必选验收项，但属于你当天提前完成的有效产出，已并入正式记录。

### B. 检索与重排（预支完成）

- 完成中文关键词匹配重写（tokenization 改造）：
  - 中文连续串不再以超长原 token 直接匹配；
  - 引入中文 bi-gram + tri-gram；
  - 英文/数字 token 保留。
- 新增 embedding 检索模块与混合召回最小可用实现：
  - 新增 embedding 客户端封装（`src/agent/llm/embeddings.py`）；
  - 检索支持 `sparse_only` 与 `hybrid`；
  - 融合策略支持 `weighted_sum` 与 `rrf`（配置化）。
- 保持 `/api/rag` 主字段契约不变，仅增强检索解释能力与调试信息。

### 对应证据产物

- `runtime/day3/embedding_ab_report.quick.json`
- `runtime/day3/summary.json`
- `runtime/INDEX.json`

---

## 4. 关键验收证据（Day 2）

- 参数覆盖对比：默认与覆盖 `chunk_size/chunk_overlap` 均可生效，且仅影响单次请求。
- 批量入库汇总可复现（文档数、成功/失败、总块数）。
- 抽样可追溯：chunk 可映射到 `company/source`。
- Day 2 已完成交接前置：为 Day 3 固定 `retrieval_hits` 主字段契约。

---

## 5. Day 2 遗留风险（收口时保留）

1. **[P1][D 负责] 模型端点可用性不稳定**  
   历史观测到 `/api/rag` 在上游模型端点异常时失败；虽已补错误语义映射，仍需持续回归验证（404/超时/连接失败场景）。

2. **检索鲁棒性不足（已移交 Day 3）**  
   自然问句 `no_retrieval_hit` 概率偏高，属于检索策略问题，不再归入 Day 2 未完成项。

---

## 6. 交接到 Day 3（不再计入 Day 2 完成度）

以下内容虽然有文档产出，但应归类为 Day 3/Day 4 交接信息，不计入 Day 2 交付完成度：

- embedding 通道与混合召回实现细节（Day 3）。
- 生成侧引用/拒答稳定化实现细节（Day 4）。

Day 2 只保留其“前置条件”结论：契约稳定、评估基线可复现、风险与责任人明确。

---

## 7. 最终口径（供周报/复盘直接引用）

- Day 2 已完成 Ingestion 与接口契约收口，入库链路、参数覆盖、统计口径、错误语义达到可交接状态。
- Day 2 已形成可复现评估基线，并明确检索问题是下一阶段主矛盾。
- Day 2 保留 1 项 P1 风险（模型端点可用性回归，D 负责），其余进入 Day 3/Day 4 正常迭代。

