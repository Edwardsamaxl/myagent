# Day 2 当日计划（4 小时冲刺版）

本文档供**当日协作的 agent**快速对齐：今天要做什么、做到什么算完成、与现有代码的关系。  
总纲见 [week1-finance-agent-plan.md](week1-finance-agent-plan.md) 中「Day 2」一节；Day 1 落地情况见 [day1-summary.md](day1-summary.md)；**契约与冲突策略**见 [coordination.md](coordination.md)。

---

## 1. 今日目标（一句话）

在剩余约 4 小时内，把 Day 2 从“已可用”推进到“可交接”：完成参数覆盖联调、跑出可复现验收记录、并把 Day 3 依赖前置清理完。

---

## 2. 与周计划 Day 2 的对应关系

| 周计划要求 | 说明 |
|------------|------|
| 文档级元数据 `doc_type` / `company` / `date` / `source` | 已由 `derive_doc_metadata_from_source(source)` 写入每条 chunk 的 `metadata`（可选请求体 `metadata` 覆盖）；规则见 [data/README.md](../data/README.md) |
| 切块参数 `chunk_size` / `chunk_overlap` 参数化 | `DocumentIngestionPipeline` 已支持构造参数；需确认 API/脚本是否可传入并写入说明 |
| 去重日志 | 当前实现**不做内容 hash 去重**（`IngestionResult.deduplicated_chunks` 与 `total_chunks` 相同）；需二选一：**恢复可观测的去重**并打日志，或在文档中**明确「无去重」及统计口径**，避免与周计划措辞冲突 |
| 离线批量入库脚本 | `scripts/run_ingest.py` 已输出**汇总**（扫描文档数、成功/失败、总块数、失败列表；任一则非 0 退出码 1） |

---

## 3. 当前状态（已完成 / 待完成）

已完成：
- [x] `DocumentChunk.metadata` 贯通（`company/doc_type/date/source` 约定与覆盖规则已在 `coordination.md` 维护）。
- [x] `run_ingest.py` 已有汇总输出，且失败时可返回非 0 退出码。
- [x] 去重口径已统一为“无内容 hash 去重”，`deduplicated_chunks == total_chunks`。

待完成（今天剩余 4 小时）：
- [x] 跑完一次**参数覆盖联调**并固化记录（默认参数 vs 单次覆盖参数）。
- [x] 产出 Day 2 **验收纪要**（含命令、统计、结论、风险）。
- [x] 做 Day 3 的前置准备（检索解释字段最小契约，不改检索算法本体）。

---

## 4. 剩余 4 小时冲刺任务（含依赖顺序）

### 批次 A（建议 1.5 小时）：参数覆盖联调

依赖：当前 `POST /api/ingest` 支持可选 `chunk_size/chunk_overlap`（见 `coordination.md` §3.2）。

任务：
- [x] 用同一份文档跑两次 ingest：一次默认参数、一次覆盖参数（例如更小 `chunk_size`）。
- [x] 记录两次 `total_chunks` 差异，验证覆盖仅影响单次请求，不污染进程默认配置。
- [x] 将结果补写到本文「验收记录」。

验收标准：
- 返回 200，且覆盖参数请求的 `total_chunks` 与默认参数有可解释差异。
- 非法参数（如 `chunk_overlap >= chunk_size`）返回 400 且 `code=invalid_chunk_params`。

不要做什么：
- 不在本批次改检索、重排、生成代码。
- 不新增与 ingest 无关路由。

### 批次 B（建议 1 小时）：批量入库验收与样本核对

依赖：批次 A 完成。

任务：
- [x] 运行 `scripts/run_ingest.py` 完成一次全量入库。
- [x] 抽样 2~3 个 chunk，确认 `metadata.company` 与 `source` 可追溯。
- [x] 把“文档数、成功数、失败数、总块数”写入本文「验收记录」。

验收标准：
- 批量脚本输出完整汇总行；失败列表为空或有明确原因。
- 抽样结果能证明 metadata 已进入检索链路可用字段。

不要做什么：
- 不临时改数据目录结构。
- 不手工改 debug 文件来“伪造”结果。

### 批次 C（建议 1.5 小时）：Day 3 前置契约清理（不改算法）

依赖：批次 A/B 完成。

任务：
- [x] 在 `coordination.md` 明确 `retrieval_hits` 最小可解释字段（至少 `chunk_id/score/source/metadata/text_preview`）。
- [x] 在本文补一节「Day 3 启动条件」，给检索 Agent 可直接接手的输入输出定义。
- [x] 在 `agent-roles.md` 对 B 角色加一句“先对齐字段，再动评分逻辑”。

验收标准：
- Day 3 接手方无需再问“字段名/来源/口径”，即可开始实现。
- 新增文档内容不与现有代码或 `coordination.md` 冲突。

不要做什么：
- 不实现 BM25/RRF/向量库接入。
- 不改 `retrieval.py` 评分公式与权重。

---

## 5. 给其他 agent 的代码入口（勿盲改）

| 位置 | 作用 |
|------|------|
| `src/agent/core/ingestion.py` | 清洗、分块、`DocumentChunk` 组装 |
| `src/agent/core/schemas.py` | `DocumentChunk.metadata` 定义 |
| `scripts/run_ingest.py` | 批量调用 `/api/ingest` |
| 应用层与路由 | 搜索 `ingest` / `DocumentIngestionPipeline` 的调用处，改 API 时一并调整 |

---

## 6. 当日验收（对齐周计划）

1. 完成默认参数与覆盖参数的对比入库记录（含返回值与解释）。
2. 批量入库后产出**文档数/成功数/失败数/总块数**四项统计。
3. 至少 2 条抽样命中显示 `metadata`（含 `company`）可追溯。
4. Day 3 接手契约已文档化（字段和边界清晰）。

---

## 7. 今日刻意不做（防范围膨胀）

- 不接向量库、不改混合检索主逻辑（属 Day 3 及以后）。
- 不重写前端；除非 ingest API 契约变更所必需。

---

## 8. 今日验收记录（执行后补）

> 由执行中的 Agent 回填，不留空口径。

- 执行时间：2026-03-23 13:58（本地）
- 参数对比：默认参数结果 / 覆盖参数结果：
  - 默认（`chunk_size=500, chunk_overlap=80`）：`POST /api/ingest` 返回 200，`total_chunks=757`。
  - 覆盖（`chunk_size=300, chunk_overlap=60`）：返回 200，`total_chunks=1231`。
  - 非法参数（`chunk_size=200, chunk_overlap=200`）：返回 400，`code=invalid_chunk_params`。
  - 结论：覆盖参数仅影响本次请求，且与默认参数存在可解释块数差异（更小 `chunk_size` -> 更多 chunk）。
- 批量入库汇总（文档数/成功数/失败数/总块数）：
  - 文档数 2，成功 2，失败 0，总块数 1321（`scripts/run_ingest.py` 汇总行）。
- 抽样验证（chunk_id -> company/source）：
  - `sample_meta-0000 -> company=贵州茅台, source=贵州茅台/年报_2024.md`
  - `sample_meta-0031 -> company=贵州茅台, source=贵州茅台/年报_2024.md`
  - `sample_meta-0267 -> company=贵州茅台, source=贵州茅台/年报_2024.md`
- 遗留风险（最多 3 条）：
  - [P1][指派 D] `/api/rag` 在模型端点不可用时出现过 500（观测到 `http://localhost:11434/api/chat` 404 场景）；D 需完成接口兜底核查与回归，确保该类上游失败按契约返回机器可读错误码（见 `coordination.md` §3.6）。
  - 批量脚本目前只处理 `.md/.txt`，PDF 需先离线抽取再入库（符合当前流程设计）。

### 在线验收补充（模型端点恢复后复测）

- 执行时间：2026-03-23 14:06~14:08（本地）
- 可复现命令：
  - `python scripts/run_ingest.py`
  - `Invoke-RestMethod -Uri "http://127.0.0.1:7860/api/rag" -Method POST -ContentType "application/json; charset=utf-8" -Body '{"question":"<样本问题>","top_k":3}'`
- 批量入库统计（脚本标准汇总口径）：
  - 扫描文档数 2，成功 2，失败 0，总块数 1321。
- `/api/rag` 抽样（3 条）：
  - 样本 1（评估集问题）：HTTP 200，`trace_id=fa4507f8983a`，`retrieval_hits=0`。
  - 样本 2（评估集问题）：HTTP 200，`trace_id=f1c8bcded305`，`retrieval_hits=0`。
  - 样本 3（评估集问题）：HTTP 200，`trace_id=a3c2a4c4de72`，`retrieval_hits=0`。
  - 改用短关键词（如 `600519`）继续抽样时，`POST /api/rag` 返回 HTTP 500。
- 错误码与根因（按现有可观测证据）：
  - HTTP 错误码：`500`（`POST /api/rag`）。
  - 上游根因（终端 traceback）：`requests.exceptions.HTTPError: 404 Client Error: Not Found for url: http://localhost:11434/api/chat`。
  - 结论：当前模型端点未稳定可用（或模型 API 路径/提供商配置不一致），导致「命中后进入生成阶段」的在线抽样无法完成；本次仅确认入库统计可复现，`retrieval_hits.metadata` 在线追溯待模型端点恢复后重跑。

### E（评估与可观测）补充结论（2026-03-23）

- 执行概览：
  - 已完成 `run_ingest -> /api/rag 抽样 -> trace/eval 汇总` 闭环验证，产出 `runtime/day2_small_batch_eval.json`、`runtime/day2_query_rewrite_compare.json`、`runtime/day2_extended_eval_compare.json`。
  - 其中 `day2_extended_eval_compare.json` 为 12 题对照：**raw 问法命中率 0.0，rewrite 问法命中率 1.0**；top3 `source` 命中目标文档集合率分别为 0.0 / 1.0。
- 关键判断：
  - 「摄入坏了」不成立：已用最小复现实验验证同一条新摄入文本可被命中（见 `trace_id=09df5fa69f4b`）。
  - 当前主问题是检索侧问法鲁棒性不足：自然问句（如“...是多少？”）在现有切词口径下经常 `no_retrieval_hit`。
  - 次问题是召回偏：部分财务题虽命中目标文档，但 top3 语义落到“股东大会”等无关块。
- 交接与状态：
  - [x] Day 2（E 侧）任务完成：问题已可复现、可量化、可回归。
  - [ ] 已交接 Agent B：优先处理检索切词/召回排序鲁棒性（不改接口契约字段）。

## 9. Day 3 启动条件（给检索 Agent 的最小契约）

Day 3 第一优先级：
- **字段契约稳定 + 可解释输出**（先稳定字段，再迭代算法）。
- B（检索与重排 Agent）按 [agent-roles.md](agent-roles.md) 约束执行：**先对齐字段，再动评分逻辑**。

输入契约（已可用）：
- `DocumentChunk` 侧：`chunk_id/doc_id/text/source/metadata`；其中 `metadata` 至少可见 `company/doc_type/date/source`。
- `POST /api/ingest`：支持可选 `metadata` 与单次 `chunk_size/chunk_overlap` 覆盖；非法参数返回 `code=invalid_chunk_params`。

输出契约（检索侧最小可解释字段）：
- `POST /api/rag` 的 `retrieval_hits` 每项至少包含：
  `chunk_id`, `score`, `source`, `metadata`, `text_preview`。
- Day 3 在不改接口字段名的前提下，可迭代检索/重排算法；若字段扩展需先更新 `coordination.md` §3。

---

*若当日日历并非「第 1 周第 2 天」，以负责人指定的周计划日为准，可将本文标题与「Day 2」字样替换为实际日序。*
