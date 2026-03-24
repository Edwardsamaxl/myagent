# Agent 操作手册（多 Agent 协作执行版）

本文给出“今天该怎么做、按什么顺序做、怎么验收”的可执行流程。  
默认适用于当前项目角色分工：PM / A / B / C / D / E / F（见 `docs/agent-roles.md`）。

---

## 1) 先读什么（开工前 10 分钟）

每个 Agent 开工前按顺序阅读：
1. `docs/coordination.md`（接口契约与边界，必须先对齐）
2. 当日计划文档（如 `docs/day3-daily-plan.md`）
3. 与自己职责相关的核心代码目录
4. 最新评估结果（`runtime/day3/*.json`）

禁止跳过契约文档直接改代码。

---

## 2) 角色执行清单（谁做什么）

### PM（协调）
- 维护 `docs/coordination.md` 契约与放行规则。
- 拆分任务，标注依赖与冲突文件。
- 合并前检查“先文档后实现”是否满足。

### A（数据 / Ingestion）
- 负责 `ingestion.py`、`scripts/run_ingest.py`、`scripts/dump_chunks.py`。
- 保证 metadata、去重、参数覆盖、批量入库统计口径一致。
- 交付可复现命令和样本输出。

### B（检索 / 重排）
- 负责 `retrieval.py`、`rerank.py` 与检索相关配置。
- 先稳定字段契约，再迭代算法（切词、BM25、融合、rerank）。
- 所有策略调整必须配 A/B 报告。

### C（生成 / Agent 循环）
- 负责 `generation.py`、`evidence_format.py`、`agent_loop.py`。
- 保证“有证据可回答、无证据要拒答、引用可核对”。
- 优先优化拒答策略与结构化回答质量。

### D（接口 / Web）
- 负责 `web_app.py`、`main.py` 与 API 错误语义。
- 不在路由层写检索算法。
- 契约字段变化需先在 `coordination.md` 批准。

### E（评估 / 可观测）
- 负责 `evaluation.py`、`observability.py` 与 `runtime/*` 报告。
- 固定评估口径，输出对比结论和失败样例。
- 确保每次优化可复现、可回归。

### F（解释 / 学习）
- 面向用户解释：结论 -> 原理 -> 取舍 -> 验证 -> 下一步。
- 仅解释与建议，非用户明确要求时不改代码。

---

## 3) 标准工作流（每个任务都按这个走）

1. **定义目标**：一句话写清“改什么、为什么”。
2. **锁定范围**：明确可改文件和禁止改动文件。
3. **实现最小改动**：先保证契约兼容，再提升效果。
4. **本地验证**：跑最小可复现命令（必须可复制）。
5. **输出证据**：给出指标对比、典型样例、风险。
6. **更新文档**：若改了契约，先改 `coordination.md` 再合代码。

---

## 4) 常用命令手册（本项目）

### 4.1 启动服务
```powershell
python main.py
```

### 4.2 批量入库
```powershell
python scripts/run_ingest.py
```

### 4.3 本地切块检查
```powershell
python scripts/dump_chunks.py "data/raw/finance/贵州茅台/年报_2024.md"
```

### 4.4 检索问答测试
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:7860/api/rag" -Method POST -ContentType "application/json" -Body '{"question":"贵州茅台2024年营业收入是多少？","top_k":6}'
```

### 4.5 检索 A/B 评估
```powershell
python scripts/eval_embedding_ab.py
```

### 4.6 生成策略 A/B 评估
```powershell
python scripts/eval_generation_policy_ab.py
```

---

## 5) 验收模板（提交结果时必须包含）

每个 Agent 回传结果必须包含以下 5 项：

1. **结论**：一句话说明是否达成目标。
2. **改动清单**：列出文件与改动点。
3. **验证命令**：可直接复制执行。
4. **指标对比**：至少一组 before/after 数值。
5. **风险与下一步**：最多 3 条。

---

## 6) 当前阶段推荐执行顺序

基于现状（Day3+）建议按顺序推进：
1. B：检索/重排小步调优并稳定 top1 命中。
2. C：生成策略从“过度拒答”调整为“可解释可回答”。
3. E：固定评估口径，输出统一日报。
4. D：回归接口稳定性与错误码语义。
5. PM：合并文档与阶段总结，更新周计划状态。

---

## 7) 常见故障排查

### 7.1 命中为 0
- 先确认是否重新入库（重启服务后内存索引会清空）。
- 再看 `query_tokens` 与 `score_breakdown` 是否全 0。

### 7.2 全部返回拒答
- 检查是否使用了 mock 模型输出导致结构不稳定。
- 检查生成后处理阈值是否过严（`insufficient_evidence` 触发过多）。

### 7.3 评估结果波动大
- 检查 `.env`、模型端口、`EVAL_USE_MOCK` 是否一致。
- 固定同一数据集和同一配置再复跑。

---

## 8) 操作纪律（必须遵守）

- 不改契约就不要改字段名。
- 不做与任务无关重构。
- 不在未记录指标情况下宣称“优化成功”。
- 不提交敏感文件（如 `.env`、大体积 debug 输出）。

---

## 9) 交付物建议

每个阶段至少保留：
- 1 份当日总结文档（`docs/day*-*.md`）
- 1 份评估摘要（`runtime/day*/summary*.json`）
- 1 份可复现命令列表（写在总结文档中）

这样周末复盘时可直接串成“改动 -> 指标 -> 结论”的完整证据链。

