# Fin-agent 评估结果 — 2026-04-07

## RAG 模块评估结论

### 核心问题：简历数据无法证实

| 简历声称 | 实测结果 |
|----------|----------|
| Top-3 召回率 78% | **40%** |
| 降级后保持 62% | 更差（降级实际是唯一可用状态）|
| 两阶段重排后 Top-3 准确率 84% | 无法验证（BGE Reranker 未部署）|

### P0 根因

1. **Embedding 维度不匹配**
   - 索引中存储的是 768/3584 维向量
   - qwen3-embedding 实际输出 4096 维
   - 维度不匹配 → cosine similarity = 0 → 语义通道 100% 失效
   - 实测 retrieval 实际退化为纯词频（lexical + TF-IDF）

2. **BGE Reranker 未部署**
   - `ollama pull dengcao/bge-reranker-v2-m3` 从未执行
   - 两阶段重排效果无法验证

3. **评估指标格式不匹配**
   - chunk 中的数值带逗号（如 `1,234,567`）
   - 评估集期望纯数字格式
   - 导致数值匹配失败

4. **财务指标类查询 recall 仅 12%**
   - chunk 截断导致数字被切到不同 chunk
   - 15/25 财务类查询失败

### 检索实测数据（基于 25 条测试查询）

| 指标 | Top-3 | Top-6 |
|------|-------|-------|
| Recall@K | 0.40 | 0.40 |
| HitRate@K | 0.40 | 0.40 |
| MRR | 0.33 | 0.32 |

按查询类型：
- 公司基本信息（股票代码、董秘等）: **86%** recall
- 财务指标（营业收入、净利润等）: **12%** recall ← 主要失败来源

### 修复优先级

1. **重建 retrieval_index** — 删除 `data/retrieval_index/`，用当前 qwen3-embedding 重新计算 4096 维向量
2. **部署 BGE Reranker** — `ollama pull dengcao/bge-reranker-v2-m3`
3. **修复评估指标格式** — 数值归一化（去掉逗号）

---

## Agent 模块评估结论

### 整体状态：基本可用，但有 bug 已修复

| 组件 | 状态 |
|------|------|
| LangGraph ReAct Agent 构建 | PASS |
| Tool-calling 识别和调用 | PASS |
| Planner 任务分解 | PASS（已修复 2 个 bug）|
| WorkerExecutor 工具调用 | PASS（已修复 1 个 bug）|
| Coordinator 多步执行 | PASS |
| Agent 循环 | PASS |

### 已修复的 bug（`src/agent/core/planning/coordinator.py`）

1. **JSON 解析失败**
   - 模型响应包含 ```json ... ``` markdown fence
   - `json.loads()` 直接调用失败
   - 修复：`create_plan()` 在解析前 strip markdown fence

2. **Action 别名未识别**
   - 模型返回 `retrieve_financial_data`/`query_data` 等别名
   - 代码只识别 `rag`，导致步骤被跳过
   - 修复：添加 action 别名映射表

3. **工具输入前缀问题**
   - `detail="计算 99+1"` 直接传给 `calculate`
   - 导致 `eval("计算 99+1")` 失败
   - 修复：添加中英文前缀清理

### 注意事项

- 简历中"多 Worker 并行执行"实际是 ThreadPoolExecutor 调度，不是真正的 CPU 并行
- 如果被问到"CPU 层面怎么实现并行"，需要坦诚说明是协作式串行执行

---

## 后续行动建议

### 方案 A：修复后重新测试（推荐）

1. `ollama pull qwen3-embedding`（确认 4096 维）
2. 删除 `data/retrieval_index/` 目录
3. 重新摄入文档（触发 4096 维向量计算）
4. `ollama pull dengcao/bge-reranker-v2-m3`
5. 运行 `python scripts/eval_full_pipeline.py`
6. 用实测数据更新简历

### 方案 B：保守修改简历

如果不想等修复，先把简历改为：
- "Embedding 可用时 Top-3 召回率 > 40%，降级后保持词频检索能力"
- "两阶段重排（规则初筛 + BGE 精排，模型需额外部署）"
- 删除具体的 78%/84% 数字

### 评估文件位置

- 完整 RAG 评估报告：`runtime/rag_retrieval_eval_report.md`
- 评估脚本：`scripts/eval_full_pipeline.py`
- 检索评估集：`data/eval/retrieval_test_set.json`
- 生成评估集：`data/eval/week1_eval_set.jsonl`
