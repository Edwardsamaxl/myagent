# 2026-04-06 工作记录

## 一、已完成的修复

### 1. HuggingFaceReranker timeout 参数修复

**文件**: `src/agent/core/rerank.py`

**问题**: `CrossEncoder` 构造函数不接受 `timeout` 参数，导致模型加载时报错：
```
TypeError: XLMRobertaForSequenceClassification.__init__() got an unexpected keyword argument 'timeout'
```

**修复**:
- 移除构造函数中的 `model_kwargs={"timeout": 30}`
- 改在 `predict()` 调用时传入 `timeout=self.timeout`

```python
# 修复前（错误）
self._encoder = CrossEncoder(
    self.model_name,
    device=self.device,
    max_length=self.max_length,
    model_kwargs={"timeout": 30},  # ← 报错
)

# 修复后（正确）
scores = encoder.predict(pairs, show_progress_bar=False, timeout=self.timeout)
```

---

### 2. Ollama Embedding 超时问题

**现象**: Ollama embedding 请求 60 秒超时（`HTTPConnectionPool(host='127.0.0.1', port=11434): Read timed out`）

**处理方案**: 评估脚本已有健康检查和降级逻辑，embedding 不可用时自动降级为纯词频检索（lexical + tfidf），不影响整体流程。

---

### 3. 检索评估结果（维度一）

**配置**: `RETRIEVAL_TOP_K=6`，embedding 因 Ollama 超时降级为纯词频

| 指标 | 值 |
|------|-----|
| Recall@6 | 0.40 |
| HitRate@6 | 0.40 |
| MRR | 0.333 |

**分析**: 降级后纯词频检索对语义相关性问题（如"2024年营业利润同比增长率"）命中困难。有 15/25 条查询完全无命中，需要 embedding 语义检索才能改善。

---

## 二、待解决问题

### 1. Ollama 服务不稳定

`qwen3-embedding` 请求超时（60s），需确认：
- Ollama 服务是否正常运行：`ollama ps`
- 模型是否已加载：`ollama list`
- 网络连接是否正常

**临时方案**: 确保 Ollama 在评估前已启动并加载模型：
```bash
ollama serve
ollama run qwen3-embedding  # 预热模型
```

### 2. 评估未完成

原计划三个维度：
1. **RAG 检索有效性** - 已部分完成（Recall@6=0.40，embedding 降级）
2. **RAG 生成质量** - 脚本崩溃于 HuggingFaceReranker 加载（已修复 timeout，重跑即可）
3. **多 Agent 编排有效性** - 未运行

### 3. 团队状态异常

旧团队 `resume-verification` 的 TeamCreate 锁未清理，导致无法创建新团队。临时用单进程方式运行了评估。后续需要清理：
```bash
# 清理残留状态（如果 TeamCreate 再次失败）
rm -rf ~/.claude/teams/resume-verification
rm -rf ~/.claude/tasks/resume-verification
```

---

## 三、后续新进程继续工作清单

### 立即可做

- [ ] 重新运行完整评估：`python scripts/eval_full_pipeline.py`
- [ ] 确认 Ollama 正常运行后，验证 embedding 检索提升效果

### 需要修复的功能（来自 docs/unfinished-tasks.md）

#### P0 优先级
1. **RAG-Agent 路由边界** - 当前 `agent_service.py` 的 `chat()` 完全不调用 RAG
2. **Max Steps 限制** - 当前无执行步数上限控制
3. **Replan 机制** - 无重规划能力

#### P1 优先级
4. **意图分类** - 需要在路由前判断用户意图
5. **SimpleReranker → BGE Reranker** - 当前 reranker 是骨架实现
6. **Embedding 健康检查优化** - 超时时间可调

---

## 四、关键文件路径

| 文件 | 用途 |
|------|------|
| `scripts/eval_full_pipeline.py` | 三维度评估入口 |
| `src/agent/core/rerank.py` | 重排器（HuggingFaceReranker 修复在此） |
| `src/agent/application/rag_agent_service.py` | RAG 服务，含 embedding 健康检查 |
| `src/agent/config.py` | 配置，默认 embedding 模型 |
| `.env` | 运行时配置 |
| `data/eval/retrieval_test_set.json` | 检索评估集（25条） |
| `data/eval/week1_eval_set.jsonl` | 生成评估集（25条） |
| `runtime/eval_full/full_evaluation.json` | 评估结果输出 |

---

## 五、环境变量关键配置

```env
# Embedding 通道（已统一为 qwen3-embedding）
EMBEDDING_MODEL=qwen3-embedding
EMBEDDING_PROVIDER=ollama

# 检索权重
RETRIEVAL_LEXICAL_WEIGHT=0.35
RETRIEVAL_TFIDF_WEIGHT=0.25
RETRIEVAL_EMBEDDING_WEIGHT=0.40

# Rerank（评估时强制禁用 cascade 避免下载超时）
RERANK_CASCADE=false
```
