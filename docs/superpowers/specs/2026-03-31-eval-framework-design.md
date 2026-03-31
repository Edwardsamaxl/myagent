# P2: 评估指标体系设计

## 目标

建立完整的 **RAG 评估框架**，覆盖检索质量 + 生成质量，为 Ablation Study 提供数据支撑。

## 评估指标

### 检索指标

| 指标 | 定义 | 计算方式 |
|------|------|---------|
| **Recall@K** | 检索回来的文档中有多少包含正确答案 | `relevant_in_top_k / total_relevant` |
| **HitRate@K** | Top-K 中是否存在命中文档 | `1 if any(relevant_in_top_k) else 0` |
| **MRR** | 第一个命中的位置权重 | `1 / rank_of_first_hit` |

### 生成指标

| 指标 | 定义 | 计算方式 |
|------|------|---------|
| **Groundedness** | 生成回答有多少内容来自证据 | LLM 评估：`score_groundedness(answer, evidence)` |
| **Answer Relevance** | 回答是否切题 | LLM 评估：`score_relevance(answer, question)` |

### 评估数据结构

```python
@dataclass
class RetrievalEvalRecord:
    query: str
    expected_answer: str          # 人工标注的期望答案片段
    retrieval_hits: list[str]    # 检索返回的文档
    metrics: dict[str, float]     # Recall@K, HitRate@K, MRR

@dataclass
class GenerationEvalRecord:
    query: str
    evidence: list[str]           # 使用的证据
    generated_answer: str
    metrics: dict[str, float]     # Groundedness, Relevance
```

## 数据格式

评估数据保存在 `data/eval/` 目录：

```
data/eval/
├── retrieval_test_set.json   # 检索评估集（query + expected_answer）
├── generation_test_set.json   # 生成评估集（query + evidence + expected_answer）
└── eval_records/
    ├── retrieval_results.jsonl   # 检索评估结果
    └── generation_results.jsonl  # 生成评估结果
```

### 评估集格式

```json
// retrieval_test_set.json
[
  {
    "id": "q1",
    "query": "贵州茅台2024年营业收入是多少？",
    "expected_answers": ["1475亿元", "1475亿"]
  },
  ...
]
```

## 评估脚本

- `scripts/eval_retrieval.py` → 批量跑检索指标
- `scripts/eval_generation.py` → 批量跑生成指标
- `scripts/run_ablation.py` → 配置开关，A/B 对比实验

## Ablation 实验设计

```python
# run_ablation.py 配置
experiments = [
    {"name": "baseline", "EMBEDDING_ENABLED": True, "RERANK_ENABLED": True},
    {"name": "no-embedding", "EMBEDDING_ENABLED": False, "RERANK_ENABLED": True},
    {"name": "no-rerank", "EMBEDDING_ENABLED": True, "RERANK_ENABLED": False},
    {"name": "full-off", "EMBEDDING_ENABLED": False, "RERANK_ENABLED": False},
]
```

每个实验跑相同的 test set，输出对比表格。

## 指标聚合

`GET /api/metrics` 返回：

```json
{
  "retrieval": {
    "recall@3": 0.85,
    "hit_rate@3": 0.92,
    "mrr": 0.78
  },
  "generation": {
    "groundedness": 0.91,
    "relevance": 0.88
  }
}
```

## 实施步骤

1. 定义 `RetrievalEvalRecord` / `GenerationEvalRecord` 数据类
2. 实现 `recall_at_k`, `hit_rate_at_k`, `mrr` 计算函数
3. 实现 `GroundednessEvaluator` / `RelevanceEvaluator`（基于 LLM 判断）
4. 编写 `data/eval/retrieval_test_set.json`（至少 10 条）
5. 编写 `scripts/eval_retrieval.py` 批量脚本
6. 集成到 `core/evaluation.py` 的在线评估逻辑
