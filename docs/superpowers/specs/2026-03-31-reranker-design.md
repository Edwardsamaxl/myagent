# P1: BGE Reranker 实现设计

## 目标

将 `SimpleReranker`（骨架实现）替换为真实的 **BGE Reranker v2-m3**，实现可用的文档重排序。

## 方案选择

**BGE Reranker v2-m3** (`BAAI/bge-reranker-v2-m3`)：
- 智源开源，中文效果好
- 支持 Ollama 本地部署（`ollama pull BAAI/bge-reranker-v2-m3`）
- HuggingFace 格式也可用
- 直接输入 query + passage，输出 0~1 相关性分数

## 核心实现

### Ollama 部署

```bash
ollama pull BAAI/bge-reranker-v2-m3
```

API 调用（Ollama）：

```
POST /api/rerank
{
  "model": "BAAI/bge-reranker-v2-m3",
  "query": "...",
  "documents": ["...", "..."]
}
```

返回每个 doc 的相关度分数列表。

### 代码实现

```python
# src/agent/core/rerank.py

class BGEReranker:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    def rerank(self, query: str, documents: list[str], top_k: int = 3) -> list[RerankResult]:
        url = f"{self.base_url}/api/rerank"
        payload = {
            "model": "BAAI/bge-reranker-v2-m3",
            "query": query,
            "documents": documents
        }
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        scores = resp.json().get("results", [])  # [{"index": 0, "relevance_score": 0.95}, ...]
        sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i]["relevance_score"], reverse=True)
        return [RerankResult(documents[sorted_idx[i]], scores[sorted_idx[i]]["relevance_score"])
                for i in range(min(top_k, len(sorted_idx)))]
```

### 配置

- 新增 `RERANKER_PROVIDER=ollama|openai_compatible|huggingface`
- 新增 `RERANKER_MODEL=BAAI/bge-reranker-v2-m3`
- 新增 `RERANK_ENABLED=true|false`（可 bypass）

### 集成到 RAG Pipeline

```
retrieval (top-6) → BGEReranker → rerank (top-3) → generation
```

- `RagAgentService` 在 retrieval 后、generation 前调用 rerank
- rerank 结果覆盖原有 retrieval 顺序

## 兼容性

- 保留 `SimpleReranker` 作为 fallback（当 Ollama 无 BGE 时）
- 配置 `RERANK_ENABLED=false` 可完全跳过 rerank 步骤

## 测试验证

- 相同 query，对比 rerank 前 top-6 和 rerank 后 top-3 的顺序变化
- 主观验证：重排后的顺序是否更相关
