# 简历问答专用（可直接复用）

说明：下面每一题都按“面试可口述”的粒度写，尽量做到你只要照着说就能解释清楚。你后续问的新问题，我会继续往这里追加。

---

## 目标检测：YOLO 是什么 / YOLOv8 有什么特点？

### Q：端到端的检测流程是 YOLO 的特点吗？能简要解释 YOLO 是什么吗？

**A（简历口径，建议 20-40 秒）：**
端到端（end-to-end）的核心是：**一次前向推理直接输出检测结果**，不需要像两阶段检测那样先生成候选框（proposal）再逐个分类。  
YOLO（You Only Look Once）是典型的**单阶段目标检测器**：把图像划分为网格/特征位置，在同一次网络输出里同时预测每个位置的 **边界框（bbox）+类别概率（cls）+置信度**。因此它通常具备**速度快、实时性强**的优势，适合工业检测场景。

在你的缺陷检测系统里，这种“端到端直接输出框与类别”的结构意味着：输入图像 -> 网络一次推理 -> 输出缺陷位置与类别，整体延迟更低、更工程友好。

---

### Q：YOLOv8 的特点又是什么？

**A（简历口径，建议 20-40 秒）：**
YOLOv8 是 Ultralytics 团队在 YOLO 系列上的改进版，主要特点可以概括为三点：  
1. **更好的精度-速度平衡**：采用更高效的特征提取模块（如 C2f 结构）与改进的检测头设计，在保持实时性的同时提升检测质量。  
2. **更稳的定位回归**：边界框回归引入分布式/离散化思路（如 DFL），让框的位置预测更细致、训练更稳定。  
3. **多任务能力与工程化完善**：YOLOv8 在同一框架下更易扩展到检测以外的任务（如分割/姿态），同时内置了工业常用的数据增强与训练管线，便于在少样本缺陷数据上做自训练或迁移学习。

如果面试追问，我可以结合我项目回答：我在 YOLOv8 的基础上完成了数据清洗、标准监督训练与半监督自训练（Self-training / 伪标签），并用 mAP/AUC 等指标做对比验证。

---

## 目标检测/机器学习：常用有监督与弱监督策略

### Q：常用的有监督和弱监督策略有哪些？

**A（简历口径，建议 30-60 秒）：**
有监督学习（Supervised）是用完整标注数据直接训练模型，常见做法包括：标准监督训练、数据增强、迁移学习/微调、类别不平衡处理（如重加权或更合适的损失）、以及正则化/学习率调度来提升泛化能力。

弱监督（Weakly Supervised / 弱标签）指训练信号不够精确或不完整：常见路线包括“粗粒度标注 + 多实例学习（MIL）”、伪标签/自训练（Self-training / Pseudo-labeling：用当前模型对无标注数据预测高置信伪标签再迭代训练）、一致性正则化（对同一样本不同增强要求输出一致）、以及教师-学生框架（Teacher-Student / EMA Teacher）用更稳定的教师产生软标签。由于弱标签可能噪声较大，通常会用置信度阈值或 Top-K 筛选来降低噪声影响。

在我的项目里，缺陷检测少标注方案本质属于半监督/弱监督中的伪标签自训练：先用 seed 监督训练 warm-up，再在未标注数据上生成伪标签并进行两阶段训练，同时用独立验证集评估，验证小样本场景下的性能提升。

---

## RAG 检索：混合检索实现（lexical + TF-IDF/BM25 + embedding）与融合（weighted_sum/RRF）

### Q：构建混合检索通道（lexical + TF-IDF/BM25 + embedding）与融合策略（weighted_sum/RRF），并通过中文 tokenization 改造显著改善自然问句召回结果。这里的每一步你分别怎么实现的？

**A（简历口径，建议 1-2 分钟）：**

我在 `src/agent/core/retrieval.py` 实现了“三路召回打分 + 融合排序”的混合检索：
**chunk 入库建索引（token/稀疏统计/embedding）→ query 侧做 token/TF 与 embedding → 每个 chunk 计算三路分数 → weighted_sum 或 RRF 融合 → 输出 top-k 命中。**

1) **tokenization（中文 2/3-gram + 数字锚点）**
- 英文/数字：保留连续 token
- 中文连续串：生成 bi-gram / tri-gram，避免之前中文整段 token 太粗导致 lexical/tfidf 命中为 0

```src/agent/core/retrieval.py
def _tokenize(text: str) -> list[str]:
    raw_tokens = [t for t in re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", text.lower()) if t]
    ...
    # 中文仅加入 bi-gram / tri-gram，不保留原始长串 token
    for n in (2, 3):
        for i in range(0, len(zh) - n + 1):
            gram = zh[i : i + n]
            out.append(gram)
```

2) **chunk 入库（`upsert_chunks`：建 TF 表 + 向量缓存）**
- 对每个 chunk：`terms = _tokenize(chunk.text)`，统计 `tf(term)`，保存到 `_chunk_terms`
- 若启用 embedding：`embedding_provider.embed_texts(embedding_inputs)`，缓存到 `_chunk_embeddings`

```src/agent/core/retrieval.py
def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
    for chunk in chunks:
        terms = _tokenize(chunk.text)
        tf = defaultdict(int)
        for term in terms:
            tf[term] += 1
        self._chunk_terms[chunk.chunk_id] = dict(tf)
        ...
        if self._embedding_provider:
            vectors = self._embedding_provider.embed_texts(embedding_inputs)
            self._chunk_embeddings[chunk_id] = vectors[idx]
    self._rebuild_df()
```

3) **query 侧（`search_with_debug`：q_terms/q_tf + q_embedding）**

```src/agent/core/retrieval.py
q_terms = _tokenize(query)
q_tf = defaultdict(int)
for t in q_terms:
    q_tf[t] += 1
q_embedding = self._embed_query(query)
```

4) **三路相关度计算（每个 chunk 算 lexical / sparse / embedding）**
- `lexical`：query token 与 chunk token 的集合重合比例（顺序无关）
- `sparse`：TF-IDF 或 BM25 或 TF-IDF-BM25（由 `SPARSE_MODE` 控制）
- `embedding`：dense cosine（embedding 向量余弦相似度）

```src/agent/core/retrieval.py
lexical = self._lexical_score(q_terms, chunk.text)
tfidf = self._cosine_tfidf(q_tf, self._chunk_terms.get(chunk_id, {}))
bm25 = self._bm25_score(q_tf, self._chunk_terms.get(chunk_id, {}), chunk_id)
sparse = self._sparse_score(tfidf, bm25)
embedding = self._embedding_score(q_embedding, self._chunk_embeddings.get(chunk_id))
score = (self._lexical_weight * lexical
         + self._tfidf_weight * sparse
         + self._embedding_weight * embedding)
```

5) **融合策略（`weighted_sum` 或 `RRF`）**
- `weighted_sum`：直接用上面加权 `score` 排序取 top-k
- `RRF`：分别按三路 rank 做 Reciprocal Rank Fusion 再汇总

```src/agent/core/retrieval.py
if self._fusion_mode == "rrf":
    self._apply_rrf(rrf_scores, lexical_rank, self._lexical_weight)
    self._apply_rrf(rrf_scores, tfidf_rank, self._tfidf_weight)
    self._apply_rrf(rrf_scores, embedding_rank[: self._embedding_top_k], self._embedding_weight)
    final_scores = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

def _apply_rrf(..., k: int = 60):
    out_scores[chunk_id] += weight * (1.0 / (k + rank))
```

简而言之：混合检索把“精确词项匹配（lexical）+ 稀疏词项权重（TF-IDF/BM25）+ 语义向量相似（embedding）”三种信号合在一起，中文 tokenization 用 2/3-gram 提升了 token-level 重合概率，从而显著提升自然问句召回命中。

---

## 补充：RRF / TF-IDF vs BM25 / 中文 2/3-gram 是否需要写进简历

### Q：RRF 是什么？

**A（简历口径，建议 20-40 秒）：**
RRF（Reciprocal Rank Fusion，倒数名次融合）是一种把多路检索结果融合的排序方法。它不直接比较分数尺度，而是把每一路的命中按“名次 rank”来打分：某个命中在第 `rank` 名出现时贡献大约是 `1/(k+rank)`（k 为常数，减小长尾影响）。最后把不同通道的贡献相加得到融合分，从而得到更稳的最终排序。

在我的系统里，RRF 用于把 lexical / sparse（TF-IDF 或 BM25）/ embedding 三路的候选融合成最终 top-k。

### Q：TF-IDF 和 BM25 的区别是什么？

**A（简历口径，建议 30-60 秒）：**
两者都是“稀疏词项”相关性模型，但核心差异在于：
1) **权重建模不同**：  
- TF-IDF 更像是 `tf * idf` 的向量余弦相似度；  
- BM25 在 TF 上做了“饱和”（term 频次不会无限线性放大），并加入文档长度归一化，更符合检索排序的经验。
2) **归一化与 TF 处理不同**：  
BM25 对长文/重复词的惩罚与提升方式更直接，通常在信息检索场景表现更稳，但需要合理的参数（如 k1、b）。

### Q：为什么在本系统中 TF-IDF 的表现更好？

**A（简历口径，建议 30-60 秒）：**
我当前系统默认的稀疏通道是 TF-IDF（`SPARSE_MODE` 默认即为 `tfidf`），并且使用了中文 n-gram tokenization（2/3-gram）把中文拆成更稳定的“词项”。在这种“短语级片段更容易共享 token”的设置下，TF-IDF 的词项权重与余弦相似度更容易稳定地拉开相关候选的差距；BM25 虽然也在代码中提供（可切换 `SPARSE_MODE=bm25`），但它更依赖 TF 的饱和与长度归一化参数（k1/b），在当前默认参数下未必比 TF-IDF 更合适。  
因此我选择 TF-IDF 作为当前默认的稀疏基线，并在需要时通过 A/B 继续验证 BM25 的收益。

### Q：中文 tokenization 就是简单划分成 2/3-gram？那我感觉没有写进简历的必要啊

**A（简历口径，建议 10-20 秒）：**
不需要写到实现细节（比如滑窗循环），但建议用一句话提到“中文 n-gram tokenization 以提升稀疏检索的召回稳定性”。简历篇幅有限时，你可以只写“通过中文 n-gram tokenization 提升自然问句召回率”，面试追问我再展开实现逻辑与实验对比。

