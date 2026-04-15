from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

from .schemas import DocumentChunk, RetrievalHit


def _tokenize(text: str) -> list[str]:
    raw_tokens = [t for t in re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", text.lower()) if t]
    out: list[str] = []
    seen: set[str] = set()
    for token in raw_tokens:
        # 仅保留英文/数字原 token；中文连续串改用 n-gram，避免超长中文 token 带来噪声
        if re.fullmatch(r"[a-z0-9]+", token) and token not in seen:
            out.append(token)
            seen.add(token)
        # 混合串中补数字子 token（如 2024、600519），增强时间/代码约束
        for num in re.findall(r"\d+", token):
            if num and num not in seen:
                out.append(num)
                seen.add(num)
        zh_runs = re.findall(r"[\u4e00-\u9fff]+", token)
        for zh in zh_runs:
            # 中文仅加入 bi-gram / tri-gram，但长词（>=4）保留完整形式以保留完整语义
            if len(zh) >= 4 and zh not in seen:
                out.append(zh)
                seen.add(zh)
            for n in (2, 3):
                for i in range(0, len(zh) - n + 1):
                    gram = zh[i : i + n]
                    if gram and gram not in seen:
                        out.append(gram)
                        seen.add(gram)
    return out


class InMemoryHybridRetriever:
    """
    MVP hybrid retriever:
    - lexical score: token overlap
    - semantic score: simplified tf-idf cosine
    """

    def __init__(
        self,
        *,
        embedding_provider: object | None = None,
        fusion_mode: str = "weighted_sum",
        lexical_weight: float = 0.35,
        tfidf_weight: float = 0.25,
        embedding_weight: float = 0.40,
        embedding_top_k: int = 12,
        numeric_weight: float = 0.0,
        index_dir: Path | None = None,
    ) -> None:
        sparse_mode = os.getenv("SPARSE_MODE", "tfidf").strip().lower()
        if sparse_mode not in {"tfidf", "bm25", "tfidf_bm25"}:
            sparse_mode = "tfidf"
        self._chunks: dict[str, DocumentChunk] = {}
        self._doc_freq: defaultdict[str, int] = defaultdict(int)
        self._chunk_terms: dict[str, dict[str, int]] = {}
        self._chunk_lengths: dict[str, int] = {}
        self._chunk_embeddings: dict[str, list[float]] = {}
        self._chunk_numbers: dict[str, set[str]] = {}  # chunk_id → 数字字符串集合
        self._embedding_provider = embedding_provider
        self._fusion_mode = fusion_mode if fusion_mode in {"weighted_sum", "rrf"} else "weighted_sum"
        self._lexical_weight = lexical_weight
        self._tfidf_weight = tfidf_weight
        self._embedding_weight = embedding_weight
        self._embedding_top_k = max(1, embedding_top_k)
        self._numeric_weight = max(0.0, min(1.0, numeric_weight))
        self._sparse_mode = sparse_mode
        self._bm25_k1 = float(os.getenv("BM25_K1", "1.5"))
        self._bm25_b = float(os.getenv("BM25_B", "0.75"))
        self._tfidf_bm25_alpha = float(os.getenv("TFIDF_BM25_ALPHA", "0.5"))
        self._index_dir: Path | None = index_dir

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        embedding_inputs: list[str] = []
        embedding_ids: list[str] = []
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
            terms = _tokenize(chunk.text)
            tf: dict[str, int] = defaultdict(int)
            for term in terms:
                tf[term] += 1
            self._chunk_terms[chunk.chunk_id] = dict(tf)
            self._chunk_lengths[chunk.chunk_id] = len(terms)
            # 数值索引：支持数值类查询的正则匹配
            self._chunk_numbers[chunk.chunk_id] = self._extract_numbers(chunk.text)
            if self._embedding_provider:
                embedding_ids.append(chunk.chunk_id)
                embedding_inputs.append(chunk.text)
        if self._embedding_provider and embedding_inputs:
            try:
                vectors = self._embedding_provider.embed_texts(embedding_inputs)
                for idx, chunk_id in enumerate(embedding_ids):
                    if idx < len(vectors):
                        self._chunk_embeddings[chunk_id] = vectors[idx]
            except Exception:
                # embedding 通道异常时降级到 lexical/tfidf，避免阻塞主链路
                self._chunk_embeddings = {}
        self._rebuild_df()

    def set_weights(
        self,
        *,
        lexical: float | None = None,
        tfidf: float | None = None,
        embedding: float | None = None,
        numeric: float | None = None,
    ) -> None:
        """动态更新各通道权重（支持按查询类型动态调权）。"""
        if lexical is not None:
            self._lexical_weight = max(0.0, min(1.0, lexical))
        if tfidf is not None:
            self._tfidf_weight = max(0.0, min(1.0, tfidf))
        if embedding is not None:
            self._embedding_weight = max(0.0, min(1.0, embedding))
        if numeric is not None:
            self._numeric_weight = max(0.0, min(1.0, numeric))

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        hits, _ = self.search_with_debug(query=query, top_k=top_k)
        return hits

    def search_with_debug(
        self,
        query: str,
        top_k: int = 5,
        *,
        numeric_indicators: set[str] | None = None,
    ) -> tuple[list[RetrievalHit], dict[str, object]]:
        # 惰性加载 embeddings（首次搜索时触发）
        self.ensure_embeddings_loaded()
        if not query.strip() or not self._chunks:
            return [], {
                "query_tokens": [],
                "total_chunks": len(self._chunks),
                "positive_score_count": 0,
                "top_scores": [],
                "zero_score_reasons": {
                    "both_zero": 0,
                    "lexical_zero_only": 0,
                    "semantic_zero_only": 0,
                },
            }
        top_k = max(1, top_k)
        q_terms = _tokenize(query)
        q_tf: dict[str, int] = defaultdict(int)
        for t in q_terms:
            q_tf[t] += 1
        q_embedding = self._embed_query(query)
        # 数值索引提取（支持数值类查询）
        q_numbers = self._extract_numbers(query)

        weighted_scores: list[tuple[str, float]] = []
        rrf_scores: defaultdict[str, float] = defaultdict(float)
        per_chunk_scores: dict[str, dict[str, float]] = {}
        zero_score_reasons = {
            "both_zero": 0,
            "lexical_zero_only": 0,
            "semantic_zero_only": 0,
            "embedding_zero_only": 0,
        }
        lexical_rank: list[tuple[str, float]] = []
        tfidf_rank: list[tuple[str, float]] = []
        embedding_rank: list[tuple[str, float]] = []
        numeric_rank: list[tuple[str, float]] = []
        for chunk_id, chunk in self._chunks.items():
            lexical = self._lexical_score(q_terms, chunk.text)
            tfidf = self._cosine_tfidf(q_tf, self._chunk_terms.get(chunk_id, {}))
            bm25 = self._bm25_score(q_tf, self._chunk_terms.get(chunk_id, {}), chunk_id)
            sparse = self._sparse_score(tfidf, bm25)
            embedding = self._embedding_score(q_embedding, self._chunk_embeddings.get(chunk_id))
            per_chunk_scores[chunk_id] = {
                "lexical": lexical,
                "tfidf": tfidf,
                "bm25": bm25,
                "sparse": sparse,
                "embedding": embedding,
            }
            if lexical > 0:
                lexical_rank.append((chunk_id, lexical))
            if sparse > 0:
                tfidf_rank.append((chunk_id, sparse))
            if embedding > 0:
                embedding_rank.append((chunk_id, embedding))
            # 数值通道：数字序列 Jaccard 匹配
            numeric = self._numeric_score(q_numbers, chunk_id) if q_numbers else 0.0
            if numeric > 0:
                numeric_rank.append((chunk_id, numeric))
            per_chunk_scores[chunk_id]["numeric"] = numeric

            score = (
                self._lexical_weight * lexical
                + self._tfidf_weight * sparse
                + self._embedding_weight * embedding
                + self._numeric_weight * numeric
            )
            # 数值类查询关键词 Boost：如果提供了指标关键词且 chunk 中包含
            if numeric_indicators:
                kw_boost = sum(1 for kw in numeric_indicators if kw in chunk.text)
                if kw_boost > 0:
                    # additive boost: 直接加分而非乘法，确保金融表格类低词频chunk能追上
                    boost = 0.20 * min(kw_boost, 3) / 3.0
                    score = score + boost
            if embedding == 0.0 and self._embedding_weight > 0:
                # embedding 不可用时，重新归一化词频权重
                total_w = self._lexical_weight + self._tfidf_weight
                if total_w > 0:
                    score = (self._lexical_weight / total_w) * lexical + (self._tfidf_weight / total_w) * sparse
            if score > 0:
                weighted_scores.append((chunk_id, score))
            if lexical == 0 and tfidf == 0 and embedding == 0:
                zero_score_reasons["both_zero"] += 1
            elif lexical == 0:
                if tfidf > 0 and embedding > 0:
                    zero_score_reasons["lexical_zero_only"] += 1
                elif tfidf == 0 and embedding > 0:
                    zero_score_reasons["semantic_zero_only"] += 1
            elif tfidf == 0 and embedding == 0:
                zero_score_reasons["semantic_zero_only"] += 1
            elif embedding == 0 and (lexical > 0 or tfidf > 0):
                zero_score_reasons["embedding_zero_only"] += 1

        lexical_rank.sort(key=lambda item: item[1], reverse=True)
        tfidf_rank.sort(key=lambda item: item[1], reverse=True)
        embedding_rank.sort(key=lambda item: item[1], reverse=True)
        numeric_rank.sort(key=lambda item: item[1], reverse=True)
        weighted_scores.sort(key=lambda item: item[1], reverse=True)

        if self._fusion_mode == "rrf":
            self._apply_rrf(rrf_scores, lexical_rank, self._lexical_weight)
            self._apply_rrf(rrf_scores, tfidf_rank, self._tfidf_weight)
            self._apply_rrf(
                rrf_scores,
                embedding_rank[: self._embedding_top_k],
                self._embedding_weight,
            )
            if self._numeric_weight > 0 and numeric_rank:
                self._apply_rrf(rrf_scores, numeric_rank, self._numeric_weight)
            final_scores = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        else:
            final_scores = weighted_scores

        hits: list[RetrievalHit] = []
        score_breakdown: list[dict[str, object]] = []
        for chunk_id, score in final_scores[:top_k]:
            chunk = self._chunks[chunk_id]
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    score=round(score, 6),
                    text=chunk.text,
                    source=chunk.source,
                    metadata=chunk.metadata,
                )
            )
            channels = per_chunk_scores.get(chunk_id, {})
            sources = [
                key
                for key, val in channels.items()
                if isinstance(val, float) and val > 0
            ]
            score_breakdown.append(
                {
                    "chunk_id": chunk_id,
                    "fused": round(score, 6),
                    "lexical": round(channels.get("lexical", 0.0), 6),
                    "tfidf": round(channels.get("tfidf", 0.0), 6),
                    "bm25": round(channels.get("bm25", 0.0), 6),
                    "sparse": round(channels.get("sparse", 0.0), 6),
                    "embedding": round(channels.get("embedding", 0.0), 6),
                    "numeric": round(channels.get("numeric", 0.0), 6),
                    "sources": sources,
                }
            )
        debug: dict[str, object] = {
            "query_tokens": q_terms,
            "total_chunks": len(self._chunks),
            "fusion_mode": self._fusion_mode,
            "sparse_mode": self._sparse_mode,
            "positive_score_count": len(final_scores),
            "top_scores": [round(score, 6) for _, score in final_scores[:top_k]],
            "zero_score_reasons": zero_score_reasons,
            "score_breakdown": score_breakdown,
        }
        return hits, debug

    def _sparse_score(self, tfidf: float, bm25: float) -> float:
        if self._sparse_mode == "bm25":
            return bm25
        if self._sparse_mode == "tfidf_bm25":
            a = min(1.0, max(0.0, self._tfidf_bm25_alpha))
            return a * tfidf + (1.0 - a) * bm25
        return tfidf

    def _bm25_score(self, q_tf: dict[str, int], d_tf: dict[str, int], chunk_id: str) -> float:
        if not q_tf or not d_tf:
            return 0.0
        n_docs = max(1, len(self._chunk_terms))
        dl = max(1, self._chunk_lengths.get(chunk_id, sum(d_tf.values())))
        avgdl = self._avg_doc_len()
        k1 = max(0.01, self._bm25_k1)
        b = min(1.0, max(0.0, self._bm25_b))
        score = 0.0
        for term in q_tf.keys():
            tf = d_tf.get(term, 0)
            if tf <= 0:
                continue
            df = self._doc_freq.get(term, 0)
            idf = math.log(max(1e-9, (n_docs - df + 0.5) / (df + 0.5)))
            denom = tf + k1 * (1.0 - b + b * (dl / max(1e-9, avgdl)))
            score += idf * ((tf * (k1 + 1.0)) / max(1e-9, denom))
        return score

    def _avg_doc_len(self) -> float:
        if not self._chunk_lengths:
            return 1.0
        return sum(self._chunk_lengths.values()) / max(1, len(self._chunk_lengths))

    @staticmethod
    def _extract_numbers(text: str) -> set[str]:
        """从文本中提取所有数字字符串（支持整数、小数、会计格式）。"""
        raw = re.findall(r"[\d]+(?:[.,]\d+)*(?:[eE][+-]?\d+)?", text)
        out: set[str] = set()
        for token in raw:
            cleaned = token.replace(",", "")
            if cleaned:
                out.add(cleaned)
        return out

    def _numeric_score(self, query_numbers: set[str], chunk_id: str) -> float:
        """
        数值匹配分数：query 中数字在 chunk 中出现的 Jaccard 相似度。
        """
        chunk_numbers = self._chunk_numbers.get(chunk_id, set())
        if not query_numbers or not chunk_numbers:
            return 0.0
        intersection = query_numbers & chunk_numbers
        if not intersection:
            return 0.0
        union = query_numbers | chunk_numbers
        return len(intersection) / len(union)

    def _embed_query(self, query: str) -> list[float] | None:
        if not self._embedding_provider:
            return None
        try:
            vectors = self._embedding_provider.embed_texts([query])
        except Exception:
            return None
        if not vectors:
            return None
        return vectors[0]

    def _embedding_score(self, q_vec: list[float] | None, d_vec: list[float] | None) -> float:
        if not q_vec or not d_vec:
            return 0.0
        return max(0.0, self._cosine_dense(q_vec, d_vec))

    @staticmethod
    def _cosine_dense(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        numerator = sum(x * y for x, y in zip(a, b))
        a_norm = math.sqrt(sum(x * x for x in a))
        b_norm = math.sqrt(sum(y * y for y in b))
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return numerator / (a_norm * b_norm)

    @staticmethod
    def _apply_rrf(
        out_scores: defaultdict[str, float],
        rank_list: list[tuple[str, float]],
        weight: float,
        *,
        k: int = 60,
    ) -> None:
        if weight <= 0:
            return
        for rank, (chunk_id, _) in enumerate(rank_list, start=1):
            out_scores[chunk_id] += weight * (1.0 / (k + rank))

    def _rebuild_df(self) -> None:
        self._doc_freq.clear()
        for term_tf in self._chunk_terms.values():
            for term in term_tf.keys():
                self._doc_freq[term] += 1

    def _lexical_score(self, query_terms: list[str], chunk_text: str) -> float:
        if not query_terms:
            return 0.0
        chunk_terms = set(_tokenize(chunk_text))
        overlap = sum(1 for t in query_terms if t in chunk_terms)
        return overlap / max(1, len(set(query_terms)))

    def _cosine_tfidf(self, q_tf: dict[str, int], d_tf: dict[str, int]) -> float:
        if not q_tf or not d_tf:
            return 0.0
        n_docs = max(1, len(self._chunk_terms))

        # Precompute global IDF shared by query and document
        all_terms = set(q_tf.keys()) | set(d_tf.keys())
        global_idf = {}
        for term in all_terms:
            df = self._doc_freq.get(term, 0)
            global_idf[term] = math.log((n_docs + 1) / (1 + df)) + 1.0

        q_vec: dict[str, float] = {}
        d_vec: dict[str, float] = {}

        for term, freq in q_tf.items():
            q_vec[term] = freq * global_idf.get(term, 1.0)
        for term, freq in d_tf.items():
            d_vec[term] = freq * global_idf.get(term, 1.0)

        numerator = sum(q_vec[t] * d_vec.get(t, 0.0) for t in q_vec.keys())
        q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
        d_norm = math.sqrt(sum(v * v for v in d_vec.values()))
        if q_norm == 0 or d_norm == 0:
            return 0.0
        return numerator / (q_norm * d_norm)

    def save_index(self, index_dir: Path) -> None:
        """持久化检索索引到磁盘（含 embeddings）"""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        # 保存 chunks
        chunks_data = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "text": c.text,
                "source": c.source,
                "metadata": c.metadata,
            }
            for c in self._chunks.values()
        ]
        (index_dir / "chunks.json").write_text(
            json.dumps(chunks_data, ensure_ascii=False),
            encoding="utf-8",
        )

        # 保存词频统计
        term_data = {
            "doc_freq": dict(self._doc_freq),
            "chunk_terms": self._chunk_terms,
            "chunk_lengths": self._chunk_lengths,
        }
        (index_dir / "terms.json").write_text(
            json.dumps(term_data, ensure_ascii=False),
            encoding="utf-8",
        )

        # 保存 embeddings（若已有计算结果）
        if self._chunk_embeddings:
            (index_dir / "embeddings.json").write_text(
                json.dumps(self._chunk_embeddings, ensure_ascii=False),
                encoding="utf-8",
            )

    def load_index(self, index_dir: Path) -> bool:
        """从磁盘加载检索索引，返回是否成功"""
        index_dir = Path(index_dir)
        chunks_file = index_dir / "chunks.json"
        terms_file = index_dir / "terms.json"

        if not chunks_file.exists() or not terms_file.exists():
            return False

        try:
            chunks_data = json.loads(chunks_file.read_text(encoding="utf-8"))
            self._chunks = {
                c["chunk_id"]: DocumentChunk(
                    chunk_id=c["chunk_id"],
                    doc_id=c["doc_id"],
                    text=c["text"],
                    source=c["source"],
                    metadata=c.get("metadata", {}),
                )
                for c in chunks_data
            }

            term_data = json.loads(terms_file.read_text(encoding="utf-8"))
            self._doc_freq = defaultdict(int, term_data.get("doc_freq", {}))
            self._chunk_terms = term_data.get("chunk_terms", {})
            self._chunk_lengths = term_data.get("chunk_lengths", {})
            # 重建数值索引
            self._chunk_numbers = {
                cid: self._extract_numbers(self._chunks[cid].text)
                for cid in self._chunks
            }

            # 尝试加载 embeddings（若文件不存在，后续 ensure_embeddings_loaded 会惰性计算）
            embeddings_file = index_dir / "embeddings.json"
            if embeddings_file.exists():
                try:
                    embeddings_data = json.loads(embeddings_file.read_text(encoding="utf-8"))
                    # JSON 的 list 值在反序列化后已是 list，无需转换
                    self._chunk_embeddings = embeddings_data
                except (json.JSONDecodeError, ValueError):
                    self._chunk_embeddings.clear()

            return True
        except (json.JSONDecodeError, KeyError):
            return False

    def ensure_embeddings_loaded(self) -> None:
        """确保 embeddings 已加载（惰性计算），首次计算后自动持久化到磁盘"""
        if self._chunk_embeddings or not self._embedding_provider or not self._chunks:
            return
        try:
            texts = [c.text for c in self._chunks.values()]
            chunk_ids = list(self._chunks.keys())
            vectors = self._embedding_provider.embed_texts(texts)
            self._chunk_embeddings.clear()
            for idx, chunk_id in enumerate(chunk_ids):
                if idx < len(vectors):
                    self._chunk_embeddings[chunk_id] = vectors[idx]
            # 首次计算后自动持久化
            if self._index_dir and self._chunk_embeddings:
                self.save_index(self._index_dir)
        except Exception:
            self._chunk_embeddings.clear()

