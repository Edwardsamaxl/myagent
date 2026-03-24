from __future__ import annotations

import math
import os
import re
from collections import defaultdict

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
            # 中文仅加入 bi-gram / tri-gram，不保留原始长串 token
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
    ) -> None:
        sparse_mode = os.getenv("SPARSE_MODE", "tfidf").strip().lower()
        if sparse_mode not in {"tfidf", "bm25", "tfidf_bm25"}:
            sparse_mode = "tfidf"
        self._chunks: dict[str, DocumentChunk] = {}
        self._doc_freq: defaultdict[str, int] = defaultdict(int)
        self._chunk_terms: dict[str, dict[str, int]] = {}
        self._chunk_lengths: dict[str, int] = {}
        self._chunk_embeddings: dict[str, list[float]] = {}
        self._embedding_provider = embedding_provider
        self._fusion_mode = fusion_mode if fusion_mode in {"weighted_sum", "rrf"} else "weighted_sum"
        self._lexical_weight = lexical_weight
        self._tfidf_weight = tfidf_weight
        self._embedding_weight = embedding_weight
        self._embedding_top_k = max(1, embedding_top_k)
        self._sparse_mode = sparse_mode
        self._bm25_k1 = float(os.getenv("BM25_K1", "1.5"))
        self._bm25_b = float(os.getenv("BM25_B", "0.75"))
        self._tfidf_bm25_alpha = float(os.getenv("TFIDF_BM25_ALPHA", "0.5"))

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

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        hits, _ = self.search_with_debug(query=query, top_k=top_k)
        return hits

    def search_with_debug(self, query: str, top_k: int = 5) -> tuple[list[RetrievalHit], dict[str, object]]:
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

            score = (
                self._lexical_weight * lexical
                + self._tfidf_weight * sparse
                + self._embedding_weight * embedding
            )
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
        weighted_scores.sort(key=lambda item: item[1], reverse=True)

        if self._fusion_mode == "rrf":
            self._apply_rrf(rrf_scores, lexical_rank, self._lexical_weight)
            self._apply_rrf(rrf_scores, tfidf_rank, self._tfidf_weight)
            self._apply_rrf(
                rrf_scores,
                embedding_rank[: self._embedding_top_k],
                self._embedding_weight,
            )
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
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            denom = tf + k1 * (1.0 - b + b * (dl / max(1e-9, avgdl)))
            score += idf * ((tf * (k1 + 1.0)) / max(1e-9, denom))
        return score

    def _avg_doc_len(self) -> float:
        if not self._chunk_lengths:
            return 1.0
        return sum(self._chunk_lengths.values()) / max(1, len(self._chunk_lengths))

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
        q_vec: dict[str, float] = {}
        d_vec: dict[str, float] = {}

        for term, freq in q_tf.items():
            idf = math.log((n_docs + 1) / (1 + self._doc_freq.get(term, 0))) + 1.0
            q_vec[term] = freq * idf
        for term, freq in d_tf.items():
            idf = math.log((n_docs + 1) / (1 + self._doc_freq.get(term, 0))) + 1.0
            d_vec[term] = freq * idf

        numerator = sum(q_vec[t] * d_vec.get(t, 0.0) for t in q_vec.keys())
        q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
        d_norm = math.sqrt(sum(v * v for v in d_vec.values()))
        if q_norm == 0 or d_norm == 0:
            return 0.0
        return numerator / (q_norm * d_norm)

