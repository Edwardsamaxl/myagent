from __future__ import annotations

import math
import re
from collections import defaultdict

from .schemas import DocumentChunk, RetrievalHit


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", text.lower()) if t]


class InMemoryHybridRetriever:
    """
    MVP hybrid retriever:
    - lexical score: token overlap
    - semantic score: simplified tf-idf cosine
    """

    def __init__(self) -> None:
        self._chunks: dict[str, DocumentChunk] = {}
        self._doc_freq: defaultdict[str, int] = defaultdict(int)
        self._chunk_terms: dict[str, dict[str, int]] = {}

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
            terms = _tokenize(chunk.text)
            tf: dict[str, int] = defaultdict(int)
            for term in terms:
                tf[term] += 1
            self._chunk_terms[chunk.chunk_id] = dict(tf)
        self._rebuild_df()

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        if not query.strip() or not self._chunks:
            return []
        top_k = max(1, top_k)
        q_terms = _tokenize(query)
        q_tf: dict[str, int] = defaultdict(int)
        for t in q_terms:
            q_tf[t] += 1

        scores: list[tuple[str, float]] = []
        for chunk_id, chunk in self._chunks.items():
            lexical = self._lexical_score(q_terms, chunk.text)
            semantic = self._cosine_tfidf(q_tf, self._chunk_terms.get(chunk_id, {}))
            score = 0.45 * lexical + 0.55 * semantic
            if score > 0:
                scores.append((chunk_id, score))
        scores.sort(key=lambda item: item[1], reverse=True)

        hits: list[RetrievalHit] = []
        for chunk_id, score in scores[:top_k]:
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
        return hits

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

