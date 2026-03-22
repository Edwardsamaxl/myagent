from __future__ import annotations

from .schemas import RetrievalHit


class SimpleReranker:
    """MVP reranker: secondary score by text length balance and keyword hints."""

    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int = 3) -> list[RetrievalHit]:
        if not hits:
            return []
        query_terms = {term for term in query.lower().split() if term}

        scored: list[tuple[RetrievalHit, float]] = []
        for hit in hits:
            keyword_bonus = sum(1 for t in query_terms if t and t in hit.text.lower()) * 0.03
            length_penalty = 0.0 if 120 <= len(hit.text) <= 800 else 0.05
            final_score = hit.score + keyword_bonus - length_penalty
            scored.append((hit, final_score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return [item[0] for item in scored[: max(1, top_k)]]

