"""RAG Chain with observability - query rewrite, retrieval, rerank."""

from __future__ import annotations

import time
from typing import Any

from ...core.dialogue.intent_classifier import classify_intent
from ...core.dialogue.query_rewrite import QueryRewriteResult, rewrite_for_rag
from ...rag.rerank import SimpleReranker
from ...rag.retrieval import InMemoryHybridRetriever
from ...rag.schemas import RetrievalHit
from ..observability.trace_record import TraceRecord, LatencyRecord, TraceLogger
from ..observability.trace_store import TraceStore


class RAGChain:
    """RAG execution chain with per-stage latency tracking."""

    def __init__(
        self,
        config: dict[str, Any],
        trace_store: TraceStore | None = None,
        retriever: InMemoryHybridRetriever | None = None,
        reranker: SimpleReranker | None = None,
        model_provider: Any | None = None,
    ) -> None:
        self.config = config
        self.trace_store = trace_store or TraceStore()
        self._trace_logger: TraceLogger | None = None
        self._retriever = retriever
        self._reranker = reranker or SimpleReranker()
        self._model_provider = model_provider

    def set_trace_logger(self, logger: TraceLogger) -> None:
        """Attach a trace logger for event emission."""
        self._trace_logger = logger

    def execute(self, query: str, rewrite_mode: str = "hybrid") -> dict[str, Any]:
        """Execute full RAG chain: rewrite -> retrieval -> rerank, with latency tracking."""
        latency: dict[str, int] = {
            "rewrite_ms": 0,
            "retrieval_ms": 0,
            "rerank_ms": 0,
            "total_ms": 0,
        }

        # 1. Query rewrite
        start = time.time()
        rewritten = self._rewrite(query, mode=rewrite_mode)
        latency["rewrite_ms"] = int((time.time() - start) * 1000)

        # 2. Retrieval
        start = time.time()
        hits = self._retrieve(rewritten)
        latency["retrieval_ms"] = int((time.time() - start) * 1000)

        # 3. Rerank
        start = time.time()
        reranked = self._rerank(rewritten, hits)
        latency["rerank_ms"] = int((time.time() - start) * 1000)

        latency["total_ms"] = sum(latency.values())

        return {
            "hits": reranked,
            "latency": latency,
            "rewritten_query": rewritten,
        }

    def _rewrite(self, query: str, mode: str) -> str:
        """Apply query rewrite using actual query_rewrite component."""
        try:
            intent = classify_intent(query, None)
            result = rewrite_for_rag(
                turn_text=query,
                history=None,
                intent=intent,
                model=self._model_provider,
                mode=mode,
            )
            return str(result) if isinstance(result, QueryRewriteResult) else result
        except Exception:
            return query

    def _retrieve(self, query: str) -> list[RetrievalHit]:
        """Retrieve relevant chunks using actual retrieval component."""
        if self._retriever is None:
            return []
        try:
            top_k = self.config.get("retrieval_top_k", 5) if isinstance(self.config, dict) else 5
            return self._retriever.search(query, top_k=top_k)
        except Exception:
            return []

    def _rerank(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        """Rerank retrieved hits using actual reranker."""
        if not hits:
            return hits
        try:
            top_k = self.config.get("rerank_top_k", 3) if isinstance(self.config, dict) else 3
            return self._reranker.rerank(query=query, hits=hits, top_k=top_k)
        except Exception:
            return hits
