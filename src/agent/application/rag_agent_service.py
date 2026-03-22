from __future__ import annotations

import time

from ..config import AgentConfig
from ..core.evaluation import EvaluationStore
from ..core.generation import GroundedGenerator
from ..core.ingestion import DocumentIngestionPipeline
from ..core.observability import TraceEvent, TraceLogger
from ..core.retrieval import InMemoryHybridRetriever
from ..core.rerank import SimpleReranker
from ..core.schemas import EvalRecord
from ..llm.providers import ModelProvider


class RagAgentService:
    """RAG + Agent framework skeleton for incremental module upgrades."""

    def __init__(self, config: AgentConfig, model: ModelProvider) -> None:
        self.config = config
        self.ingestion = DocumentIngestionPipeline(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        self.retriever = InMemoryHybridRetriever()
        self.reranker = SimpleReranker()
        self.generator = GroundedGenerator(
            model=model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        self.eval_store = EvaluationStore(config.eval_records_file)
        self.trace_logger = TraceLogger(config.trace_file)

    def update_model(self, model: ModelProvider) -> None:
        self.generator = GroundedGenerator(
            model=model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

    def ingest_document(self, doc_id: str, source: str, content: str) -> dict[str, int | str]:
        trace_id = self.trace_logger.new_trace_id()
        self.trace_logger.log(
            TraceEvent(
                trace_id=trace_id,
                stage="ingestion_start",
                message="Start document ingestion",
                payload={"doc_id": doc_id, "source": source},
            )
        )
        chunks, result = self.ingestion.ingest_text(doc_id=doc_id, source=source, content=content)
        self.retriever.upsert_chunks(chunks)
        self.trace_logger.log(
            TraceEvent(
                trace_id=trace_id,
                stage="ingestion_done",
                message="Ingestion completed",
                payload={
                    "total_chunks": str(result.total_chunks),
                    "deduplicated_chunks": str(result.deduplicated_chunks),
                },
            )
        )
        return {
            "trace_id": trace_id,
            "doc_id": result.doc_id,
            "source": result.source,
            "total_chunks": result.total_chunks,
            "deduplicated_chunks": result.deduplicated_chunks,
        }

    def answer(self, question: str, top_k: int | None = None) -> dict[str, object]:
        trace_id = self.trace_logger.new_trace_id()
        start = time.perf_counter()
        effective_top_k = top_k or self.config.retrieval_top_k
        hits = self.retriever.search(question, top_k=effective_top_k)
        reranked_hits = self.reranker.rerank(
            query=question,
            hits=hits,
            top_k=self.config.rerank_top_k,
        )
        generation = self.generator.generate(question=question, hits=reranked_hits)
        latency_ms = int((time.perf_counter() - start) * 1000)

        self.trace_logger.log(
            TraceEvent(
                trace_id=trace_id,
                stage="rag_answer",
                message="RAG request completed",
                payload={
                    "retrieved": str(len(hits)),
                    "reranked": str(len(reranked_hits)),
                    "refusal": str(generation.refusal),
                    "latency_ms": str(latency_ms),
                },
            )
        )
        self.eval_store.append(
            EvalRecord(
                trace_id=trace_id,
                question=question,
                answer=generation.answer,
                references=generation.citations,
                latency_ms=latency_ms,
            )
        )
        return {
            "trace_id": trace_id,
            "answer": generation.answer,
            "refusal": generation.refusal,
            "reason": generation.reason,
            "citations": generation.citations,
            "retrieval_hits": [
                {
                    "chunk_id": hit.chunk_id,
                    "score": hit.score,
                    "source": hit.source,
                    "text_preview": hit.text[:180],
                }
                for hit in reranked_hits
            ],
        }

    def get_metrics(self) -> dict[str, float | int]:
        return self.eval_store.summary()

