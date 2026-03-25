from __future__ import annotations

import json
import os
import time

from ..config import AgentConfig
from ..core.evaluation import EvaluationStore
from ..core.generation import GroundedGenerator
from ..core.ingestion import DocumentIngestionPipeline
from ..core.observability import TraceEvent, TraceLogger, TraceStage
from ..core.retrieval import InMemoryHybridRetriever
from ..core.rerank import SimpleReranker
from ..core.schemas import EvalRecord
from ..llm.embeddings import build_embedding_provider
from ..llm.providers import ModelProvider


class RagAgentService:
    """RAG 编排：检索 → 重排 → 生成；`retrieval_hits` 与 `GroundedGenerator` 使用同一套 `RetrievalHit` 结构。

    若 `retrieval_hits` 的字段或顺序变更，须同步 `core/evidence_format.py` 与生成侧 prompt（与检索/协调方对齐后再改）。
    """

    def __init__(self, config: AgentConfig, model: ModelProvider) -> None:
        self.config = config
        self.ingestion = DocumentIngestionPipeline(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        self.retriever = InMemoryHybridRetriever(
            embedding_provider=build_embedding_provider(config),
            fusion_mode=config.retrieval_fusion_mode,
            lexical_weight=config.retrieval_lexical_weight,
            tfidf_weight=config.retrieval_tfidf_weight,
            embedding_weight=config.retrieval_embedding_weight,
            embedding_top_k=config.embedding_top_k,
        )
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

    def ingest_document(
        self,
        doc_id: str,
        source: str,
        content: str,
        doc_metadata: dict[str, str] | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        dedup_across_docs: bool = False,
    ) -> dict[str, int | str]:
        trace_id = self.trace_logger.new_trace_id()
        payload: dict[str, str] = {"doc_id": doc_id, "source": source}
        if doc_metadata:
            payload["metadata_keys"] = ",".join(sorted(doc_metadata.keys()))
        effective_chunk_size = chunk_size if chunk_size is not None else self.config.chunk_size
        effective_chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else self.config.chunk_overlap
        )
        payload["chunk_size"] = str(effective_chunk_size)
        payload["chunk_overlap"] = str(effective_chunk_overlap)
        payload["dedup_across_docs"] = str(dedup_across_docs)
        self.trace_logger.log(
            TraceEvent(
                trace_id=trace_id,
                stage=TraceStage.INGESTION_START,
                message="Start document ingestion",
                payload=payload,
            )
        )
        ingestion = self.ingestion
        if chunk_size is not None or chunk_overlap is not None:
            ingestion = DocumentIngestionPipeline(
                chunk_size=effective_chunk_size,
                chunk_overlap=effective_chunk_overlap,
            )
        chunks, result = ingestion.ingest_text(
            doc_id=doc_id,
            source=source,
            content=content,
            doc_metadata=doc_metadata,
            dedup_across_docs=dedup_across_docs,
        )
        self.retriever.upsert_chunks(chunks)
        self.trace_logger.log(
            TraceEvent(
                trace_id=trace_id,
                stage=TraceStage.INGESTION_DONE,
                message="Ingestion completed",
                payload={
                    "total_chunks": str(result.total_chunks),
                    "deduplicated_chunks": str(result.deduplicated_chunks),
                    "dropped_duplicates": str(result.dropped_duplicates),
                },
            )
        )
        return {
            "trace_id": trace_id,
            "doc_id": result.doc_id,
            "source": result.source,
            "total_chunks": result.total_chunks,
            "deduplicated_chunks": result.deduplicated_chunks,
            "dropped_duplicates": result.dropped_duplicates,
        }

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        *,
        append_to_eval_store: bool = True,
    ) -> dict[str, object]:
        """返回中的 `retrieval_hits` 与 `citations` 序号一致；`citations` 行格式见 `format_evidence_block_from_hits` / `format_citation_lines`。

        `append_to_eval_store=False` 时仍写入 trace，但不追加 `eval_records.jsonl`（供离线批评估避免污染在线聚合）。
        reason 语义约定（与检索侧对齐）：
        - `no_retrieval_hit`: 重排后无可用命中；
        - `insufficient_evidence`: 有命中，但证据不足以支持可靠回答。
        """
        trace_id = self.trace_logger.new_trace_id()
        start = time.perf_counter()
        effective_top_k = top_k or self.config.retrieval_top_k
        hits, retrieval_debug = self.retriever.search_with_debug(question, top_k=effective_top_k)
        reranked_hits, rerank_debug = self.reranker.rerank_with_debug(
            query=question,
            hits=hits,
            top_k=self.config.rerank_top_k,
        )
        generation = self.generator.generate(question=question, hits=reranked_hits)
        latency_ms = int((time.perf_counter() - start) * 1000)

        self.trace_logger.log(
            TraceEvent(
                trace_id=trace_id,
                stage=TraceStage.RAG_ANSWER,
                message="RAG request completed",
                payload={
                    "retrieved": str(len(hits)),
                    "reranked": str(len(reranked_hits)),
                    "refusal": str(generation.refusal),
                    "latency_ms": str(latency_ms),
                    "query_tokens": " ".join(str(x) for x in retrieval_debug.get("query_tokens", [])),
                    "top_scores": ",".join(str(x) for x in retrieval_debug.get("top_scores", [])),
                    "zero_score_reasons": json.dumps(
                        retrieval_debug.get("zero_score_reasons", {}),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "fusion_mode": str(retrieval_debug.get("fusion_mode", "")),
                    "score_breakdown": json.dumps(
                        retrieval_debug.get("score_breakdown", []),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "rerank_score_breakdown": json.dumps(
                        rerank_debug,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            )
        )
        if append_to_eval_store:
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
            "latency_ms": latency_ms,
            "answer": generation.answer,
            "refusal": generation.refusal,
            "reason": generation.reason,
            "citations": generation.citations,
            "retrieval_hits": [
                {
                    "chunk_id": hit.chunk_id,
                    "score": hit.score,
                    "source": hit.source,
                    "metadata": hit.metadata,
                    # 供对话层用作“证据上下文”的截断预览长度。
                    # 过短会导致数值/关键句被截断，进而出现 rag.answer 正确但 chat.answer 无关/拒答的现象。
                    "text_preview": hit.text[: int(os.getenv("RAG_HIT_TEXT_PREVIEW_CHARS", "500"))],
                }
                for hit in reranked_hits
            ],
        }

    def get_metrics(self) -> dict[str, float | int | None]:
        return self.eval_store.summary()

