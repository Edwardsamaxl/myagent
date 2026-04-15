"""RAG pipeline: retrieval → rerank → generation → evidence formatting."""

from .schemas import DocumentChunk, RetrievalHit, GenerationResult
from .retrieval import InMemoryHybridRetriever
from .rerank import (
    SimpleReranker,
    BGEReranker,
    HuggingFaceReranker,
    CascadeReranker,
    build_reranker,
)
from .ingestion import DocumentIngestionPipeline, IngestionResult, derive_doc_metadata_from_source
from .generation import GroundedGenerator
from .evidence_format import (
    format_citation_lines,
    format_evidence_block_from_hits,
    format_evidence_block_from_api_dicts,
    citations_are_valid,
    citation_ids_for_hits,
    contains_citation_marker,
    evaluate_anchor_coverage,
    extract_citation_indices,
    normalize_structured_answer,
    pick_key_evidence_snippet,
    select_evidence_hits,
    is_low_information_snippet,
)

__all__ = [
    # Schemas
    "DocumentChunk",
    "RetrievalHit",
    "GenerationResult",
    # Retrieval
    "InMemoryHybridRetriever",
    # Rerank
    "SimpleReranker",
    "BGEReranker",
    "HuggingFaceReranker",
    "CascadeReranker",
    "build_reranker",
    # Ingestion
    "DocumentIngestionPipeline",
    "IngestionResult",
    "derive_doc_metadata_from_source",
    # Generation
    "GroundedGenerator",
    # Evidence format
    "format_citation_lines",
    "format_evidence_block_from_hits",
    "format_evidence_block_from_api_dicts",
    "citations_are_valid",
    "citation_ids_for_hits",
    "contains_citation_marker",
    "evaluate_anchor_coverage",
    "extract_citation_indices",
    "normalize_structured_answer",
    "pick_key_evidence_snippet",
    "select_evidence_hits",
    "is_low_information_snippet",
]
