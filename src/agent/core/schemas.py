from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    text: str
    source: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class RetrievalHit:
    chunk_id: str
    score: float
    text: str
    source: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class GenerationResult:
    answer: str
    citations: list[str]
    refusal: bool
    reason: str = ""


@dataclass
class EvalRecord:
    trace_id: str
    question: str
    answer: str
    references: list[str]
    latency_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

