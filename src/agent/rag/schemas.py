from __future__ import annotations

from dataclasses import dataclass, field


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
