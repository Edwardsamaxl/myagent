"""TraceRecord 数据结构。"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class TraceStage(str, Enum):
    """追踪阶段枚举（与 RagAgentService 的 stage 保持一致）。"""

    ROUTING = "routing"
    RAG_REWRITE = "rag_rewrite"
    RAG_RETRIEVAL = "rag_retrieval"
    RAG_RERANK = "rag_rerank"
    RAG_ANSWER = "rag_answer"
    INGESTION_START = "ingestion_start"
    INGESTION_DONE = "ingestion_done"


@dataclass
class TraceEvent:
    """单条追踪事件。"""
    trace_id: str
    stage: TraceStage
    message: str
    payload: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "stage": self.stage.value,
            "message": self.message,
            "payload": self.payload,
            "timestamp": datetime.utcnow().isoformat(),
        }


class TraceLogger:
    """轻量级追踪日志写入器（JSONL）。"""

    def __init__(self, trace_file: Path | str | None = None) -> None:
        self._file = Path(trace_file) if trace_file else Path("data/routing_observability/traces.jsonl")
        self._file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: TraceEvent) -> None:
        with self._file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")


class RouteQuality(str, Enum):
    """路由质量评级。"""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


@dataclass
class LatencyRecord:
    """各阶段延迟记录（毫秒）。"""

    rewrite_ms: int = 0
    retrieval_ms: int = 0
    rerank_ms: int = 0
    routing_ms: int = 0
    total_ms: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "rewrite_ms": self.rewrite_ms,
            "retrieval_ms": self.retrieval_ms,
            "rerank_ms": self.rerank_ms,
            "routing_ms": self.routing_ms,
            "total_ms": self.total_ms,
        }


@dataclass
class TraceRecord:
    """完整追踪记录。"""

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    query: str = ""
    history: list[str] = field(default_factory=list)
    route_type: str = ""
    route_reasoning: str = ""
    route_confidence: float = 0.0
    selected_tools: list[str] = field(default_factory=list)
    rag_config: dict[str, Any] | None = None
    latency: LatencyRecord = field(default_factory=LatencyRecord)
    retrieval_hits_count: int = 0
    retrieval_top_score: float = 0.0
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    quality: RouteQuality | None = None
    quality_score: float | None = None
    quality_reasoning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "query": self.query,
            "history": self.history,
            "route_type": self.route_type,
            "route_reasoning": self.route_reasoning,
            "route_confidence": round(self.route_confidence, 4),
            "selected_tools": self.selected_tools,
            "rag_config": self.rag_config,
            "latency": self.latency.to_dict(),
            "retrieval_hits_count": self.retrieval_hits_count,
            "retrieval_top_score": round(self.retrieval_top_score, 6),
            "answer": self.answer[:500] + "..." if len(self.answer) > 500 else self.answer,
            "citations": self.citations,
            "quality": self.quality.value if self.quality else None,
            "quality_score": round(self.quality_score, 4) if self.quality_score is not None else None,
            "quality_reasoning": self.quality_reasoning,
        }

    def to_jsonl(self) -> str:
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False)
