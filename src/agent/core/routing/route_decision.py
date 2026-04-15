"""RouteDecision data structure."""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any


class RouteType(str, Enum):
    SINGLE_STEP = "single_step"
    MULTI_STEP = "multi_step"
    CLARIFY = "clarify"


@dataclass
class RouteDecision:
    route_type: RouteType
    selected_tools: list[str] = field(default_factory=list)
    rag_chain: dict[str, Any] | None = None  # e.g. {"rewrite_mode": "hyde", "top_k": 10}
    reasoning: str = ""
    confidence: float = 0.0
    router_latency_ms: int = 0
