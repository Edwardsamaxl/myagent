from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class IntentKind(str, Enum):
    """对话意图（规则分类初版，后续可接模型 JSON）。"""

    KNOWLEDGE_CORPUS = "knowledge_corpus"
    TOOL_ONLY = "tool_only"
    MIXED = "mixed"
    CHITCHAT = "chitchat"
    AMBIGUOUS = "ambiguous"
    UNSAFE_OR_REFUSE = "unsafe_or_refuse"


@dataclass
class IntentResult:
    intent: IntentKind
    confidence: float
    normalized_query: str | None = None
    slots: dict[str, str] = field(default_factory=dict)
    clarify_prompt: str | None = None
