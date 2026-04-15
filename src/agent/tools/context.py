from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import AgentConfig
    from ..llm.providers import ModelProvider
    from .registry import ToolRegistry


@dataclass
class ToolUseContext:
    config: "AgentConfig"
    llm: "ModelProvider"
    registry: "ToolRegistry"
    session_id: str = ""
    session_history: list[dict[str, str]] | None = None
