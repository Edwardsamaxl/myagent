from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class ToolSource(Enum):
    BUILTIN = "builtin"
    MCP = "mcp"
    CUSTOM = "custom"


@dataclass
class ToolSchema:
    name: str
    description: str
    input_schema: dict = field(default_factory=lambda: {"type": "object", "properties": {}})
    output_schema: dict | None = None


@dataclass
class Tool:
    schema: ToolSchema
    source: ToolSource
    handler: Callable[..., str]
    mcp_server: str | None = None
