from __future__ import annotations

from typing import Any, Callable

from .schemas import Tool, ToolSchema, ToolSource
from .context import ToolUseContext
import inspect


def build_tool(
    name: str,
    description: str,
    handler: Callable[..., str],
    *,
    input_schema: dict | None = None,
    output_schema: dict | None = None,
    source: ToolSource = ToolSource.CUSTOM,
    mcp_server: str | None = None,
) -> Tool:
    schema = ToolSchema(
        name=name,
        description=description,
        input_schema=input_schema or {"type": "object", "properties": {}},
        output_schema=output_schema,
    )
    normalized = _normalize_handler(handler)
    return Tool(
        schema=schema,
        source=source,
        handler=normalized,
        mcp_server=mcp_server,
    )


def _normalize_handler(handler: Callable[..., str]) -> Callable[..., str]:
    sig = inspect.signature(handler)
    params = list(sig.parameters.values())

    if len(params) == 1 and params[0].name == "ctx":
        return handler

    if len(params) == 1:
        first_name = params[0].name
        def wrapper(ctx: ToolUseContext, **kwargs) -> str:
            if first_name in kwargs:
                return handler(kwargs[first_name])
            args = [v for k, v in kwargs.items() if k != "ctx"]
            return handler(args[0]) if args else handler("")
        return wrapper

    def wrapper(ctx: ToolUseContext, **kwargs) -> str:
        filtered = {k: v for k, v in kwargs.items() if k != "ctx"}
        return handler(**filtered)
    return wrapper
