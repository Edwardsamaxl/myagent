from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool as langchain_tool

from .state import AgentState


def _make_langchain_tools(tools: dict[str, Any]) -> list[Any]:
    """把 registry.Tool dict 转换为 LangChain tool 对象。

    每个 Tool.func 是 Callable[[str], str]，直接转发即可。
    """

    @langchain_tool
    def dynamic_tool(query: str, name: str, description: str, func: Any) -> str:
        """动态工具包装器，实际执行委托给 func。"""
        return func(query)

    result = []
    for name, t in tools.items():
        # 用 @langchain_tool 装饰器包装
        tool_def = {"name": t.name, "description": t.description, "func": t.func}
        # 用 functools.partial 绑定参数
        import functools
        import inspect

        sig = inspect.signature(t.func)
        params = list(sig.parameters.keys())

        def make_wrapper(tool_name: str, tool_desc: str, tool_func: Any):
            @langchain_tool(name=tool_name, description=tool_desc)
            def wrapper(query: str) -> str:
                return tool_func(query)

            return wrapper

        wrapped = make_wrapper(t.name, t.description, t.func)
        result.append(wrapped)

    return result


def tools_node(state: AgentState) -> dict[str, Any]:
    """本节点由 LangGraph prebuilt ReAct agent 内部使用。

    这里声明是为了让 graph builder 能引用到，实际执行由 prebuilt agent 接管。
    """
    return state


def increment_step(state: AgentState) -> dict[str, int]:
    """每轮 ReAct 循环后 step +1。"""
    current = state.get("step", 0)
    return {"step": current + 1}


def should_continue(state: AgentState) -> str:
    """ReAct 循环终止条件：无 tool_calls（即模型直接回复）或达到 MAX_STEPS。

    LangGraph 的 prebuilt create_react_agent 会自己管理循环，
    此函数仅用于安全上限检测（当被外部调用时）。
    """
    if state.get("final_answer"):
        return "end"
    return "continue"
