from __future__ import annotations

from typing import Any, Callable

from langchain_core.tools import BaseTool, StructuredTool

from .state import AgentState


def _make_langchain_tools(tools: dict[str, Any]) -> list[BaseTool]:
    """把 registry.Tool dict 转换为 LangChain BaseTool 对象。

    每个 Tool.func 是 Callable[[str], str]，直接转发。
    使用 StructuredTool.from_function 以兼容新版 langchain-core。
    """

    def make_tool(name: str, description: str, func: Callable[[str], str]) -> BaseTool:
        return StructuredTool.from_function(
            name=name,
            description=description,
            func=func,
        )

    result = []
    for name, t in tools.items():
        wrapped = make_tool(t.name, t.description, t.func)
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
