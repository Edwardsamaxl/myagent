from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph import add_messages


class AgentState(TypedDict, total=False):
    """LangGraph Agent 的状态定义。

    新版 langgraph 要求 remaining_steps 字段。
    LangGraph 的 add_messages 注解会自动处理消息的追加逻辑，
    保证同一条消息不会被重复添加。
    """

    messages: Annotated[list[dict], add_messages]
    remaining_steps: int
    step: int
    final_answer: str | None
    tool_call_results: list[str]
