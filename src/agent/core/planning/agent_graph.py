from __future__ import annotations

from typing import Any

from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import should_continue


def build_agent_graph(
    model: Any,
    tools: list[Any],
) -> Any:
    """构建 LangGraph ReAct Agent。

    使用 langgraph.prebuilt.create_react_agent，
    内部自动处理 Tool-calling 协议的解析与执行：
    - 模型输出结构化 tool_call（MiniMax-M2.7 native tool-calling）
    - 框架解析 {name, arguments}
    - 执行工具并注入 ToolMessage
    - 循环直到模型不再调用工具

    参数:
        model: LangChain model provider（需要支持 tool_calling）
        tools: LangChain tool 对象列表（来自 nodes._make_langchain_tools）
    """

    # create_react_agent 返回一个 CompiledGraph（AgentExecutor）
    # 注：max_iterations 在新版 langgraph 中已移除，通过 agent.invoke 的 config 控制
    agent = create_react_agent(
        model=model,
        tools=tools,
        state_schema=AgentState,
        debug=False,
    )
    return agent


def run_agent(agent: Any, user_query: str, history_messages: list[dict] | None = None) -> dict[str, Any]:
    """在 user_query 上运行 agent，返回最终结果。

    参数:
        agent: build_agent_graph 返回的 CompiledGraph
        user_query: 用户问题
        history_messages: 可选的对话历史

    返回:
        包含 "answer" (str) 和 "messages" (list) 的 dict
    """
    messages = history_messages.copy() if history_messages else []
    messages.append({"role": "user", "content": user_query})

    result = agent.invoke({"messages": messages})

    # 从最终 state 中提取回复
    final_messages = result.get("messages", [])
    last_message = final_messages[-1] if final_messages else None

    answer = ""
    if hasattr(last_message, "content"):
        answer = last_message.content
    elif isinstance(last_message, dict):
        answer = last_message.get("content", "")

    return {
        "answer": answer,
        "messages": final_messages,
    }
