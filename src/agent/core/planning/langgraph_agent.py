from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from ...config import AgentConfig
from ...llm.providers import ModelProvider
from ...tools.registry import Tool
from .agent_graph import build_agent_graph, run_agent
from .nodes import _make_langchain_tools

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    answer: str
    steps_used: int
    messages: list[dict]
    tool_calls: list[str]


def _convert_langchain_tools_to_anthropic(langchain_tools: list[Any]) -> list[dict]:
    """把 LangChain tool 对象列表转成 Anthropic tool-calling 格式。"""
    anthropic_tools = []
    for t in langchain_tools:
        name = getattr(t, "name", None) or getattr(t, "tool_name", None)
        description = getattr(t, "description", "")
        schema = getattr(t, "args_schema", None)
        if schema is None:
            schema = getattr(t, "schema", {})

        if hasattr(schema, "model_json_schema"):
            input_schema = schema.model_json_schema()
        elif isinstance(schema, dict):
            input_schema = schema
        else:
            input_schema = {"type": "object", "properties": {}}

        anthropic_tools.append({
            "name": name,
            "description": description,
            "input_schema": input_schema,
        })
    return anthropic_tools


def _wrap_model_provider(model: ModelProvider) -> Any:
    """把项目自有的 ModelProvider 包装成 LangChain BaseChatModel。"""
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class ModelProviderWrapper(BaseChatModel):
        """把 ModelProvider 适配成 LangChain BaseChatModel。

        直接处理 MiniMax-M2.7 的 tool_use 响应格式，
        将 tool_use 转换为 LangChain 的 ToolCall 格式。
        """

        def __init__(self, inner: ModelProvider) -> None:
            super().__init__()
            self._inner = inner
            self._bound_tools: list[dict] = []

        def _to_lc_messages(self, messages: list[BaseMessage]) -> list[dict]:
            result = []
            for m in messages:
                role = "user" if m.type == "human" else "assistant"
                result.append({"role": role, "content": m.content})
            return result

        @property
        def _llm_type(self) -> str:
            return "model-provider-wrapper"

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: Optional[list[str]] = None,
            **kwargs: Any,
        ) -> ChatResult:
            lc_messages = self._to_lc_messages(messages)

            # 使用 generate_raw 获取完整响应（包含 tool_use 信息）
            raw_response = self._inner.generate_raw(
                messages=lc_messages,
                tools=self._bound_tools if self._bound_tools else None,
                temperature=kwargs.get("temperature", 0.2),
                max_tokens=kwargs.get("max_tokens", 768),
            )

            # raw_response 是完整的 API 响应 dict
            # 检查是否包含 tool_use 信息
            tool_use = None
            content_list = raw_response.get("content", [])
            for item in content_list:
                if isinstance(item, dict) and item.get("type") == "tool_use":
                    tool_use = item
                    break

            if tool_use:
                # 构建带 tool_call 的 AIMessage
                tool_name = tool_use.get("name", "")
                tool_input = tool_use.get("input", {})
                tool_id = tool_use.get("id", "")

                # 使用 LangChain 的 ToolCall 格式
                from langchain_core.messages import ToolCall
                tc = ToolCall(
                    name=tool_name,
                    args=tool_input if isinstance(tool_input, dict) else {},
                    id=tool_id,
                )
                ai_msg = AIMessage(
                    content="",
                    tool_calls=[tc],
                )
            else:
                # 普通文本响应：提取 text 类型的内容
                text_content = ""
                for item in content_list:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_content = item.get("text", "")
                        break
                if not text_content:
                    # fallback: 使用 stop_reason 或空字符串
                    text_content = raw_response.get("stop_reason", "")

                ai_msg = AIMessage(content=text_content)

            chat_gen = ChatGeneration(message=ai_msg)
            return ChatResult(generations=[chat_gen])

        def bind_tools(self, tools: list[Any], **kwargs: Any) -> "ModelProviderWrapper":
            """LangGraph 要求 chat model 实现 bind_tools。

            把 LangChain tool list 转成 Anthropic 格式并存起来，
            _generate 时注入请求。
            """
            self._bound_tools = _convert_langchain_tools_to_anthropic(tools)
            return self

        def with_config(self, config: dict) -> "ModelProviderWrapper":
            return self

    return ModelProviderWrapper(model)


class LangGraphAgent:
    """基于 LangGraph + Tool-calling 的 ReAct Agent。

    使用 MiniMax-M2.7 native tool-calling 协议，
    LangGraph create_react_agent 内部自动处理结构化解码与工具执行，
    不依赖 action 字符串解析。
    """

    def __init__(
        self,
        config: AgentConfig,
        model: ModelProvider,
        tools: dict[str, Tool],
    ) -> None:
        self.config = config
        self.model = model
        self.tools = tools

        lc_tools = _make_langchain_tools(tools)
        if not lc_tools:
            logger.warning("[LangGraphAgent] 没有任何可用工具，Agent 将退化为纯生成模式。")

        lc_model = _wrap_model_provider(model)

        self._agent = build_agent_graph(
            model=lc_model,
            tools=lc_tools,
        )

    def run(
        self,
        user_input: str,
        history_messages: list[dict] | None = None,
        long_context: str = "",
    ) -> AgentResult:
        messages = []
        if long_context:
            messages.append({"role": "system", "content": f"【长期上下文】\n{long_context}\n\n【当前对话】"})

        result = run_agent(self._agent, user_input, messages)

        tool_calls = []
        lc_messages = result.get("messages", [])
        for msg in lc_messages:
            if hasattr(msg, "type") and msg.type == "tool":
                tool_calls.append(getattr(msg, "name", "unknown"))

        steps_used = len([m for m in lc_messages if hasattr(m, "type") and m.type == "tool"])

        # 转换 LangChain 消息对象为 dict（供 session_store 使用）
        def msg_to_dict(m):
            if isinstance(m, dict):
                return m
            role = "assistant" if getattr(m, "type", "") in ("ai", "tool") else getattr(m, "type", "user")
            content = getattr(m, "content", "")
            return {"role": role, "content": content}

        return AgentResult(
            answer=result.get("answer", ""),
            steps_used=steps_used,
            messages=[msg_to_dict(m) for m in lc_messages],
            tool_calls=tool_calls,
        )
