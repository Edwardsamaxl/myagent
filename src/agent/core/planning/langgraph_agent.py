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
    """把 LangChain tool 对象列表转成 Anthropic tool-calling 格式。

    每个 LangChain tool 有 .name, .description, .args_schema (Pydantic model)。
    转成 Anthropic 的 tools 格式：
    {
      "name": "...",
      "description": "...",
      "input_schema": {...}
    }
    """
    anthropic_tools = []
    for t in langchain_tools:
        name = getattr(t, "name", None) or getattr(t, "tool_name", None)
        description = getattr(t, "description", "")
        # 取 input_schema（可能是 Pydantic model 或 dict）
        schema = getattr(t, "args_schema", None)
        if schema is None:
            # 尝试从 tool 本身取
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
    """把项目自有的 ModelProvider 包装成 LangChain BaseChatModel。

    LangGraph create_react_agent 需要 LangChain chat model 接口
    (.invoke, .bind_tools, .with_config 等)。
    """
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

    class ModelProviderWrapper(BaseChatModel):
        """把 ModelProvider 适配成 LangChain BaseChatModel。"""

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
        ) -> Any:
            lc_messages = self._to_lc_messages(messages)

            # 注入 tools 参数（如果 bind_tools 被调用过）
            extra_kwargs: dict[str, Any] = {}
            if self._bound_tools:
                extra_kwargs["tools"] = self._bound_tools

            content = self._inner.generate(
                messages=lc_messages,
                temperature=kwargs.get("temperature", 0.2),
                max_tokens=kwargs.get("max_tokens", 768),
            )

            ai_msg = AIMessage(content=content)

            # 构造 FakeStreamedGenerator 输出
            class _Result:
                generations = [[type("_Gen", (), {"message": ai_msg, "text": content})()]]
                llm_output = {}

            return _Result()

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

    工具列表：
    - get_time / calculate / read_memory / remember / list_skills
    - read_skill / save_skill / read_workspace_file / web_search
    - search_knowledge_base（RAG 工具，由 rag_service 构造）
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

        # 把 Tool dict 转成 LangChain tool
        lc_tools = _make_langchain_tools(tools)
        if not lc_tools:
            logger.warning("[LangGraphAgent] 没有任何可用工具，Agent 将退化为纯生成模式。")

        # 包装成 LangChain compatible model
        lc_model = _wrap_model_provider(model)

        self._agent = build_agent_graph(
            model=lc_model,
            tools=lc_tools,
            max_steps=config.max_steps,
        )

    def run(
        self,
        user_input: str,
        history_messages: list[dict] | None = None,
        long_context: str = "",
    ) -> AgentResult:
        """执行单轮 Agent 对话。

        long_context 作为 system prompt 注入到首条消息。
        """
        messages = []
        if long_context:
            messages.append({"role": "system", "content": f"【长期上下文】\n{long_context}\n\n【当前对话】"})

        result = run_agent(self._agent, user_input, messages)

        # 统计 tool_calls
        tool_calls = []
        for msg in result.get("messages", []):
            if hasattr(msg, "type") and msg.type == "tool":
                tool_calls.append(getattr(msg, "name", "unknown"))

        steps_used = len([m for m in result.get("messages", [])
                          if hasattr(m, "type") and m.type == "tool"])

        return AgentResult(
            answer=result.get("answer", ""),
            steps_used=steps_used,
            messages=result.get("messages", []),
            tool_calls=tool_calls,
        )
