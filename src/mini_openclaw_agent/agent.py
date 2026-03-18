from __future__ import annotations

import json
from dataclasses import dataclass

from .config import AgentConfig
from .providers import Message, ModelProvider
from .tools import Tool


SYSTEM_PROMPT = """你是一个可调用工具的助手。
你有以下工具可用（name: description）：
{tool_desc}

当你需要调用工具时，请严格只输出 JSON，格式如下：
{{"tool":"工具名","input":"传给工具的字符串参数"}}

当你已经可以直接回答用户时，请输出自然语言答案，不要输出 JSON。
"""


@dataclass
class AgentResult:
    answer: str
    steps_used: int


class SimpleAgent:
    def __init__(
        self,
        config: AgentConfig,
        model: ModelProvider,
        tools: dict[str, Tool],
    ) -> None:
        self.config = config
        self.model = model
        self.tools = tools

    def _build_system_prompt(self) -> str:
        desc = "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())
        return SYSTEM_PROMPT.format(tool_desc=desc)

    def run(self, user_input: str) -> AgentResult:
        messages: list[Message] = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_input},
        ]

        for step in range(1, self.config.max_steps + 1):
            model_text = self.model.generate(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            ).strip()

            tool_call = self._try_parse_tool_call(model_text)
            if tool_call is None:
                return AgentResult(answer=model_text, steps_used=step)

            tool_name = tool_call.get("tool", "").strip()
            tool_input = tool_call.get("input", "")
            tool = self.tools.get(tool_name)
            if not tool:
                tool_result = f"工具不存在: {tool_name}"
            else:
                tool_result = tool.func(str(tool_input))

            messages.append({"role": "assistant", "content": model_text})
            messages.append(
                {
                    "role": "user",
                    "content": f"工具 `{tool_name}` 执行结果: {tool_result}",
                }
            )

        return AgentResult(
            answer=f"已达到最大步骤限制（{self.config.max_steps}），请简化问题后再试。",
            steps_used=self.config.max_steps,
        )

    @staticmethod
    def _try_parse_tool_call(text: str) -> dict[str, str] | None:
        if not text.startswith("{") or not text.endswith("}"):
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        if "tool" not in data or "input" not in data:
            return None
        return {"tool": str(data["tool"]), "input": str(data["input"])}

