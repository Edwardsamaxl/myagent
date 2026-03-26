from __future__ import annotations

"""多步工具循环（SimpleAgent）。

职责边界（与 RAG 解耦）：
- 本模块只处理「工具 JSON ↔ 执行 ↔ 结果写回消息」；不负责检索、重排或证据拼接。
- 当上层在 user 消息中注入「[检索证据]」块时，你应把它当作只读上下文回答用户，
  不要用工具去「替代检索」或重复拉取同一批证据；工具用于时间/计算/记忆/技能/workspace 文件等。
- 纯 RAG 直答请走 RagAgentService.answer / HTTP 专用接口；本循环用于需要工具能力的对话。

若检索返回结构或证据块格式变化，由 application 层与 evidence_format 对齐后再改注入模板，勿在本文件硬编码字段名。
"""

import json
import re
from dataclasses import dataclass

from ..config import AgentConfig
from ..llm.providers import Message, ModelProvider
from ..tools.registry import Tool


# 工具描述与记忆/技能块；RAG 证据不在此定义，由 AgentService.chat 拼进 user 消息。
SYSTEM_PROMPT = """你是一个可调用工具的助手。
你有以下工具可用（name: description）：
{tool_desc}

你有以下长期上下文（记忆与技能）：
{long_context}

【工具调用协议】
当你需要调用工具时，必须输出且只能输出一行纯 JSON，格式如下：
{"tool":"工具名","input":"参数"}

禁止在 JSON 前后添加任何解释、 Markdown、代码块或标点符号。

【示例】
用户：现在几点？ → 输出：{"tool":"get_time","input":""}
用户：读取记忆 → 输出：{"tool":"read_memory","input":""}
用户：计算 2+3*4 → 输出：{"tool":"calculate","input":"2+3*4"}
用户：搜索知识库 关于公司财务数据 → 输出：{"tool":"search_knowledge_base","input":"公司 财务数据"}
用户：今天天气如何？ → 输出：{"tool":"web_search","input":"今天天气"}
"""


def _extract_first_json_object(text: str) -> str | None:
    """从文本中提取第一个完整的 `{...}` JSON 对象片段。"""
    s = (text or "").strip()
    if not s:
        return None

    depth = 0
    start_idx: int | None = None
    for idx, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start_idx = idx
            depth += 1
            continue
        if ch != "}":
            continue
        if depth <= 0 or start_idx is None:
            continue
        depth -= 1
        if depth == 0:
            return s[start_idx : idx + 1]
    return None


def _try_parse_tool_call_json(candidate: str) -> dict[str, str] | None:
    """解析并校验工具调用 JSON：{"tool": ..., "input": ...}。"""
    if not candidate:
        return None
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if "tool" not in data or "input" not in data:
        return None
    return {"tool": str(data["tool"]), "input": str(data["input"])}


_TOOL_CALL_FALLBACK_RE = re.compile(
    r"(?:^\s*\[工具\]\s*)?(?P<tool>[a-zA-Z_]\w*)\s*\(\s*(?P<input>[^)]*)\s*\)\s*$"
)


def _try_parse_tool_call_fallback(text: str) -> dict[str, str] | None:
    """兜底解析：兼容模型输出 `[工具] get_time()` 这类非 JSON 格式。

    仅在整条输出"看起来像一次工具调用"时触发，避免误判普通文本。
    """
    s = (text or "").strip()
    if not s:
        return None
    m = _TOOL_CALL_FALLBACK_RE.match(s)
    if not m:
        return None
    tool = str(m.group("tool") or "").strip()
    raw_inp = str(m.group("input") or "").strip()
    # 去掉常见的引号包裹
    if (raw_inp.startswith('"') and raw_inp.endswith('"')) or (
        raw_inp.startswith("'") and raw_inp.endswith("'")
    ):
        raw_inp = raw_inp[1:-1]
    return {"tool": tool, "input": raw_inp}


@dataclass
class AgentResult:
    answer: str
    steps_used: int
    messages: list[Message]
    tool_calls: list[str]


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

    def _build_system_prompt(self, long_context: str) -> str:
        desc = "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())
        context = long_context.strip() or "暂无。"
        # 避免 `str.format` 与提示词中的 `{...}` 示例冲突导致 KeyError。
        return (
            SYSTEM_PROMPT.replace("{tool_desc}", desc).replace("{long_context}", context)
        )

    def run(
        self,
        user_input: str,
        history_messages: list[Message] | None = None,
        long_context: str = "",
    ) -> AgentResult:
        history_messages = history_messages or []
        messages: list[Message] = [
            {"role": "system", "content": self._build_system_prompt(long_context)},
            *history_messages,
        ]
        messages.append({"role": "user", "content": user_input})
        tool_calls: list[str] = []

        for step in range(1, self.config.max_steps + 1):
            model_text = self.model.generate(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            ).strip()

            tool_call = self._try_parse_tool_call(model_text)
            if tool_call is None:
                return AgentResult(
                    answer=model_text,
                    steps_used=step,
                    messages=messages + [{"role": "assistant", "content": model_text}],
                    tool_calls=tool_calls,
                )

            tool_name = tool_call.get("tool", "").strip()
            tool_input = tool_call.get("input", "")
            tool = self.tools.get(tool_name)
            if not tool:
                tool_result = f"工具不存在: {tool_name}"
            else:
                try:
                    tool_result = tool.func(str(tool_input))
                except Exception as exc:  # noqa: BLE001
                    tool_result = f"工具执行异常: {exc}"
            tool_calls.append(f"{tool_name}({tool_input})")

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
            messages=messages,
            tool_calls=tool_calls,
        )

    @staticmethod
    def _try_parse_tool_call(text: str) -> dict[str, str] | None:
        """尽量鲁棒地解析工具调用 JSON。

        允许模型输出包含前后解释文本，例如：
        "好的，接下来我会……\n{\"tool\":\"read_memory\",\"input\":\"\"}"
        只要能在文本中找到一个形如 {"tool":..., "input":...} 的 JSON 对象即可。
        """
        s = (text or "").strip()
        if not s:
            return None

        # 快路径：整条就是 JSON
        if s.startswith("{") and s.endswith("}"):
            parsed = _try_parse_tool_call_json(s)
            if parsed:
                return parsed

        # 慢路径：在文本中寻找第一个完整 `{...}` JSON 对象
        candidate = _extract_first_json_object(s)
        if not candidate:
            return _try_parse_tool_call_fallback(s)
        parsed = _try_parse_tool_call_json(candidate)
        if parsed:
            return parsed
        return _try_parse_tool_call_fallback(s)
