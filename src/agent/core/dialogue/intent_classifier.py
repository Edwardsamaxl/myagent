from __future__ import annotations

import re

from ...llm.providers import Message
from .clarify_policy import default_clarify_prompt, should_clarify_for_finance_without_anchor
from .intent_schema import IntentKind, IntentResult

_TOOL_PATTERNS = (
    r"现在几点|几点了|当前时间|时间",
    r"计算|算一下|等于多少|^\s*\d",
    r"read_workspace|读.*文件|打开.*文件|workspace",
    r"记住|保存技能|列技能|MEMORY|技能",
)
_CHITCHAT_PATTERNS = (
    r"^(你好|您好|谢谢|多谢|再见|拜拜|早上好|晚上好)[\s!！。.]*$",
    r"^你是谁",
    r"^你能做什么",
)


def _history_has_substantive_user(history: list[Message], min_chars: int = 12) -> bool:
    for m in reversed(history[-10:]):
        if m.get("role") != "user":
            continue
        c = str(m.get("content", "")).strip()
        if len(c) >= min_chars:
            return True
    return False


def classify_intent(user_message: str, history: list[Message] | None = None) -> IntentResult:
    """规则版意图分类（零额外模型调用）。"""
    history = history or []
    text = user_message.strip()

    if not text:
        return IntentResult(
            intent=IntentKind.AMBIGUOUS,
            confidence=0.2,
            clarify_prompt=default_clarify_prompt(text),
        )

    low = text.lower()
    for pat in _TOOL_PATTERNS:
        if re.search(pat, text, re.I) or re.search(pat, low, re.I):
            return IntentResult(
                intent=IntentKind.TOOL_ONLY,
                confidence=0.85,
                normalized_query=text,
            )

    for pat in _CHITCHAT_PATTERNS:
        if re.match(pat, text.strip(), re.I):
            return IntentResult(
                intent=IntentKind.CHITCHAT,
                confidence=0.8,
                normalized_query=text,
            )

    # 指代过短且无历史支撑 → 澄清
    if len(text) < 6 and not _history_has_substantive_user(history):
        vague = re.match(r"^(这个|那个|同上|继续|呢|嗯|啊)[\s?？!！。]*$", text)
        if vague or len(text) < 4:
            return IntentResult(
                intent=IntentKind.AMBIGUOUS,
                confidence=0.35,
                clarify_prompt=default_clarify_prompt(text),
            )

    # 默认：语料知识型（财报/事实）
    ir = IntentResult(
        intent=IntentKind.KNOWLEDGE_CORPUS,
        confidence=0.72,
        normalized_query=text,
    )
    if should_clarify_for_finance_without_anchor(text):
        return IntentResult(
            intent=IntentKind.AMBIGUOUS,
            confidence=0.45,
            normalized_query=None,
            clarify_prompt=default_clarify_prompt(text),
        )
    return ir
