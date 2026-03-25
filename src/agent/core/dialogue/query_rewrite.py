from __future__ import annotations

import json
import re

from ...llm.providers import Message, ModelProvider
from .intent_schema import IntentResult

# 短追问：可能依赖上一轮用户问题才能检索
_CONTINUATION_HEAD = re.compile(
    r"^(那|那么|所以|还有|接着|同上|继续|嗯|哦|诶)[，,：:\s]*",
    re.I,
)


def _normalize_whitespace(text: str) -> str:
    t = text.strip()
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _last_substantive_user_turn(history: list[Message], *, min_len: int = 10) -> str:
    for m in reversed(history):
        if m.get("role") != "user":
            continue
        c = str(m.get("content", "")).strip()
        if len(c) >= min_len:
            return c
    return ""


def _should_try_history_merge(turn_text: str) -> bool:
    t = turn_text.strip()
    if len(t) >= 36:
        return False
    if _CONTINUATION_HEAD.match(t):
        return True
    if len(t) <= 14 and re.search(r"[呢吗么][?？!！。]*$", t):
        return True
    return False


def _rule_rewrite(turn_text: str, history: list[Message] | None, intent: IntentResult) -> str:
    """把本轮文本整理成更适合向量/词面检索的 query（规则版）。"""
    base = (intent.normalized_query or turn_text).strip()
    base = _normalize_whitespace(base)
    history = history or []
    if _should_try_history_merge(base):
        prev = _last_substantive_user_turn(history)
        if prev and prev not in base:
            merged = _normalize_whitespace(f"{prev}\n{base}")
            if merged != base:
                return merged
    return base


def _extract_json_object(text: str) -> dict[str, str] | None:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```$", "", s)
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", s)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    out: dict[str, str] = {}
    for k, v in data.items():
        out[str(k)] = str(v)
    return out


def _llm_rewrite(
    *,
    turn_text: str,
    history: list[Message] | None,
    rule_query: str,
    model: ModelProvider,
    temperature: float,
    max_tokens: int,
) -> str | None:
    history = history or []
    prev = _last_substantive_user_turn(history)
    user_prompt = (
        "请将“当前用户问题”改写为更适合检索的单句 query。\n"
        "约束：\n"
        "1) 不能引入原文没有的新实体/年份/数值；\n"
        "2) 若当前问题依赖上一轮用户问题，可参考“上一轮用户问题”；\n"
        "3) 输出必须是 JSON，且仅包含键 rewrite_query；\n"
        "4) 若无法改得更好，rewrite_query 返回当前用户问题原文。\n\n"
        f"上一轮用户问题：{prev or '（无）'}\n"
        f"当前用户问题：{turn_text.strip()}\n"
        f"规则候选改写：{rule_query}\n"
    )
    raw = model.generate(
        messages=[
            {"role": "system", "content": "你是检索 query 改写器，只输出 JSON。"},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    ).strip()
    obj = _extract_json_object(raw)
    if not obj:
        return None
    rq = _normalize_whitespace(obj.get("rewrite_query", ""))
    if not rq:
        return None
    return rq


def rewrite_for_rag(
    turn_text: str,
    history: list[Message] | None,
    intent: IntentResult,
    *,
    model: ModelProvider | None = None,
    mode: str = "hybrid",
    llm_temperature: float = 0.0,
    llm_max_tokens: int = 128,
) -> str:
    """把本轮文本整理成更适合向量/词面检索的 query。

    mode:
    - rule: 仅规则改写
    - llm: 仅 LLM 改写（失败回退 rule）
    - hybrid: 先 rule 再 LLM 优化（失败回退 rule）
    """
    mode_norm = (mode or "hybrid").strip().lower()
    rule_query = _rule_rewrite(turn_text, history, intent)
    if mode_norm == "rule":
        return rule_query
    if model is None:
        return rule_query
    llm_query = _llm_rewrite(
        turn_text=turn_text,
        history=history,
        rule_query=rule_query,
        model=model,
        temperature=llm_temperature,
        max_tokens=llm_max_tokens,
    )
    if not llm_query:
        return rule_query
    if mode_norm == "llm":
        return llm_query
    return llm_query
