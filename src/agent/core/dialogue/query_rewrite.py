from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

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


# ---------------------------------------------------------------------------
# QueryRewriteResult - 携带多形态输出，str() 兼容旧接口
# ---------------------------------------------------------------------------


@dataclass
class QueryRewriteResult:
    """
    包含多种改写形态的对象。
    str(result) 返回主 query（backward-compatible），
    调用方通过字段按需获取 hyde_doc / expanded_queries。
    """

    #: 主 query 字符串（str() 默认返回这个）
    query: str
    #: HyDE 模式下的假设文档（描述"如果存在这样一份文档，它会包含什么"）
    hyde_doc: str | None = None
    #: 多查询展开列表（expand / hyde_expand 模式）
    expanded_queries: list[str] = field(default_factory=list)
    #: 各 expanded_query 对应的 hyde_doc（若同时使用 HyDE）
    expanded_hyde_docs: list[str] = field(default_factory=list)
    #: 使用的模式标签
    mode: str = "rule"

    def __str__(self) -> str:
        return self.query


# ---------------------------------------------------------------------------
# LLM Rewrite（查询精简改写）
# ---------------------------------------------------------------------------


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
        "请将[当前用户问题]改写为更适合检索的单句query。\n"
        "约束：\n"
        "1) 不能引入原文没有的新实体/年份/数值；\n"
        "2) 若当前问题依赖上一轮用户问题，可参考[上一轮用户问题]；\n"
        "3) 输出必须是JSON，且仅包含键rewrite_query；\n"
        "4) 若无法改得更好，rewrite_query返回当前用户问题原文。\n\n"
        "上一轮用户问题："
        + (prev or "（无）")
        + "\n"
        "当前用户问题："
        + turn_text.strip()
        + "\n"
        "规则候选改写："
        + rule_query
        + "\n"
    )
    raw = model.generate(
        messages=[
            {"role": "system", "content": "你是检索query改写器，只输出JSON。"},
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


# ---------------------------------------------------------------------------
# HyDE - Hypothesis-Driven Exploration
# ---------------------------------------------------------------------------


def _hyde_generate(
    *,
    turn_text: str,
    history: list[Message] | None,
    rule_query: str,
    model: ModelProvider,
    temperature: float,
    max_tokens: int,
) -> str | None:
    """
    生成假设文档（hypothetical document）。
    引导LLM扮演一份相关文档的作者，写出一段能回答问题的[伪文档]。
    返回的hyde_doc描述[如果存在这样一份答案文档，它的内容是什么]。
    """
    history = history or []
    prev = _last_substantive_user_turn(history)

    system_prompt = (
        "你是一个检索增强生成系统的假设文档生成器。"
        "给定用户问题，你需要生成一段[假设性文档]，这段文档的内容是："
        "如果它真实存在并且包含了用户问题的答案，它会怎么写。"
        "要求："
        "1) 仅基于用户问题中的实体、年份、数值进行合理推断，不要捏造具体数字；"
        "2) 结构上像一份真实的报告段落（包含上下文、数据、解释）；"
        "3) 不要说[根据某文档]等元信息，直接写内容；"
        "4) 长度控制在100-200字；"
        "5) 输出必须是JSON，仅包含键hyde_doc。"
    )

    user_prompt = (
        "上一轮用户问题："
        + (prev or "（无）")
        + "\n"
        "当前用户问题："
        + turn_text.strip()
        + "\n"
        "规则候选改写："
        + rule_query
        + "\n"
    )

    raw = model.generate(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    ).strip()

    obj = _extract_json_object(raw)
    if not obj:
        return None
    doc = _normalize_whitespace(obj.get("hyde_doc", ""))
    return doc if doc else None


# ---------------------------------------------------------------------------
# Query Expansion - 多查询生成
# ---------------------------------------------------------------------------


def _expand_queries(
    *,
    turn_text: str,
    history: list[Message] | None,
    rule_query: str,
    model: ModelProvider,
    temperature: float,
    max_tokens: int,
    num_queries: int = 3,
) -> list[str]:
    """
    用LLM生成多个语义等价但表述不同的查询。
    用于对冲单一查询的表述偏差，提升召回率。
    """
    history = history or []
    prev = _last_substantive_user_turn(history)

    system_prompt = (
        "你是一个检索查询扩展器。根据给定的用户问题，生成多个语义等价但表述不同的检索查询。"
        "要求："
        "1) 生成3-5个不同的查询，涵盖问题的不同表述角度；"
        "2) 每个查询都是独立、完整的问句；"
        "3) 可以包含同义词替换、主动/被动转换、简短/详细表述变化；"
        "4) 不能引入原问题不包含的实体、年份或数值；"
        "5) 输出必须是JSON，仅包含键expanded_queries（字符串数组）。"
    )

    user_prompt = (
        "上一轮用户问题："
        + (prev or "（无）")
        + "\n"
        "当前用户问题："
        + turn_text.strip()
        + "\n"
        "规则候选改写："
        + rule_query
        + "\n"
    )

    raw = model.generate(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    ).strip()

    obj = _extract_json_object(raw)
    if not obj:
        return []

    raw_queries = obj.get("expanded_queries", [])
    if isinstance(raw_queries, str):
        raw_queries = [q.strip() for q in raw_queries.split("；") if q.strip()]
    if not isinstance(raw_queries, list):
        return []

    out: list[str] = []
    for q in raw_queries:
        q_str = _normalize_whitespace(str(q).strip())
        if q_str and q_str not in out:
            out.append(q_str)

    return out[:num_queries]


# ---------------------------------------------------------------------------
# HyDE + Expansion（组合模式）
# ---------------------------------------------------------------------------


def _hyde_expand(
    *,
    turn_text: str,
    history: list[Message] | None,
    rule_query: str,
    model: ModelProvider,
    temperature: float,
    max_tokens: int,
    num_queries: int = 3,
) -> tuple[list[str], list[str]]:
    """
    对每个展开的查询生成对应的假设文档。
    返回 (expanded_queries, expanded_hyde_docs)。
    """
    expanded = _expand_queries(
        turn_text=turn_text,
        history=history,
        rule_query=rule_query,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        num_queries=num_queries,
    )

    hyde_docs: list[str] = []
    for eq in expanded:
        doc = _hyde_generate(
            turn_text=eq,
            history=None,
            rule_query=rule_query,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        hyde_docs.append(doc if doc else eq)

    return expanded, hyde_docs


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def rewrite_for_rag(
    turn_text: str,
    history: list[Message] | None,
    intent: IntentResult,
    *,
    model: ModelProvider | None = None,
    mode: str = "hybrid",
    llm_temperature: float = 0.0,
    llm_max_tokens: int = 256,
) -> str | QueryRewriteResult:
    """把本轮文本整理成更适合向量/词面检索的query。

    mode:
    - rule:      仅规则改写（最快，无LLM调用）
    - llm:       仅LLM精简改写（失败回退rule）
    - hybrid:    先rule再LLM优化（失败回退rule）<- 旧默认行为
    - hyde:      HyDE模式：生成假设文档，以hyde_doc作为检索query
    - expand:    多查询展开：生成多个表述不同的query，换行拼接
    - hyde_expand: 对每个展开query生成对应hyde_doc，返回组合列表

    返回str时为backward-compatible（主query字符串）；
    返回QueryRewriteResult时携带hyde_doc/expanded_queries供高级调用方使用。
    在eval_retrieval.py等旧调用点会自动转str。
    """
    mode_norm = (mode or "hybrid").strip().lower()
    rule_query = _rule_rewrite(turn_text, history, intent)

    # rule / 无model -> 直接返回字符串
    if mode_norm == "rule" or model is None:
        return rule_query

    # ---- llm / hybrid ----
    if mode_norm in ("llm", "hybrid"):
        try:
            llm_query = _llm_rewrite(
                turn_text=turn_text,
                history=history,
                rule_query=rule_query,
                model=model,
                temperature=llm_temperature,
                max_tokens=llm_max_tokens,
            )
        except Exception:
            return rule_query
        final = llm_query if llm_query else rule_query
        return final

    # ---- hyde ----
    if mode_norm == "hyde":
        try:
            hyde_doc = _hyde_generate(
                turn_text=turn_text,
                history=history,
                rule_query=rule_query,
                model=model,
                temperature=llm_temperature,
                max_tokens=llm_max_tokens,
            )
        except Exception:
            return rule_query
        if not hyde_doc:
            return rule_query
        return QueryRewriteResult(
            query=hyde_doc,
            hyde_doc=hyde_doc,
            mode="hyde",
        )

    # ---- expand ----
    if mode_norm == "expand":
        try:
            expanded = _expand_queries(
                turn_text=turn_text,
                history=history,
                rule_query=rule_query,
                model=model,
                temperature=llm_temperature,
                max_tokens=llm_max_tokens,
                num_queries=3,
            )
        except Exception:
            return rule_query
        if not expanded:
            return rule_query
        return QueryRewriteResult(
            query="\n".join([rule_query] + expanded),
            expanded_queries=[rule_query] + expanded,
            mode="expand",
        )

    # ---- hyde_expand ----
    if mode_norm == "hyde_expand":
        try:
            expanded, hyde_docs = _hyde_expand(
                turn_text=turn_text,
                history=history,
                rule_query=rule_query,
                model=model,
                temperature=llm_temperature,
                max_tokens=llm_max_tokens,
                num_queries=3,
            )
        except Exception:
            return rule_query
        if not expanded:
            return rule_query
        return QueryRewriteResult(
            query="\n".join([rule_query] + expanded),
            expanded_queries=[rule_query] + expanded,
            expanded_hyde_docs=hyde_docs,
            mode="hyde_expand",
        )

    # 未识别的mode -> 回退hybrid行为
    return rule_query
