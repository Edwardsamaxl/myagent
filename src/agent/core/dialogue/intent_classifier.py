from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from .clarify_policy import default_clarify_prompt, should_clarify_for_finance_without_anchor
from .intent_schema import (
    IntentTier, SubIntent, IntentSource,
    IntentContext, IntentResult,
    INTENT_DESCRIPTIONS,
)

if TYPE_CHECKING:
    from ...llm.providers import Message


# ============================================================
# Stage 1: 规则快速过滤（零开销）
# ============================================================
_TIER_RULES: dict[IntentTier, list[tuple[str, float]]] = {
    IntentTier.TOOL_ONLY: [
        (r"现在几点|几点了|当前时间|今天日期|现在.*点", 0.90),
        (r"计算|算一下|等于多少|^\s*\d", 0.88),
        (r"read_workspace|读.*文件|打开.*文件|workspace", 0.85),
        (r"记住|保存技能|列技能|MEMORY", 0.85),
        (r"技能", 0.80),
        (r"搜索|上网|查一下|look up", 0.80),
    ],
    IntentTier.CHITCHAT: [
        (r"^(你好|您好|谢谢|多谢|再见|拜拜|早上好|晚上好)[\s!！。.]*$", 0.90),
        (r"^(你好|您好|谢谢|多谢|再见|拜拜|早上好|晚上好)[呀啊哦嘛呢啦]*[\s!！。.]*$", 0.88),
        (r"^(你好|您好|谢谢|多谢|再见|拜拜|早上好|晚上好)$", 0.90),
        (r"^你是谁", 0.90),
        (r"^你能做什么", 0.85),
        (r"^帮?个?忙", 0.60),
    ],
}


def _apply_rule_classification(text: str) -> IntentResult | None:
    """Stage 1: 规则快速过滤。命中有高置信度直接返回；未命中返回 None 继续后续阶段。"""
    low = text.lower().strip()

    for tier, patterns in _TIER_RULES.items():
        for pat, conf in patterns:
            if re.search(pat, text, re.I) or re.search(pat, low, re.I):
                sub = _tier_to_default_sub(tier)
                return IntentResult(
                    tier=tier,
                    sub=sub,
                    confidence=conf,
                    source=IntentSource.RULE,
                    resolved_query=text,
                )
    return None


def _tier_to_default_sub(tier: IntentTier) -> SubIntent:
    """顶层意图到默认子意图的映射。"""
    mapping = {
        IntentTier.TOOL_ONLY: SubIntent.SINGLE_TOOL_CALL,
        IntentTier.CHITCHAT: SubIntent.GREETING,
        IntentTier.KNOWLEDGE: SubIntent.GENERAL_FACT,
        IntentTier.MIXED: SubIntent.DATA_THEN_ANALYZE,
        IntentTier.OOS: SubIntent.OFF_TOPIC,
        IntentTier.AMBIGUOUS: SubIntent.CASUAL,
    }
    return mapping.get(tier, SubIntent.CASUAL)


# ============================================================
# Stage 2: Embedding 相似度检索（REIC 启发）
# ============================================================
_EMBEDDING_MODEL: Any = None  # Lazy load: sentence-transformers


async def _get_embedding_model():
    """懒加载 sentence-transformers 模型。"""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL = SentenceTransformer("all-mpnet-base-v2")
    return _EMBEDDING_MODEL


async def _apply_embedding_classification(
    text: str,
    threshold: float = 0.80,
    top_k: int = 3,
) -> IntentResult | None:
    """Stage 2: Embedding 相似度分类。返回 top-1 超过阈值的结果，否则返回 None。"""
    try:
        model = await _get_embedding_model()
        sub_intents = list(INTENT_DESCRIPTIONS.keys())
        descriptions = [INTENT_DESCRIPTIONS[s] for s in sub_intents]

        # 批量编码（单次模型调用）
        query_vec = model.encode([text])
        desc_vecs = model.encode(descriptions)

        # 余弦相似度
        import numpy as np
        similarities = np.dot(query_vec, desc_vecs.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        top_sub = sub_intents[top_indices[0]]
        top_score = float(similarities[top_indices[0]])

        if top_score >= threshold:
            tier = _sub_to_tier(top_sub)
            return IntentResult(
                tier=tier,
                sub=top_sub,
                confidence=top_score,
                source=IntentSource.EMBEDDING,
                resolved_query=text,
            )
    except Exception:
        pass
    return None


def _sub_to_tier(sub: SubIntent) -> IntentTier:
    """子意图到顶层意图的反向映射。"""
    if sub in (SubIntent.TIME_QUERY, SubIntent.CALCULATION, SubIntent.FILE_READ,
              SubIntent.MEMORY_OP, SubIntent.SKILL_INVOKE, SubIntent.SINGLE_TOOL_CALL,
              SubIntent.MULTI_STEP_TOOL):
        return IntentTier.TOOL_ONLY
    if sub in (SubIntent.FINANCIAL_QUERY, SubIntent.COMPANY_QUERY,
              SubIntent.MARKET_DATA, SubIntent.GENERAL_FACT):
        return IntentTier.KNOWLEDGE
    if sub in (SubIntent.DATA_THEN_ANALYZE, SubIntent.REPORT_WITH_CALC):
        return IntentTier.MIXED
    if sub in (SubIntent.GREETING, SubIntent.SELF_INTRO, SubIntent.CASUAL):
        return IntentTier.CHITCHAT
    return IntentTier.OOS


# ============================================================
# Stage 3: LLM 分类（Zero-shot，REIC/FCSLM 启发）
# ============================================================
_LLM_CLASSIFIER_PROMPT = """你是一个意图分类专家。根据用户输入，从以下类别中选择最合适的意图。

## 顶层意图（选择其中一个）
- tool_only: 纯工具调用，用户想执行单一操作（如查时间、计算、读文件、搜索）
- knowledge: 知识问答，用户想从文档/语料库中查询信息（如财务数据、公司信息）
- mixed: 混合意图，需要先检索数据再执行分析/计算
- chitchat: 闲聊/寒暄（如问候、自报家门）
- ambiguous: 意图不清晰，需要澄清
- oos: 超出系统能力范围的请求

## 子意图（选择与顶层对应的子类别）
tool_only: time_query | calculation | file_read | memory_op | skill_invoke | single_tool_call | multi_step_tool
knowledge: financial_query | company_query | market_data | general_fact
mixed: data_then_analyze | report_with_calc
chitchat: greeting | self_intro | casual
ambiguous: (无需子意图)
oos: off_topic | sensitive | unsafe

## 输出格式（仅输出 JSON，不要有其他内容）
{
  "tier": "意图类型",
  "sub": "子意图",
  "confidence": 0.0-1.0之间的置信度,
  "reasoning": "简短分类理由（用于调试）"
}

## 注意事项
- 如果问题涉及公司财务数据（营收、利润、ROE等）且缺少具体公司名或年份，confidence 应降低
- 如果问题过于简短（<6字）且无历史上下文，倾向于 ambiguous
- 保持 JSON 格式，不要有 markdown 代码块包裹
"""


async def _apply_llm_classification(
    text: str,
    context: IntentContext,
    model: Any,
) -> IntentResult | None:
    """Stage 3: LLM Zero-shot 分类。低于阈值(0.5)则归类为 OOS。"""
    messages: list[dict] = [
        {"role": "system", "content": _LLM_CLASSIFIER_PROMPT},
    ]
    # 注入上下文（历史摘要）
    if context.session_history:
        history_summary = _summarize_history(context.session_history)
        messages.append({"role": "system", "content": f"对话历史摘要：\n{history_summary}"})

    messages.append({"role": "user", "content": text})

    try:
        response = model.generate(messages=messages, temperature=0.1, max_tokens=256)
        raw = response.strip()
        # 去掉可能的 markdown 包装
        if raw.startswith("```"):
            parts = raw.split("```", 2)
            if len(parts) >= 3:
                raw = parts[1].strip()
                if raw.startswith("json"):
                    raw = raw.split("\n", 1)[1].strip()
        import json
        data = json.loads(raw)

        tier = IntentTier(data["tier"])
        sub = SubIntent(data["sub"])
        confidence = float(data["confidence"])

        # 低于阈值 → OOS
        is_oos = confidence < 0.5
        if is_oos:
            tier = IntentTier.OOS
            sub = SubIntent.OFF_TOPIC

        return IntentResult(
            tier=tier,
            sub=sub,
            confidence=confidence,
            source=IntentSource.LLM,
            resolved_query=text,
            is_oos=is_oos,
        )
    except Exception:
        return None


def _summarize_history(history: list[Message], max_turns: int = 5) -> str:
    """从历史中提取关键实体和话题（用于 LLM 上下文注入）。"""
    lines = []
    for m in history[-max_turns:]:
        if m.get("role") == "user":
            content = str(m.get("content", ""))[:100]
            lines.append(f"用户: {content}")
    return "\n".join(lines[-max_turns:])


# ============================================================
# Stage 4: 子意图精细化 + 上下文感知
# ============================================================
def _refine_sub_intent(
    text: str,
    context: IntentContext,
    result: IntentResult,
) -> IntentResult:
    """Stage 4: 基于上下文和规则对子意图进行精细化调整。"""
    # 财务类问题无锚点 → 降置信度，触发澄清
    if result.tier == IntentTier.KNOWLEDGE:
        if should_clarify_for_finance_without_anchor(text):
            result.confidence = min(result.confidence, 0.55)
            result.clarify_prompt = default_clarify_prompt(text)

    # 过短 query 且无历史支撑 → AMBIGUOUS
    if len(text) < 6 and not _history_has_substantive_user(context.session_history):
        vague = re.match(r"^(这个|那个|同上|继续|呢|嗯|啊)[\s?？!！。]*$", text)
        if vague or len(text) < 4:
            return IntentResult(
                tier=IntentTier.AMBIGUOUS,
                sub=SubIntent.CASUAL,
                confidence=0.35,
                source=IntentSource.RULE,
                clarify_prompt=default_clarify_prompt(text),
            )

    # 指代消解：引用历史中的实体
    if context.last_intent and context.session_history:
        resolved, slots = _resolve_coreference(text, context)
        if resolved != text:
            result.resolved_query = resolved
            result.resolved_slots.update(slots)

    return result


# ============================================================
# 主分类函数（Async，支持完整流水线）
# ============================================================
async def classify_intent_async(
    user_message: str,
    history: list[Message] | None = None,
    model: Any | None = None,       # LLM provider for Stage 3
    context: IntentContext | None = None,
) -> IntentResult:
    """
    层次化多策略意图分类主函数。

    流水线：Stage 1(规则) → Stage 2(Embedding) → Stage 3(LLM) → Stage 4(精细化)
    """
    history = history or []
    text = user_message.strip()
    context = context or IntentContext(session_history=history)
    context.session_history = history

    if not text:
        return IntentResult(
            tier=IntentTier.AMBIGUOUS,
            sub=SubIntent.CASUAL,
            confidence=0.2,
            source=IntentSource.RULE,
            clarify_prompt=default_clarify_prompt(text),
        )

    # Stage 1: 规则（同步，无开销）
    rule_result = _apply_rule_classification(text)
    if rule_result and rule_result.confidence >= 0.85:
        return _refine_sub_intent(text, context, rule_result)

    # Stage 2: Embedding（异步，REIC 启发）
    embed_result = await _apply_embedding_classification(text)
    if embed_result and embed_result.confidence >= 0.80:
        return _refine_sub_intent(text, context, embed_result)

    # Stage 3: LLM（异步，Zero-shot）
    if model is not None:
        llm_result = await _apply_llm_classification(text, context, model)
        if llm_result:
            return _refine_sub_intent(text, context, llm_result)

    # Fallback: 默认 KNOWLEDGE（保留当前行为）
    fallback = IntentResult(
        tier=IntentTier.KNOWLEDGE,
        sub=SubIntent.GENERAL_FACT,
        confidence=0.72,
        source=IntentSource.FALLBACK,
        resolved_query=text,
    )
    return _refine_sub_intent(text, context, fallback)


# ============================================================
# 同步包装器（兼容现有调用）
# ============================================================
def classify_intent(
    user_message: str,
    history: list[Message] | None = None,
) -> IntentResult:
    """同步包装器，Stage 1 + Stage 4 简化版（无 Embedding/LLM 调用）。"""
    history = history or []
    text = user_message.strip()

    if not text:
        return IntentResult(
            tier=IntentTier.AMBIGUOUS,
            sub=SubIntent.CASUAL,
            confidence=0.2,
            source=IntentSource.RULE,
            clarify_prompt=default_clarify_prompt(text),
        )

    # Stage 1
    rule_result = _apply_rule_classification(text)
    if rule_result:
        result = rule_result
    else:
        # 未命中规则，默认
        result = IntentResult(
            tier=IntentTier.KNOWLEDGE,
            sub=SubIntent.GENERAL_FACT,
            confidence=0.72,
            source=IntentSource.FALLBACK,
            resolved_query=text,
        )

    # Stage 4: 精细化
    ctx = IntentContext(session_history=history)
    return _refine_sub_intent(text, ctx, result)


# ============================================================
# 辅助函数
# ============================================================
def _history_has_substantive_user(history: list[Message], min_chars: int = 12) -> bool:
    """检查历史中是否有实质性的用户消息（用于指代消解判断）。"""
    for m in reversed(history[-10:]):
        if m.get("role") != "user":
            continue
        c = str(m.get("content", "")).strip()
        if len(c) >= min_chars:
            return True
    return False


def _resolve_coreference(text: str, context: IntentContext) -> tuple[str, dict[str, str]]:
    """
    简单指代消解：检测代词/简写引用历史实体。
    返回 (消解后文本, 提取的槽位).
    """
    slots: dict[str, str] = {}
    resolved = text

    # 简单规则（后续可接入更复杂的消解模型）
    coref_patterns = [
        (r"^那个公司", _extract_last_company(context.session_history)),
        (r"^该公司", _extract_last_company(context.session_history)),
        (r"^上年|^去年", _extract_last_year(context.session_history)),
        (r"^那笔|^那笔交易", _extract_last_transaction(context.session_history)),
    ]

    for pat, value in coref_patterns:
        if value and re.match(pat, resolved):
            resolved = re.sub(pat, value, resolved)
            break

    return resolved, slots


def _extract_last_company(history: list[Message]) -> str | None:
    for m in reversed(history):
        if m.get("role") == "user":
            c = str(m.get("content", ""))
            match = re.search(r"(贵州茅台|五粮液|同花顺|\w+公司|\w+股份)", c)
            if match:
                return match.group(1)
    return None


def _extract_last_year(history: list[Message]) -> str | None:
    for m in reversed(history):
        if m.get("role") == "user":
            c = str(m.get("content", ""))
            match = re.search(r"(20\d{2})", c)
            if match:
                return match.group(1) + "年"
    return None


def _extract_last_transaction(history: list[Message]) -> str | None:
    return None  # TODO: 实现交易相关实体抽取
