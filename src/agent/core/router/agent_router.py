"""
AgentRouter: 根据复杂度 + 困惑度判断路由到 ReAct / Coordinator / Clarify

判断维度:
1. 复杂度: 步骤数量（1步=简单，N步=复杂）
2. 困惑度: 模型对当前上下文的置信度
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..dialogue.intent_schema import IntentResult


class Route(str, Enum):
    """路由目标"""
    ReAct = "react"         # 单步工具调用（简单问题）
    Coordinator = "coordinator"  # 多步规划（复杂问题）
    Clarify = "clarify"    # 需要澄清（模糊问题）


# ============ 复杂度判断（基于 IntentTier + 查询特征）============

# 触发 Coordinator 的 IntentTier
_COMPLEX_TIERS = {"mixed", "oos"}

# 触发 Clarify 的 IntentTier
模糊_TIERS = {"ambiguous", "chitchat"}

# 复杂查询的语义信号（包含这些词通常是复合问题）
_COMPLEX_PATTERNS = [
    re.compile(r"并[且和]"),
    re.compile(r"对比|比较"),
    re.compile(r"增长率|同比|环比"),
    re.compile(r"计算|分析|评估"),
    re.compile(r"原因|为什么|如何"),
    re.compile(r".*和.*"),  # A和B
    re.compile(r"首先|然后|最后"),
    re.compile(r"\?.*\?"),  # 多个问号
]

# 触发 Clarify 的模糊信号
_AMBIGUOUS_PATTERNS = [
    re.compile(r"^(那个|这个|它|这|那)\s*$"),  # 单独指代词
    re.compile(r"^(怎么了|为什么|如何)\s*$"),   # 单独疑问词无宾语
    re.compile(r"^继续$"),
    re.compile(r"^嗯$|^啊$"),
]


@dataclass
class RouterDecision:
    """路由决策结果"""
    route: Route                    # 路由目标
    confidence: float              # 决策置信度
    reasoning: str                 # 决策理由
    estimated_steps: int           # 预估步骤数


class AgentRouter:
    """
    Agent 路由决策器。

    决策逻辑：
    1. 命中 Clarify 规则 → Clarify
    2. IntentTier 为 MIXED/OOS → Coordinator（复杂多步）
    3. IntentTier 为 AMBIGUOUS/CHITCHAT → Clarify
    4. IntentTier 为 TOOL_ONLY + 置信度 >= 0.85 → ReAct
    5. 包含复杂语义 pattern → Coordinator（预估多步）
    6. 默认 → ReAct（单步简单任务）
    """

    def decide(
        self,
        query: str,
        history: list,
        intent_result: IntentResult,
    ) -> RouterDecision:
        """
        根据查询、历史和意图分类结果决定路由。

        Args:
            query: 用户原始查询
            history: 对话历史
            intent_result: IntentClassifier 的输出结果

        Returns:
            RouterDecision: 包含路由目标、置信度和理由
        """
        tier = intent_result.tier.value
        confidence = intent_result.confidence
        sub = intent_result.sub.value if intent_result.sub else ""

        # ---- 1. 明确需要澄清的情况 ----
        if tier in 模糊_TIERS or intent_result.clarify_prompt:
            return RouterDecision(
                route=Route.Clarify,
                confidence=0.95,
                reasoning=f"IntentTier={tier}，需要澄清",
                estimated_steps=0,
            )

        # 模糊 pattern 检测
        for pat in _AMBIGUOUS_PATTERNS:
            if pat.match(query.strip()):
                return RouterDecision(
                    route=Route.Clarify,
                    confidence=0.90,
                    reasoning="查询过于简短或指代不明确",
                    estimated_steps=0,
                )

        # ---- 2. 明确需要 Coordinator 的情况 ----
        if tier in _COMPLEX_TIERS:
            return RouterDecision(
                route=Route.Coordinator,
                confidence=0.90,
                reasoning=f"IntentTier={tier}，多步复杂任务",
                estimated_steps=3,
            )

        # MIXED 相关子意图 → Coordinator
        if sub in ("data_then_analyze", "report_with_calc"):
            return RouterDecision(
                route=Route.Coordinator,
                confidence=0.90,
                reasoning=f"子意图={sub}，需要检索+分析多步",
                estimated_steps=3,
            )

        # ---- 3. 复杂语义 pattern 检测 ----
        complex_matches = sum(1 for pat in _COMPLEX_PATTERNS if pat.search(query))
        if complex_matches >= 2:
            return RouterDecision(
                route=Route.Coordinator,
                confidence=0.85,
                reasoning=f"查询包含{complex_matches}个复杂语义pattern，多步任务",
                estimated_steps=3,
            )

        # 包含"和"字连接多个实体/事件 → 很可能需要多步
        if re.search(r".+和.+和.+", query) or (
            re.search(r".+和.+", query) and confidence < 0.80
        ):
            return RouterDecision(
                route=Route.Coordinator,
                confidence=0.80,
                reasoning="多实体并列查询，预估多步",
                estimated_steps=2,
            )

        # ---- 4. TOOL_ONLY 高置信度 → ReAct ----
        if tier == "tool_only" and confidence >= 0.85:
            return RouterDecision(
                route=Route.ReAct,
                confidence=0.95,
                reasoning=f"TOOL_ONLY高置信度({confidence:.2f})，单步工具调用",
                estimated_steps=1,
            )

        # ---- 5. 单步工具调用特征明显 → ReAct ----
        _SINGLE_STEP_SIGNALS = [
            r"现在几点",
            r"几点了",
            r"今天日期",
            r"计算\d",
            r"^\d+[+\-*/]",
            r"read_workspace",
            r"搜索.*",
            r"查一下.*",
        ]
        for sig in _SINGLE_STEP_SIGNALS:
            if re.search(sig, query):
                return RouterDecision(
                    route=Route.ReAct,
                    confidence=0.95,
                    reasoning=f"匹配单步信号: {sig}",
                    estimated_steps=1,
                )

        # ---- 6. 低置信度 + 非 TOOL_ONLY → Clarify ----
        if confidence < 0.60 and tier != "knowledge":
            return RouterDecision(
                route=Route.Clarify,
                confidence=0.85,
                reasoning=f"置信度低({confidence:.2f})且意图不明确",
                estimated_steps=0,
            )

        # ---- 7. 兜底：默认 ReAct（简单 knowledge 查询）----
        return RouterDecision(
            route=Route.ReAct,
            confidence=0.75,
            reasoning="简单知识查询，直接单步回答",
            estimated_steps=1,
        )
