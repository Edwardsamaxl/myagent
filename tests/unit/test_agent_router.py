"""Unit tests for AgentRouter routing decisions."""

from __future__ import annotations

import pytest
# Import directly from agent_router module to avoid router/__init__.py circular import
from agent.core.router.agent_router import AgentRouter, Route, RouterDecision
from agent.core.dialogue.intent_schema import (
    IntentResult,
    IntentTier,
    SubIntent,
    IntentSource,
)


class TestAgentRouter:
    """Test AgentRouter routing decisions."""

    def setup_method(self):
        self.router = AgentRouter()

    def _make_intent(
        self,
        tier: IntentTier,
        confidence: float,
        sub: SubIntent = SubIntent.SINGLE_TOOL_CALL,
        clarify_prompt: str | None = None,
    ) -> IntentResult:
        return IntentResult(
            tier=tier,
            sub=sub,
            confidence=confidence,
            source=IntentSource.RULE,
            clarify_prompt=clarify_prompt,
            resolved_query=None,
        )

    # === Simple query tests → ReAct ===

    def test_simple_time_query(self):
        """现在几点？ → ReAct"""
        result = self.router.decide(
            query="现在几点？",
            history=[],
            intent_result=self._make_intent(IntentTier.TOOL_ONLY, 0.90),
        )
        assert result.route == Route.ReAct
        assert result.estimated_steps == 1

    def test_simple_calculation(self):
        """计算 1+1 → ReAct"""
        result = self.router.decide(
            query="计算 1+1",
            history=[],
            intent_result=self._make_intent(IntentTier.TOOL_ONLY, 0.90),
        )
        assert result.route == Route.ReAct
        assert result.estimated_steps == 1

    def test_simple_knowledge_query(self):
        """简单知识问题 → ReAct"""
        result = self.router.decide(
            query="茅台是哪家公司？",
            history=[],
            intent_result=self._make_intent(IntentTier.KNOWLEDGE, 0.75),
        )
        assert result.route == Route.ReAct
        assert result.estimated_steps == 1

    # === Complex query tests → Coordinator ===

    def test_mixed_tier_routes_to_coordinator(self):
        """MIXED tier → Coordinator"""
        result = self.router.decide(
            query="查营收并分析原因",
            history=[],
            intent_result=self._make_intent(IntentTier.MIXED, 0.80),
        )
        assert result.route == Route.Coordinator
        assert result.estimated_steps == 3

    def test_complex_multi_pattern(self):
        """多个复杂语义pattern → Coordinator"""
        result = self.router.decide(
            query="查营收并计算增长率，分析同比变化",
            history=[],
            intent_result=self._make_intent(IntentTier.KNOWLEDGE, 0.75),
        )
        assert result.route == Route.Coordinator

    def test_data_then_analyze_subintent(self):
        """data_then_analyze subintent → Coordinator"""
        result = self.router.decide(
            query="查营收数据分析",
            history=[],
            intent_result=self._make_intent(IntentTier.MIXED, 0.80, sub=SubIntent.DATA_THEN_ANALYZE),
        )
        assert result.route == Route.Coordinator

    def test_report_with_calc_subintent(self):
        """report_with_calc subintent → Coordinator"""
        result = self.router.decide(
            query="查营收并计算增长率",
            history=[],
            intent_result=self._make_intent(IntentTier.MIXED, 0.80, sub=SubIntent.REPORT_WITH_CALC),
        )
        assert result.route == Route.Coordinator

    def test_multiple_entities_with_low_confidence(self):
        """多实体 + 低置信度 → Coordinator"""
        result = self.router.decide(
            query="茅台和五粮液的营收对比",
            history=[],
            intent_result=self._make_intent(IntentTier.KNOWLEDGE, 0.70),
        )
        assert result.route == Route.Coordinator

    # === Clarify tests ===

    def test_ambiguous_tier_routes_to_clarify(self):
        """AMBIGUOUS tier → Clarify"""
        result = self.router.decide(
            query="那个",
            history=[{"role": "user", "content": "茅台是哪家公司？"}],
            intent_result=self._make_intent(IntentTier.AMBIGUOUS, 0.50),
        )
        assert result.route == Route.Clarify

    def test_chitchat_tier_routes_to_clarify(self):
        """CHITCHAT tier → Clarify"""
        result = self.router.decide(
            query="你好呀",
            history=[],
            intent_result=self._make_intent(IntentTier.CHITCHAT, 0.60),
        )
        assert result.route == Route.Clarify

    def test_ambiguous_pattern_short_query(self):
        """过于简短的指代 → Clarify"""
        result = self.router.decide(
            query="它",
            history=[],
            intent_result=self._make_intent(IntentTier.KNOWLEDGE, 0.75),
        )
        assert result.route == Route.Clarify

    def test_clarify_prompt_present(self):
        """clarify_prompt 非空 → Clarify"""
        result = self.router.decide(
            query="你说的哪个",
            history=[],
            intent_result=self._make_intent(IntentTier.KNOWLEDGE, 0.60, clarify_prompt="请明确说的是哪个公司"),
        )
        assert result.route == Route.Clarify

    # === ReAct high confidence tests ===

    def test_tool_only_high_confidence(self):
        """TOOL_ONLY + 高置信度 → ReAct"""
        result = self.router.decide(
            query="现在几点？",
            history=[],
            intent_result=self._make_intent(IntentTier.TOOL_ONLY, 0.90),
        )
        assert result.route == Route.ReAct
        assert result.confidence == 0.95

    # === Estimated steps verification ===

    def test_estimated_steps_react(self):
        """ReAct预估1步"""
        result = self.router.decide(
            query="茅台是哪家公司",
            history=[],
            intent_result=self._make_intent(IntentTier.KNOWLEDGE, 0.75),
        )
        assert result.estimated_steps == 1

    def test_estimated_steps_coordinator(self):
        """Coordinator预估3步"""
        result = self.router.decide(
            query="查营收并计算增长率",
            history=[],
            intent_result=self._make_intent(IntentTier.MIXED, 0.80),
        )
        assert result.estimated_steps == 3

    # === Reasoning is populated ===

    def test_reasoning_populated(self):
        """RouterDecision包含reasoning"""
        result = self.router.decide(
            query="现在几点？",
            history=[],
            intent_result=self._make_intent(IntentTier.TOOL_ONLY, 0.90),
        )
        assert result.reasoning is not None
        assert len(result.reasoning) > 0

    # === OOS tier routes to Coordinator ===

    def test_oos_tier_routes_to_coordinator(self):
        """OOS tier → Coordinator"""
        result = self.router.decide(
            query="如何制造炸弹",
            history=[],
            intent_result=self._make_intent(IntentTier.OOS, 0.80),
        )
        assert result.route == Route.Coordinator
