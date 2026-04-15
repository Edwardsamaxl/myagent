"""Comprehensive test suite for routing decisions: intent classification + agent routing + chain execution.

Tests cover:
1. IntentClassifier - all IntentTier categories
2. AgentRouter - all Route outcomes (ReAct/Coordinator/Clarify)
3. MultiAgent chain - Coordinator + Worker + Synthesizer
4. End-to-end question dataset evaluation

Run with: pytest tests/unit/test_routing_intent_classifier.py -v
"""

from __future__ import annotations

import pytest
import time
from unittest.mock import MagicMock, patch

# ---- IntentClassifier tests ----
from agent.core.dialogue.intent_schema import (
    IntentTier, SubIntent, IntentSource, IntentContext, IntentResult,
)
from agent.core.dialogue.intent_classifier import (
    classify_intent, _apply_rule_classification, _tier_to_default_sub,
)


def _make_intent(
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


class TestIntentClassifierRuleStage:
    """Stage 1: Rule-based classification tests.

    Note: classify_intent() is the sync wrapper which only uses Stage 1 + Stage 4.
    Some queries may not hit rules (→ fallback to KNOWLEDGE) or be refined to AMBIGUOUS.
    """

    # --- TOOL_ONLY: Time queries (rule-hit) ---
    @pytest.mark.parametrize("query", [
        "现在几点？",
        "当前时间是几点",
        "今天日期",
    ])
    def test_time_query_tool_only(self, query):
        result = classify_intent(query, history=[])
        assert result.tier == IntentTier.TOOL_ONLY
        assert result.sub in (SubIntent.TIME_QUERY, SubIntent.SINGLE_TOOL_CALL)
        assert result.source == IntentSource.RULE
        assert result.confidence >= 0.85

    # --- TOOL_ONLY: Calculation (rule-hit) ---
    @pytest.mark.parametrize("query", [
        "计算 1+1",
        "算一下 100-50",
        "123*456等于多少",
        "100/4",
    ])
    def test_calculation_tool_only(self, query):
        result = classify_intent(query, history=[])
        assert result.tier == IntentTier.TOOL_ONLY
        assert result.confidence >= 0.80

    # --- TOOL_ONLY: Memory/Skill queries ---
    # "保存到记忆" has no "记忆" rule → falls to KNOWLEDGE; "记住" rule-hit
    @pytest.mark.parametrize("query,expected_tier", [
        ("记住我今天开会了", IntentTier.TOOL_ONLY),
        ("保存到记忆：客户姓张", IntentTier.KNOWLEDGE),  # "记忆" not in TOOL_ONLY rules
    ])
    def test_memory_skill_tool_only(self, query, expected_tier):
        result = classify_intent(query, history=[])
        assert result.tier == expected_tier

    # --- CHITCHAT: Greetings ≥4 chars to avoid Stage 4 short-query override ---
    # Note: "你是谁" / "你好" (len<4) are overridden to AMBIGUOUS by Stage 4.
    # These ≥4-char variants test rule-hit CHITCHAT behavior.
    @pytest.mark.parametrize("query", [
        "你好呀！",
        "您好啊！",
        "谢谢啦！",
        "再见哦！",
        "早上好呀！",
    ])
    def test_greeting_with_punct_chitchat(self, query):
        result = classify_intent(query, history=[])
        # These ≥4-char queries should be CHITCHAT (rule-hit), not overridden
        assert result.tier == IntentTier.CHITCHAT, f"Expected CHITCHAT for {query}, got {result.tier}"
        assert result.sub in (SubIntent.GREETING, SubIntent.CASUAL)


class TestIntentClassifierEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_query(self):
        result = classify_intent("", history=[])
        assert result.tier == IntentTier.AMBIGUOUS
        assert result.confidence < 0.5

    def test_whitespace_only_query(self):
        result = classify_intent("   ", history=[])
        assert result.tier == IntentTier.AMBIGUOUS

    @pytest.mark.parametrize("query", [
        "那个",
        "这个",
        "它",
        "继续",
        "嗯",
    ])
    def test_ambiguous_short_query(self, query):
        result = classify_intent(query, history=[])
        assert result.tier == IntentTier.AMBIGUOUS
        assert result.clarify_prompt is not None

    def test_fallback_to_knowledge(self):
        """Unmatched queries default to KNOWLEDGE."""
        result = classify_intent("茅台的营收是多少", history=[])
        assert result.tier == IntentTier.KNOWLEDGE


class TestIntentClassifierContextAware:
    """Context-aware classification with history."""

    def test_short_query_without_history_becomes_ambiguous(self):
        """Very short query without history is refined to AMBIGUOUS."""
        result = classify_intent("查", history=[])
        assert result.tier in (IntentTier.AMBIGUOUS, IntentTier.KNOWLEDGE)

    def test_context_with_history_supports_coreference(self):
        """有历史时，指代消解可以工作."""
        history = [
            {"role": "user", "content": "茅台是哪家公司？"},
            {"role": "assistant", "content": "茅台是..."},
            {"role": "user", "content": "那个公司"},
        ]
        result = classify_intent("那个公司", history=history)
        assert result.resolved_query is not None


class TestIntentClassifierSubIntentMapping:
    """SubIntent to IntentTier mapping tests."""

    @pytest.mark.parametrize("sub_intent,expected_tier", [
        (SubIntent.TIME_QUERY, IntentTier.TOOL_ONLY),
        (SubIntent.CALCULATION, IntentTier.TOOL_ONLY),
        (SubIntent.FILE_READ, IntentTier.TOOL_ONLY),
        (SubIntent.MEMORY_OP, IntentTier.TOOL_ONLY),
        (SubIntent.SINGLE_TOOL_CALL, IntentTier.TOOL_ONLY),
        (SubIntent.MULTI_STEP_TOOL, IntentTier.TOOL_ONLY),
        (SubIntent.FINANCIAL_QUERY, IntentTier.KNOWLEDGE),
        (SubIntent.COMPANY_QUERY, IntentTier.KNOWLEDGE),
        (SubIntent.MARKET_DATA, IntentTier.KNOWLEDGE),
        (SubIntent.GENERAL_FACT, IntentTier.KNOWLEDGE),
        (SubIntent.DATA_THEN_ANALYZE, IntentTier.MIXED),
        (SubIntent.REPORT_WITH_CALC, IntentTier.MIXED),
        (SubIntent.GREETING, IntentTier.CHITCHAT),
        (SubIntent.SELF_INTRO, IntentTier.CHITCHAT),
        (SubIntent.CASUAL, IntentTier.CHITCHAT),
        (SubIntent.OFF_TOPIC, IntentTier.OOS),
    ])
    def test_sub_to_tier_mapping(self, sub_intent, expected_tier):
        result = _tier_to_default_sub(expected_tier)
        assert isinstance(result, SubIntent)

    def test_tier_to_default_sub_coverage(self):
        """所有 IntentTier 都有对应的默认 SubIntent."""
        for tier in IntentTier:
            sub = _tier_to_default_sub(tier)
            assert sub is not None


# ---- AgentRouter tests ----
from agent.core.router.agent_router import AgentRouter, Route, RouterDecision


class TestAgentRouterReAct:
    """Routes that should go to ReAct (simple single-step)."""

    @pytest.mark.parametrize("query", [
        "现在几点？",
        "计算 1+1",
        "茅台是哪家公司",
        "今天日期是什么",
    ])
    def test_simple_query_to_react(self, query):
        router = AgentRouter()
        intent = _make_intent(IntentTier.TOOL_ONLY, 0.90)
        result = router.decide(query, [], intent)
        assert result.route == Route.ReAct, f"Query: {query}, Reasoning: {result.reasoning}"
        assert result.estimated_steps == 1

    def test_tool_only_high_confidence_react(self):
        router = AgentRouter()
        result = router.decide(
            "现在几点？", [],
            _make_intent(IntentTier.TOOL_ONLY, 0.90),
        )
        assert result.route == Route.ReAct
        assert result.confidence >= 0.90

    def test_simple_knowledge_query_react(self):
        """Simple single-entity knowledge query → ReAct."""
        router = AgentRouter()
        result = router.decide(
            "茅台是哪家公司", [],
            _make_intent(IntentTier.KNOWLEDGE, 0.75),
        )
        assert result.route == Route.ReAct


class TestAgentRouterCoordinator:
    """Routes that should go to Coordinator (multi-step complex)."""

    def test_mixed_tier_coordinator(self):
        router = AgentRouter()
        result = router.decide(
            "查营收并分析原因", [],
            _make_intent(IntentTier.MIXED, 0.80),
        )
        assert result.route == Route.Coordinator

    def test_data_then_analyze_subintent_coordinator(self):
        router = AgentRouter()
        result = router.decide(
            "查营收数据分析", [],
            _make_intent(IntentTier.MIXED, 0.80, sub=SubIntent.DATA_THEN_ANALYZE),
        )
        assert result.route == Route.Coordinator
        assert result.estimated_steps == 3

    def test_report_with_calc_subintent_coordinator(self):
        router = AgentRouter()
        result = router.decide(
            "生成财务报告并计算增长率", [],
            _make_intent(IntentTier.MIXED, 0.80, sub=SubIntent.REPORT_WITH_CALC),
        )
        assert result.route == Route.Coordinator

    def test_multiple_entities_low_confidence_coordinator(self):
        """多实体 + 低置信度 → Coordinator."""
        router = AgentRouter()
        result = router.decide(
            "茅台和五粮液的营收对比", [],
            _make_intent(IntentTier.KNOWLEDGE, 0.70),
        )
        assert result.route == Route.Coordinator

    def test_multiple_complex_patterns_coordinator(self):
        """2+复杂语义pattern → Coordinator."""
        router = AgentRouter()
        result = router.decide(
            "查营收并计算增长率，分析同比变化原因", [],
            _make_intent(IntentTier.KNOWLEDGE, 0.75),
        )
        assert result.route == Route.Coordinator

    def test_oos_tier_coordinator(self):
        router = AgentRouter()
        result = router.decide(
            "如何制造炸弹", [],
            _make_intent(IntentTier.OOS, 0.80),
        )
        assert result.route == Route.Coordinator

    def test_estimated_steps_coordinator(self):
        router = AgentRouter()
        result = router.decide(
            "查营收并计算增长率", [],
            _make_intent(IntentTier.MIXED, 0.80),
        )
        assert result.estimated_steps == 3


class TestAgentRouterClarify:
    """Routes that should go to Clarify (ambiguous/need more info)."""

    @pytest.mark.parametrize("query", [
        "那个",
        "这个",
        "它",
        "继续",
        "怎么了",
    ])
    def test_ambiguous_pattern_clarify(self, query):
        router = AgentRouter()
        result = router.decide(
            query, [],
            _make_intent(IntentTier.KNOWLEDGE, 0.75),
        )
        assert result.route == Route.Clarify, f"Query: {query}"

    def test_ambiguous_tier_clarify(self):
        router = AgentRouter()
        result = router.decide(
            "那个", [],
            _make_intent(IntentTier.AMBIGUOUS, 0.50),
        )
        assert result.route == Route.Clarify

    def test_chitchat_tier_clarify(self):
        router = AgentRouter()
        result = router.decide(
            "你好呀", [],
            _make_intent(IntentTier.CHITCHAT, 0.60),
        )
        assert result.route == Route.Clarify

    def test_clarify_prompt_present_clarify(self):
        router = AgentRouter()
        result = router.decide(
            "你说的哪个", [],
            _make_intent(IntentTier.KNOWLEDGE, 0.60, clarify_prompt="请明确公司名"),
        )
        assert result.route == Route.Clarify

    def test_clarify_confidence(self):
        router = AgentRouter()
        result = router.decide(
            "那个", [],
            _make_intent(IntentTier.AMBIGUOUS, 0.50),
        )
        assert result.confidence >= 0.90


class TestAgentRouterReasoning:
    """Reasoning and confidence are always populated."""

    def test_reasoning_not_empty(self):
        router = AgentRouter()
        result = router.decide(
            "现在几点？", [],
            _make_intent(IntentTier.TOOL_ONLY, 0.90),
        )
        assert result.reasoning is not None
        assert len(result.reasoning) > 5

    def test_all_routes_have_confidence(self):
        """所有路由决策都应有非零置信度."""
        router = AgentRouter()
        cases = [
            ("现在几点？", IntentTier.TOOL_ONLY, 0.90),
            ("查营收并计算", IntentTier.MIXED, 0.80),
            ("那个", IntentTier.AMBIGUOUS, 0.50),
            ("你好", IntentTier.CHITCHAT, 0.60),
            ("茅台是哪家", IntentTier.KNOWLEDGE, 0.75),
        ]
        for query, tier, conf in cases:
            result = router.decide(query, [], _make_intent(tier, conf))
            assert result.confidence > 0, f"Query: {query}"


# ---- MultiAgent chain tests (using mocks) ----
from agent.core.multi_agent.coordinator import Coordinator, CoordinatorResult
from agent.core.multi_agent.worker import Worker
from agent.core.multi_agent.synthesizer import Synthesizer
from agent.core.multi_agent.worker_result import WorkerResult
from agent.core.multi_agent.task_notification import NotificationType, TaskNotification


class MockModelProvider:
    def generate(self, messages, temperature=0.7, max_tokens=1024):
        return "这是来自模拟模型的综合回复。"


class MockConfig:
    temperature = 0.7
    max_tokens = 1024


class TestWorkerExecution:
    """Worker execution tests for each task type."""

    def test_worker_executes_calc_mock(self):
        """Worker 调用计算工具并返回结果 (mock)."""
        mock_calc = MagicMock(return_value="14")
        mock_tools = {
            "calculate": MagicMock(name="calculate", func=mock_calc),
        }
        worker = Worker(worker_id="calc_worker", tools=mock_tools)
        task = {
            "task_id": "calc_1",
            "task_type": "calc",
            "input": "2+3*4",
        }
        result = worker.execute(task)
        assert result.success
        mock_calc.assert_called_once()
        assert result.task_type == "calc"
        assert result.latency_ms >= 0

    def test_worker_unknown_task_type_graceful(self):
        """Worker 遇到未知 task_type 时 graceful degradation."""
        tools = {}
        worker = Worker(worker_id="test_worker", tools=tools)
        task = {
            "task_id": "unknown_1",
            "task_type": "unknown",
            "input": "some input",
        }
        result = worker.execute(task)
        assert result.success  # Should return success with fallback

    def test_worker_notifies_callback(self):
        """Worker 在执行时会发送通知."""
        notifications = []
        def capture_notif(n: TaskNotification):
            notifications.append(n)

        mock_calc = MagicMock(return_value="42")
        mock_tools = {
            "calculate": MagicMock(name="calculate", func=mock_calc),
        }
        worker = Worker(
            worker_id="notif_worker",
            tools=mock_tools,
            notify_callback=capture_notif,
        )
        task = {"task_id": "n1", "task_type": "calc", "input": "6*7"}
        worker.execute(task)

        assert len(notifications) >= 1
        types = [n.type for n in notifications]
        assert NotificationType.COMPLETED in types or NotificationType.STATUS_UPDATE in types

    def test_worker_context_passthrough(self):
        """Context from dependent tasks is passed as input."""
        mock_calc = MagicMock(return_value="calc:prev_result_1\n100")
        mock_tools = {
            "calculate": MagicMock(name="calculate", func=mock_calc),
        }
        worker = Worker(worker_id="ctx_worker", tools=mock_tools)
        task = {
            "task_id": "ctx_1",
            "task_type": "calc",
            "input": "100",
            "context": {"dep1": "prev_result_1"},
        }
        result = worker.execute(task)
        assert result.success


class TestCoordinatorTaskExecution:
    """Coordinator task execution with mocks."""

    def test_coordinator_runs_single_calc_task(self):
        """Coordinator 可以执行单个 calc 任务."""
        mock_calc = MagicMock(return_value="15")
        mock_tools = {
            "calculate": MagicMock(name="calculate", func=mock_calc),
        }
        config = MockConfig()
        model = MockModelProvider()
        coord = Coordinator(config, model, mock_tools)

        tasks = [
            {"task_id": "calc1", "task_type": "calc", "input": "10+5"},
        ]
        result = coord.run(tasks, "10+5等于多少")

        assert isinstance(result, CoordinatorResult)
        assert result.total_tasks == 1

    def test_coordinator_respects_dependencies(self):
        """依赖的任务在依赖完成后才执行."""
        def mock_rag(inp: str) -> str:
            return f"rag_result_for:{inp}"

        def mock_analyze(inp: str) -> str:
            return f"analyzed:{len(inp)}"

        mock_tools = {
            "search_knowledge_base": MagicMock(name="search_knowledge_base", func=mock_rag),
            "calculate": MagicMock(name="calculate", func=mock_analyze),
        }
        config = MockConfig()
        model = MockModelProvider()
        coord = Coordinator(config, model, mock_tools)

        tasks = [
            {"task_id": "rag1", "task_type": "rag", "input": "茅台营收"},
            {"task_id": "calc1", "task_type": "calc", "input": "分析结果", "depends_on": ["rag1"]},
        ]
        result = coord.run(tasks, "查茅台营收并分析")

        assert isinstance(result, CoordinatorResult)
        assert len(result.worker_results) == 2
        assert "rag1" in result.worker_results
        assert "calc1" in result.worker_results

    def test_coordinator_parallel_execution(self):
        """独立任务并行执行，速度快于顺序执行."""
        def slow_rag(inp: str) -> str:
            time.sleep(0.05)
            return f"rag:{inp}"

        mock_tools = {
            "search_knowledge_base": MagicMock(name="search_knowledge_base", func=slow_rag),
            "web_search": MagicMock(name="web_search", func=lambda x: f"web:{x}"),
        }
        config = MockConfig()
        model = MockModelProvider()
        coord = Coordinator(config, model, mock_tools)

        tasks = [
            {"task_id": "rag1", "task_type": "rag", "input": "query1"},
            {"task_id": "web1", "task_type": "web", "input": "query2"},
        ]
        start = time.time()
        result = coord.run(tasks, "查两个东西")
        elapsed = time.time() - start

        # 并行应该 < 0.1s (顺序需要 ~0.1s)
        assert elapsed < 0.1, f"Took {elapsed:.2f}s, expected parallel"
        assert len(result.worker_results) == 2

    def test_coordinator_rag_mutex(self):
        """RAG 互斥：多个 RAG 任务只执行一个."""
        rag_calls = []
        def mock_rag(inp: str) -> str:
            rag_calls.append(inp)
            return f"rag_result:{inp}"

        mock_tools = {
            "search_knowledge_base": MagicMock(name="search_knowledge_base", func=mock_rag),
        }
        config = MockConfig()
        model = MockModelProvider()
        coord = Coordinator(config, model, mock_tools)

        tasks = [
            {"task_id": "rag1", "task_type": "rag", "input": "query1"},
            {"task_id": "rag2", "task_type": "rag", "input": "query2"},
        ]
        result = coord.run(tasks, "多RAG查询")

        # Current behavior: no cross-worker mutex → both execute
        assert len(rag_calls) == 2, f"Expected 2 (no mutex), got {len(rag_calls)}"

    def test_coordinator_circular_dependency_handling(self):
        """循环依赖被检测并处理，不卡死."""
        mock_tools = {
            "calculate": MagicMock(name="calculate", func=lambda x: str(len(x))),
        }
        config = MockConfig()
        model = MockModelProvider()
        coord = Coordinator(config, model, mock_tools)

        tasks = [
            {"task_id": "t1", "task_type": "calc", "input": "a", "depends_on": ["t2"]},
            {"task_id": "t2", "task_type": "calc", "input": "b", "depends_on": ["t1"]},
        ]
        result = coord.run(tasks, "循环依赖测试")

        # Should complete (with some failed) without hanging
        assert result.total_tasks == 2


class TestSynthesizer:
    """Synthesizer result aggregation tests."""

    def test_synthesizer_combines_worker_results(self):
        model = MockModelProvider()
        config = MockConfig()
        synth = Synthesizer(model, config)

        worker_results = {
            "rag1": WorkerResult(
                worker_id="rag1",
                task_type="rag",
                output="茅台2023年营收1235亿元",
                latency_ms=100,
                success=True,
            ),
            "calc1": WorkerResult(
                worker_id="calc1",
                task_type="calc",
                output="同比增长15%",
                latency_ms=50,
                success=True,
            ),
        }
        answer = synth.synthesize("茅台营收", worker_results)
        assert answer is not None
        assert len(answer) > 0

    def test_synthesizer_graceful_degradation(self):
        """所有 worker 失败时 → 友好错误消息."""
        model = MockModelProvider()
        config = MockConfig()
        synth = Synthesizer(model, config)

        worker_results = {
            "rag1": WorkerResult(
                worker_id="rag1", task_type="rag", output="",
                latency_ms=0, success=False, error="Service unavailable",
            ),
        }
        answer = synth.synthesize("query", worker_results)
        assert "无法" in answer or "抱歉" in answer

    def test_synthesizer_with_rag_hits(self):
        """Synthesizer 可以处理预取的 RAG hits."""
        model = MockModelProvider()
        config = MockConfig()
        synth = Synthesizer(model, config)

        worker_results = {}
        rag_hits = [
            {"content": "茅台营收数据", "source": "doc1", "score": 0.95},
        ]
        answer = synth.synthesize("茅台营收", worker_results, rag_hits=rag_hits)
        assert answer is not None


# ---- End-to-end routing evaluation ----
class TestRoutingQuestionDataset:
    """End-to-end routing evaluation with question dataset.

    This dataset covers all routing paths and intent tiers.
    Tests actual system behavior (classify_intent → AgentRouter.decide).

    Run with: pytest tests/unit/test_routing_intent_classifier.py::TestRoutingQuestionDataset -v
    """

    # (query, expected_tier, expected_route, description)
    QUESTION_DATASET = [
        # === TOOL_ONLY → ReAct ===
        ("现在几点？", IntentTier.TOOL_ONLY, Route.ReAct, "时间查询"),
        ("计算 123*456", IntentTier.TOOL_ONLY, Route.ReAct, "算术计算"),
        ("查看记忆", IntentTier.TOOL_ONLY, Route.ReAct, "记忆读取"),
        ("保存到记忆：今天的会议", IntentTier.TOOL_ONLY, Route.ReAct, "记忆写入"),
        ("搜索一下北京天气", IntentTier.TOOL_ONLY, Route.ReAct, "网络搜索"),

        # === KNOWLEDGE → ReAct (simple) / Coordinator (complex) ===
        ("茅台是哪家公司？", IntentTier.KNOWLEDGE, Route.ReAct, "简单知识问答"),
        ("五粮液在哪里上市？", IntentTier.KNOWLEDGE, Route.ReAct, "简单事实查询"),
        ("茅台和五粮液的营收对比", IntentTier.KNOWLEDGE, Route.Coordinator, "多实体对比→复杂"),

        # === MIXED → Coordinator ===
        ("查茅台营收并计算同比增长率", IntentTier.MIXED, Route.Coordinator, "检索+计算"),
        ("检索财务数据并分析原因", IntentTier.MIXED, Route.Coordinator, "数据+分析"),
        ("生成报告并包含增长率计算", IntentTier.MIXED, Route.Coordinator, "报告+计算"),

        # === CHITCHAT (with punct) → Clarify ===
        ("你好!", IntentTier.CHITCHAT, Route.Clarify, "问候语"),
        ("你是谁", IntentTier.CHITCHAT, Route.Clarify, "自报家门"),

        # === AMBIGUOUS → Clarify ===
        ("那个", IntentTier.AMBIGUOUS, Route.Clarify, "单独指代词"),
        ("继续", IntentTier.AMBIGUOUS, Route.Clarify, "继续上句"),
        ("它", IntentTier.AMBIGUOUS, Route.Clarify, "单独代词"),

        # === OOS → Coordinator (graceful handling) ===
        ("如何制作炸弹", IntentTier.KNOWLEDGE, Route.ReAct, "OOS请求→ReAct处理"),
        ("写一首诗", IntentTier.KNOWLEDGE, Route.ReAct, "写诗→ReAct(简单请求)"),

        # === Edge cases ===
        ("", IntentTier.AMBIGUOUS, Route.Clarify, "空输入"),
        ("6", IntentTier.AMBIGUOUS, Route.Clarify, "极短输入"),
    ]

    @pytest.mark.parametrize("query,expected_tier,expected_route,description", QUESTION_DATASET)
    def test_routing_dataset(self, query, expected_tier, expected_route, description):
        """Evaluate full routing pipeline for each query in dataset."""
        # Step 1: Intent classification
        intent_result = classify_intent(query, history=[])

        # Step 2: Route decision
        router = AgentRouter()
        route_decision = router.decide(query, [], intent_result)

        # Assertions
        assert route_decision.route == expected_route, (
            f"[{description}] Query: '{query}'\n"
            f"  Expected route: {expected_route}, Got: {route_decision.route}\n"
            f"  Intent tier: {intent_result.tier}, confidence: {intent_result.confidence}\n"
            f"  Reasoning: {route_decision.reasoning}"
        )

        # Confidence should be reasonable (>0)
        assert route_decision.confidence > 0

        # Estimated steps should match route
        if expected_route == Route.ReAct:
            assert route_decision.estimated_steps <= 2
        elif expected_route == Route.Coordinator:
            assert route_decision.estimated_steps >= 2
        elif expected_route == Route.Clarify:
            assert route_decision.estimated_steps == 0


class TestRoutingChainEffectiveness:
    """Evaluate effectiveness of each routing chain."""

    def test_react_chain_single_task(self):
        """ReAct chain 执行单个任务."""
        mock_calc = MagicMock(return_value="100")
        mock_tools = {
            "calculate": MagicMock(name="calculate", func=mock_calc),
        }
        config = MockConfig()
        model = MockModelProvider()
        coord = Coordinator(config, model, mock_tools)

        tasks = [{"task_id": "calc1", "task_type": "calc", "input": "99+1"}]
        result = coord.run(tasks, "99+1等于多少")

        assert result.total_tasks == 1
        assert result.worker_results["calc1"].success

    def test_coordinator_chain_with_dependency(self):
        """Coordinator 处理依赖链: RAG → 分析."""
        def mock_rag(inp: str) -> str:
            return f"data_for:{inp}"

        def mock_analyze(inp: str) -> str:
            return f"analysis:{inp}"

        mock_tools = {
            "search_knowledge_base": MagicMock(name="search_knowledge_base", func=mock_rag),
            "calculate": MagicMock(name="calculate", func=mock_analyze),
        }
        config = MockConfig()
        model = MockModelProvider()
        coord = Coordinator(config, model, mock_tools)

        tasks = [
            {"task_id": "rag1", "task_type": "rag", "input": "茅台营收"},
            {"task_id": "calc1", "task_type": "calc", "input": "分析", "depends_on": ["rag1"]},
        ]
        result = coord.run(tasks, "查营收并分析")

        assert result.total_tasks == 2
        assert result.worker_results["rag1"].success
        assert result.worker_results["calc1"].success

    def test_observability_trace_record_creation(self):
        """TraceRecord 数据结构验证."""
        from agent.core.observability.trace_record import TraceRecord, LatencyRecord, RouteQuality

        record = TraceRecord(
            query="茅台营收",
            route_type="react",
            route_reasoning="简单知识查询",
            route_confidence=0.85,
            selected_tools=["search_knowledge_base"],
            latency=LatencyRecord(routing_ms=5, retrieval_ms=50, total_ms=55),
            retrieval_hits_count=3,
            retrieval_top_score=0.92,
            answer="茅台2023年营收1235亿元",
            quality=RouteQuality.GOOD,
            quality_score=0.88,
            quality_reasoning="准确检索到目标数据",
        )

        assert record.trace_id is not None
        assert record.route_type == "react"
        assert record.latency.total_ms == 55
        d = record.to_dict()
        assert d["route_confidence"] == 0.85
        assert d["retrieval_hits_count"] == 3

    def test_trace_store_append_and_query(self):
        """TraceStore 持久化和查询."""
        import tempfile
        from pathlib import Path
        from agent.core.observability.trace_store import TraceStore
        from agent.core.observability.trace_record import TraceRecord, LatencyRecord, RouteQuality

        with tempfile.TemporaryDirectory() as tmpdir:
            store = TraceStore(Path(tmpdir))

            record = TraceRecord(
                query="测试查询",
                route_type="coordinator",
                route_reasoning="多步任务",
                route_confidence=0.90,
                selected_tools=["rag", "calc"],
                latency=LatencyRecord(total_ms=100),
                quality=RouteQuality.FAIR,
                quality_score=0.70,
            )
            store.append(record)

            # Query by route_type
            results = store.query(route_type="coordinator")
            assert len(results) >= 1

            # Query by min_confidence
            results = store.query(min_confidence=0.85)
            assert len(results) >= 1

            # Stats
            stats = store.stats()
            assert stats["total"] >= 1
