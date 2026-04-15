"""Test script for LLM Router routing accuracy.

Tests the LLMRouterAgent with 20+ diverse queries, measures latency,
and generates a report to data/routing_observability/llm_router_report.json.

Usage:
    python scripts/test_llm_router.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

# Add project root to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.config import AgentConfig
from agent.llm.providers import build_model_provider, ModelProvider
from agent.core.routing.llm_router import LLMRouterAgent
from agent.core.routing.route_decision import RouteType


# Test dataset: (query, expected_route_type, description)
TEST_QUERIES = [
    # === TOOL_ONLY ===
    ("现在几点", RouteType.SINGLE_STEP, "TOOL_ONLY: 时间查询"),
    ("计算 1+1", RouteType.SINGLE_STEP, "TOOL_ONLY: 简单计算"),
    ("今天日期是什么", RouteType.SINGLE_STEP, "TOOL_ONLY: 日期查询"),
    ("查看记忆", RouteType.SINGLE_STEP, "TOOL_ONLY: 记忆读取"),
    ("搜索北京天气", RouteType.SINGLE_STEP, "TOOL_ONLY: 搜索"),

    # === KNOWLEDGE ===
    ("茅台是哪家公司", RouteType.SINGLE_STEP, "KNOWLEDGE: 简单公司查询"),
    ("五粮液主营什么", RouteType.SINGLE_STEP, "KNOWLEDGE: 业务范围查询"),
    ("茅台的成立时间", RouteType.SINGLE_STEP, "KNOWLEDGE: 简单事实查询"),

    # === MIXED (complex, multi-step) ===
    ("查茅台营收并计算增长率", RouteType.MULTI_STEP, "MIXED: 检索+计算"),
    ("检索财务数据并分析原因", RouteType.MULTI_STEP, "MIXED: 数据+分析"),
    ("对比茅台和五粮液的营收", RouteType.MULTI_STEP, "MIXED: 多实体对比"),
    ("查营收并生成报告", RouteType.MULTI_STEP, "MIXED: 检索+报告"),

    # === CHITCHAT ===
    ("你好", RouteType.CLARIFY, "CHITCHAT: 问候"),
    ("今天心情不错", RouteType.CLARIFY, "CHITCHAT: 闲聊"),
    ("谢谢你的帮助", RouteType.CLARIFY, "CHITCHAT: 感谢"),

    # === AMBIGUOUS ===
    ("那个", RouteType.CLARIFY, "AMBIGUOUS: 单独指代词"),
    ("继续", RouteType.CLARIFY, "AMBIGUOUS: 继续上句"),
    ("怎么了", RouteType.CLARIFY, "AMBIGUOUS: 单独疑问词"),
    ("嗯", RouteType.CLARIFY, "AMBIGUOUS: 简单语气词"),

    # === Additional edge cases ===
    ("帮我查一下", RouteType.CLARIFY, "AMBIGUOUS: 无宾语查询"),
    ("说一下", RouteType.CLARIFY, "AMBIGUOUS: 无内容查询"),
    ("继续刚才的", RouteType.CLARIFY, "AMBIGUOUS: 指代不明"),
    ("分析一下", RouteType.MULTI_STEP, "KNOWLEDGE: 需要分析但无具体对象"),
]


def run_llm_router_test(
    model_provider: ModelProvider,
    config: AgentConfig,
) -> dict:
    """Run LLM Router test with the given model provider."""

    # Initialize LLM Router
    router_agent = LLMRouterAgent(model_provider, {})

    results = []
    latencies = []
    correct = 0
    incorrect = 0

    for query, expected_route, description in TEST_QUERIES:
        start_time = time.time()

        try:
            # Perform routing decision
            context = {"available_tools": ["search", "calculate", "memory"]}
            decision = router_agent.route(query, context)

            latency_ms = int((time.time() - start_time) * 1000)
            latencies.append(latency_ms)

            # Check if routing matches expected
            is_correct = decision.route_type == expected_route

            result_entry = {
                "query": query,
                "expected_route": expected_route.value,
                "actual_route": decision.route_type.value,
                "correct": is_correct,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
                "latency_ms": latency_ms,
                "description": description,
            }

            if is_correct:
                correct += 1
            else:
                incorrect += 1

            print(f"[{'OK' if is_correct else 'FAIL'}] {query[:30]:<30} | expected: {expected_route.value:<12} | got: {decision.route_type.value:<12} | conf: {decision.confidence:.2f}")

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            latencies.append(latency_ms)

            result_entry = {
                "query": query,
                "expected_route": expected_route.value,
                "actual_route": "ERROR",
                "correct": False,
                "error": str(e),
                "latency_ms": latency_ms,
                "description": description,
            }
            incorrect += 1
            print(f"[ERROR] {query[:30]:<30} | {str(e)[:50]}")

        results.append(result_entry)

    # Calculate statistics
    total = len(TEST_QUERIES)
    accuracy = correct / total if total > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    min_latency = min(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0

    # Breakdown by category
    category_stats = {}
    for result in results:
        expected = result["expected_route"]
        if expected not in category_stats:
            category_stats[expected] = {"correct": 0, "total": 0}
        category_stats[expected]["total"] += 1
        if result["correct"]:
            category_stats[expected]["correct"] += 1

    for cat in category_stats:
        cat_total = category_stats[cat]["total"]
        cat_correct = category_stats[cat]["correct"]
        category_stats[cat]["accuracy"] = cat_correct / cat_total if cat_total > 0 else 0

    report = {
        "test_info": {
            "total_queries": total,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": accuracy,
        },
        "latency_stats": {
            "average_ms": avg_latency,
            "min_ms": min_latency,
            "max_ms": max_latency,
            "total_latency_ms": sum(latencies),
        },
        "category_breakdown": category_stats,
        "results": results,
    }

    return report


def main():
    """Main entry point for LLM Router testing."""

    # Check if LLM is configured
    config = AgentConfig.from_env()

    print("=" * 80)
    print("LLM Router Routing Accuracy Test")
    print("=" * 80)
    print(f"Model Provider: {config.model_provider}")
    print(f"Model Name: {config.model_name}")
    print(f"API Base URL: {config.anthropic_base_url}")
    print()

    # Check if API key is configured
    has_api_key = bool(config.anthropic_api_key)
    print(f"API Key Configured: {has_api_key}")
    print()

    if not has_api_key:
        print("[WARN] No API key configured. Testing with MockProvider instead.")
        print()

    # Try to build model provider
    try:
        model_provider = build_model_provider(config)
        print(f"[OK] Model provider initialized: {type(model_provider).__name__}")
    except Exception as e:
        print(f"[WARN] Failed to initialize model provider: {e}")
        print("[INFO] Falling back to mock provider for testing.")
        from agent.llm.providers import MockProvider
        model_provider = MockProvider(config.model_name)

    # Run tests
    print()
    print("Running routing tests...")
    print("-" * 80)

    report = run_llm_router_test(model_provider, config)

    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Queries: {report['test_info']['total_queries']}")
    print(f"Correct: {report['test_info']['correct']}")
    print(f"Incorrect: {report['test_info']['incorrect']}")
    print(f"Accuracy: {report['test_info']['accuracy']*100:.1f}%")
    print()
    print("Latency Statistics:")
    print(f"  Average: {report['latency_stats']['average_ms']:.1f} ms")
    print(f"  Min: {report['latency_stats']['min_ms']} ms")
    print(f"  Max: {report['latency_stats']['max_ms']} ms")
    print()
    print("Category Breakdown:")
    for cat, stats in report["category_breakdown"].items():
        acc = stats["accuracy"] * 100
        print(f"  {cat}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")

    # Save report
    output_dir = Path("E:/CursorProject/myagent/data/routing_observability")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "llm_router_report.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print(f"Report saved to: {output_file}")

    return report


if __name__ == "__main__":
    main()
