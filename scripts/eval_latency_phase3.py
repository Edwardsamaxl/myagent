#!/usr/bin/env python3
"""
Phase 3.1: 全链路延迟测试

测量从用户提问到最终回答的端到端延迟，拆解各阶段耗时：
- 路由决策延迟 (router)
- Query Rewrite 延迟
- 检索延迟 (retrieval)
- 重排延迟 (rerank)
- 总延迟 (total)

用法:
    python scripts/eval_latency_phase3.py

输出:
    data/eval/latency_phase3_results.json
"""

from __future__ import annotations

import codecs
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# Fix Windows UTF-8 console output
if sys.platform == "win32":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "backslashreplace")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "backslashreplace")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from agent.config import AgentConfig
from agent.llm.providers import AnthropicCompatibleProvider
from agent.llm.embeddings import OllamaEmbeddingProvider, build_embedding_provider
from agent.rag.retrieval import InMemoryHybridRetriever
from agent.rag.rerank import SimpleReranker, BGEReranker
from agent.core.routing.llm_router import LLMRouterAgent
from agent.core.routing.rag_chain import RAGChain


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3.1 测试用例（L1-L10）
# ─────────────────────────────────────────────────────────────────────────────
LATENCY_TEST_CASES = [
    {"id": "L1", "query": "茅台是哪家公司？", "route_type": "SINGLE_STEP"},
    {"id": "L2", "query": "茅台2024年的营收是多少？", "route_type": "MULTI_STEP"},
    {"id": "L3", "query": "茅台和五粮液的ROE对比", "route_type": "MULTI_STEP"},
    {"id": "L4", "query": "帮我查查这家公司的情况", "route_type": "CLARIFY"},
    {"id": "L5", "query": "茅台2024年营收是三年前的几倍？", "route_type": "MULTI_STEP"},
    {"id": "L6", "query": "五粮液的主营业务是什么？", "route_type": "SINGLE_STEP"},
    {"id": "L7", "query": "那个大公司的市值多少？", "route_type": "CLARIFY"},
    {"id": "L8", "query": "茅台过去3年营收复合增长率是多少？", "route_type": "MULTI_STEP"},
    {"id": "L9", "query": "主要靠什么赚钱", "route_type": "CLARIFY"},
    {"id": "L10", "query": "把茅台近5年的营收做成表格", "route_type": "MULTI_STEP"},
]


# ─────────────────────────────────────────────────────────────────────────────
# 数据类
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StageLatency:
    router_ms: int = 0      # 路由决策延迟
    rewrite_ms: int = 0     # Query Rewrite 延迟
    retrieval_ms: int = 0   # 检索延迟
    rerank_ms: int = 0     # 重排延迟
    total_ms: int = 0       # 总延迟


@dataclass
class LatencyResult:
    query_id: str
    query: str
    expected_route: str
    actual_route: str
    latency: StageLatency
    hits_count: int
    rewritten_query: str
    error: str | None


# ─────────────────────────────────────────────────────────────────────────────
# 统计工具
# ─────────────────────────────────────────────────────────────────────────────

def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = int(len(s) * p)
    return s[min(idx, len(s) - 1)]


def aggregate_latency(results: list[LatencyResult]) -> dict:
    valid = [r for r in results if r.error is None]
    errors = len(results) - len(valid)

    if not valid:
        return {"valid": 0, "errors": errors}

    router_ms = [float(r.latency.router_ms) for r in valid]
    rewrite_ms = [float(r.latency.rewrite_ms) for r in valid]
    retrieval_ms = [float(r.latency.retrieval_ms) for r in valid]
    rerank_ms = [float(r.latency.rerank_ms) for r in valid]
    total_ms = [float(r.latency.total_ms) for r in valid]

    def stats(data):
        return {
            "p50": round(percentile(data, 0.50), 1),
            "p95": round(percentile(data, 0.95), 1),
            "p99": round(percentile(data, 0.99), 1),
            "avg": round(sum(data) / len(data), 1),
            "min": round(min(data), 1),
            "max": round(max(data), 1),
        }

    return {
        "valid": len(valid),
        "errors": errors,
        "router_ms": stats(router_ms),
        "rewrite_ms": stats(rewrite_ms),
        "retrieval_ms": stats(retrieval_ms),
        "rerank_ms": stats(rerank_ms),
        "total_ms": stats(total_ms),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 主测试逻辑
# ─────────────────────────────────────────────────────────────────────────────

def run_latency_test(config: AgentConfig) -> list[LatencyResult]:
    print("\n" + "=" * 60)
    print("Phase 3.1: 全链路延迟测试")
    print("=" * 60)

    # 初始化 Provider
    try:
        provider = AnthropicCompatibleProvider(
            base_url=config.anthropic_base_url,
            api_key=config.anthropic_api_key,
            model_name=config.model_name,
        )
        print(f"  [OK] Provider: {config.model_provider}/{config.model_name}")
    except Exception as e:
        print(f"  [FAIL] Provider 初始化失败: {e}")
        return []

    # 初始化 Router（用于测量路由决策延迟）
    router = LLMRouterAgent(model_provider=provider, config={})

    # 初始化 Retriever
    embed_provider = build_embedding_provider(config)
    retriever = InMemoryHybridRetriever(
        lexical_weight=config.retrieval_lexical_weight,
        tfidf_weight=config.retrieval_tfidf_weight,
        embedding_weight=config.retrieval_embedding_weight,
    )
    if embed_provider:
        retriever._embedding_provider = embed_provider

    index_dir = config.data_dir / "retrieval_index"
    if index_dir.exists():
        retriever.load_index(index_dir)
        print(f"  [OK] 索引已加载: {len(retriever._chunks)} chunks")
    else:
        print(f"  [WARN] 索引目录不存在: {index_dir}")

    # 初始化 Reranker
    if config.rerank_provider == "ollama":
        reranker = BGEReranker(
            base_url=config.rerank_base_url,
            model="dengcao/bge-reranker-v2-m3",
        )
        if not reranker._health_check():
            print("  [WARN] BGE Reranker 健康检查失败，降级为 SimpleReranker")
            reranker = SimpleReranker()
    else:
        reranker = SimpleReranker()

    # 初始化 RAG Chain（不包含路由）
    rag = RAGChain(
        config={
            "retrieval_top_k": config.retrieval_top_k,
            "rerank_top_k": config.rerank_top_k,
        },
        retriever=retriever,
        reranker=reranker,
        model_provider=provider,
    )

    results = []
    for tc in LATENCY_TEST_CASES:
        query = tc["query"]
        expected = tc["route_type"]
        print(f"\n  [{tc['id']}] {query[:40]}...")

        try:
            # 1. 路由决策延迟（单独测量）
            t0 = time.time()
            routing_context = {"available_tools": [], "history": None}
            decision = router.route(query, context=routing_context)
            router_ms = int((time.time() - t0) * 1000)
            actual_route = decision.route_type.value.upper()

            # 2. RAG Chain（rewrite + retrieval + rerank）
            t1 = time.time()
            rag_result = rag.execute(query, rewrite_mode="hybrid")
            rag_total_ms = int((time.time() - t1) * 1000)

            latency = StageLatency(
                router_ms=router_ms,
                rewrite_ms=rag_result["latency"]["rewrite_ms"],
                retrieval_ms=rag_result["latency"]["retrieval_ms"],
                rerank_ms=rag_result["latency"]["rerank_ms"],
                total_ms=router_ms + rag_total_ms,
            )

            results.append(LatencyResult(
                query_id=tc["id"],
                query=query,
                expected_route=expected,
                actual_route=actual_route,
                latency=latency,
                hits_count=len(rag_result["hits"]),
                rewritten_query=rag_result.get("rewritten_query", ""),
                error=None,
            ))

            print(f"      路由={actual_route} | "
                  f"总延迟={latency.total_ms}ms "
                  f"(路由={latency.router_ms}ms, "
                  f"改写={latency.rewrite_ms}ms, "
                  f"检索={latency.retrieval_ms}ms, "
                  f"重排={latency.rerank_ms}ms)")

        except Exception as e:
            err = str(e)
            results.append(LatencyResult(
                query_id=tc["id"],
                query=query,
                expected_route=expected,
                actual_route="ERROR",
                latency=StageLatency(),
                hits_count=0,
                rewritten_query="",
                error=err[:100],
            ))
            print(f"      [FAIL] 错误: {err[:60]}")

        time.sleep(1.5)  # 避免 API 突发限流

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 输出汇总报告
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: list[LatencyResult], agg: dict) -> None:
    print("\n" + "=" * 60)
    print("Phase 3.1 汇总报告")
    print("=" * 60)

    print(f"\n有效测试: {agg['valid']}/{len(results)} 条")
    if agg['errors'] > 0:
        print(f"失败: {agg['errors']} 条")

    print("\n【分阶段延迟统计 (ms)】")
    print(f"{'阶段':<12} {'P50':>8} {'P95':>8} {'P99':>8} {'均值':>8} {'最小':>8} {'最大':>8}")
    print("-" * 64)

    stages = [
        ("路由决策", "router_ms"),
        ("Query改写", "rewrite_ms"),
        ("检索", "retrieval_ms"),
        ("重排", "rerank_ms"),
        ("总延迟", "total_ms"),
    ]
    for name, key in stages:
        s = agg[key]
        print(f"{name:<12} {s['p50']:>8.1f} {s['p95']:>8.1f} {s['p99']:>8.1f} {s['avg']:>8.1f} {s['min']:>8.1f} {s['max']:>8.1f}")

    # 延迟占比饼图（用文字表示）
    total_avg = agg["total_ms"]["avg"]
    if total_avg > 0:
        print(f"\n【延迟占比分析】(基于平均总延迟 {total_avg:.1f}ms)")
        for name, key in stages[:-1]:  # 不重复 total
            s = agg[key]
            pct = (s["avg"] / total_avg) * 100 if total_avg > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"  {name:<10}: {pct:5.1f}% {bar}")

    # 各查询详情
    print(f"\n【逐条详情】")
    print(f"{'ID':<4} {'路由(期望/实际)':<24} {'总ms':>8} {'路由':>8} {'改写':>8} {'检索':>8} {'重排':>8}")
    print("-" * 72)
    for r in results:
        if r.error:
            print(f"{r.query_id:<4} {'ERROR':<24} {'-':>8}")
        else:
            route_info = f"{r.expected_route}/{r.actual_route}"
            print(f"{r.query_id:<4} {route_info:<24} {r.latency.total_ms:>8} "
                  f"{r.latency.router_ms:>8} {r.latency.rewrite_ms:>8} "
                  f"{r.latency.retrieval_ms:>8} {r.latency.rerank_ms:>8}")

    # 瓶颈分析
    print(f"\n【瓶颈分析】")
    stage_avgs = {name: agg[key]["avg"] for name, key in stages}
    sorted_stages = sorted(stage_avgs.items(), key=lambda x: x[1], reverse=True)
    bottleneck = sorted_stages[0][0] if sorted_stages else "N/A"
    print(f"  主要瓶颈: {bottleneck} (平均 {stage_avgs.get(bottleneck, 0):.1f}ms)")

    # P95 延迟目标检查
    p95_total = agg["total_ms"]["p95"]
    print(f"\n【P95 延迟目标检查】")
    TARGETS = {"P50": 2000, "P95": 5000}  # ms
    for pct, target in TARGETS.items():
        actual = agg["total_ms"][pct.lower()]
        status = "✓ 通过" if actual <= target else "✗ 未达标"
        print(f"  {pct} 总延迟: {actual:.1f}ms (目标 ≤{target}ms) {status}")


def main():
    config = AgentConfig.from_env()
    output_path = Path("data/eval/latency_phase3_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 运行测试
    results = run_latency_test(config)

    # 聚合统计
    agg = aggregate_latency(results)

    # 打印汇总
    print_summary(results, agg)

    # 保存 JSON
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "test_cases": LATENCY_TEST_CASES,
        "summary": agg,
        "details": [
            {
                "query_id": r.query_id,
                "query": r.query,
                "expected_route": r.expected_route,
                "actual_route": r.actual_route,
                "latency": asdict(r.latency),
                "hits_count": r.hits_count,
                "rewritten_query": r.rewritten_query,
                "error": r.error,
            }
            for r in results
        ],
    }
    output_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
