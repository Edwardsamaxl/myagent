#!/usr/bin/env python3
"""
路由与 RAG 系统严格评估脚本

Phase 1: 环境验证
Phase 2: 路由准确性测试 + 端到端延迟测试

用法:
    python scripts/eval_rigorous.py                    # 全量测试
    python scripts/eval_rigorous.py --env-check       # 仅环境验证
    python scripts/eval_rigorous.py --routing-only    # 仅路由
    python scripts/eval_rigorous.py --latency-only    # 仅延迟
    python scripts/eval_rigorous.py --rewrite-ablation # 仅 rewrite 对比
    python scripts/eval_rigorous.py --quality-only    # 仅质量
    python scripts/eval_rigorous.py --all              # 全量
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import codecs
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Fix Windows UTF-8 console output
if sys.platform == "win32":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "backslashreplace")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "backslashreplace")

# ─────────────────────────────────────────────────────────────────────────────
# 项目路径 setup
# ─────────────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from agent.config import AgentConfig
from agent.llm.providers import (
    AnthropicCompatibleProvider,
    build_model_provider,
    Message,
)
from agent.llm.embeddings import OllamaEmbeddingProvider, build_embedding_provider
from agent.rag.retrieval import InMemoryHybridRetriever
from agent.rag.rerank import SimpleReranker, BGEReranker, build_reranker
from agent.rag.schemas import DocumentChunk
from agent.core.dialogue.intent_classifier import classify_intent
from agent.core.dialogue.query_rewrite import QueryRewriteResult, rewrite_for_rag
from agent.core.routing.rag_chain import RAGChain
from agent.core.routing.llm_router import LLMRouterAgent
from agent.core.routing.route_decision import RouteType

# ─────────────────────────────────────────────────────────────────────────────
# 复杂问题库
# ─────────────────────────────────────────────────────────────────────────────
COMPLEX_QUESTIONS: list[dict[str, Any]] = [
    # A. 独立问题（明确，不需要上下文）
    {"id": "A1", "query": "茅台是哪家公司？", "category": "独立问题", "expected_route": "SINGLE_STEP", "gold_chunk_ids": {"600519_info"}, "history": None},
    {"id": "A2", "query": "五粮液的主营业务是什么？", "category": "独立问题", "expected_route": "SINGLE_STEP", "gold_chunk_ids": {"wuliangye_business"}, "history": None},
    {"id": "A3", "query": "茅台2024年的营收是多少？", "category": "独立问题", "expected_route": "MULTI_STEP", "gold_chunk_ids": {"600519_2024_revenue"}, "history": None},
    {"id": "A4", "query": "帮我查查这家白酒公司的营收增长情况", "category": "独立问题", "expected_route": "MULTI_STEP", "gold_chunk_ids": {"600519_growth"}, "history": None},
    {"id": "A5", "query": "上市公司600519的赚钱能力如何？", "category": "独立问题", "expected_route": "MULTI_STEP", "gold_chunk_ids": {"600519_profit"}, "history": None},

    # B. 多跳推理问题
    {"id": "B1", "query": "茅台2024年的营收增长率是多少？比五粮液高多少个百分点？", "category": "多跳推理", "expected_route": "MULTI_STEP", "gold_chunk_ids": {"600519_growth"}, "history": None},
    {"id": "B2", "query": "茅台和五粮液的ROE对比，谁的股东回报更高？", "category": "多跳推理", "expected_route": "MULTI_STEP", "gold_chunk_ids": {"600519_roe"}, "history": None},
    {"id": "B3", "query": "茅台2024年营收是三年前的几倍？", "category": "多跳推理", "expected_route": "MULTI_STEP", "gold_chunk_ids": {"600519_2024_revenue"}, "history": None},
    {"id": "B4", "query": "如果茅台明年保持相同增长率，2025年营收大概多少？", "category": "多跳推理", "expected_route": "MULTI_STEP", "gold_chunk_ids": {"600519_growth"}, "history": None},
    {"id": "B5", "query": "茅台过去3年营收复合增长率是多少？", "category": "多跳推理", "expected_route": "MULTI_STEP", "gold_chunk_ids": {"600519_growth"}, "history": None},

    # C. 指代消解问题
    {"id": "C1", "query": "茅台是哪家公司？", "category": "指代消解", "expected_route": "SINGLE_STEP", "gold_chunk_ids": {"600519_info"}, "history": None},
    {"id": "C2", "query": "它2024年的净利润是多少？", "category": "指代消解", "expected_route": "SINGLE_STEP", "gold_chunk_ids": {"600519_2024_profit"}, "history": "茅台是哪家公司"},
    {"id": "C3", "query": "那它的股价现在多少？", "category": "指代消解", "expected_route": "SINGLE_STEP", "gold_chunk_ids": {"600519_price"}, "history": "茅台是哪家公司"},
    {"id": "C4", "query": "同时期五粮液的净利润是多少？", "category": "指代消解", "expected_route": "SINGLE_STEP", "gold_chunk_ids": {"wuliangye_2024_profit"}, "history": "茅台是哪家公司"},
    {"id": "C5", "query": "为什么营收增长但利润率下降了？", "category": "指代消解", "expected_route": "MULTI_STEP", "gold_chunk_ids": {"600519_2024_change"}, "history": "茅台是哪家公司"},

    # D. 需要上下文的隐含意图
    {"id": "D1", "query": "这公司有没有护城河？", "category": "隐含意图", "expected_route": "CLARIFY", "gold_chunk_ids": {"moat_analysis"}, "history": None},
    {"id": "D2", "query": "和行业平均比怎么样？", "category": "隐含意图", "expected_route": "CLARIFY", "gold_chunk_ids": {"industry_avg"}, "history": None},
    {"id": "D3", "query": "茅台现在还能买吗？", "category": "隐含意图", "expected_route": "SINGLE_STEP", "gold_chunk_ids": {"600519_analysis"}, "history": None},
    {"id": "D4", "query": "这公司值得投资吗？", "category": "隐含意图", "expected_route": "SINGLE_STEP", "gold_chunk_ids": {"600519_analysis"}, "history": "茅台现在还能买吗？"},
    {"id": "D5", "query": "三年后大概率多少？", "category": "隐含意图", "expected_route": "MULTI_STEP", "gold_chunk_ids": {"600519_forecast"}, "history": "茅台现在还能买吗？"},

    # E. 模糊表述问题（需要澄清）
    {"id": "E1", "query": "那个大公司", "category": "模糊表述", "expected_route": "CLARIFY", "gold_chunk_ids": {"600519"}, "history": None},
    {"id": "E2", "query": "他们的最新年报", "category": "模糊表述", "expected_route": "CLARIFY", "gold_chunk_ids": {"maotai_2024_annual"}, "history": None},
    {"id": "E3", "query": "业绩最好的季度是哪季", "category": "模糊表述", "expected_route": "CLARIFY", "gold_chunk_ids": {"600519_quarterly"}, "history": None},
    {"id": "E4", "query": "主要靠什么赚钱", "category": "模糊表述", "expected_route": "CLARIFY", "gold_chunk_ids": {"maotai_main_business"}, "history": None},
    {"id": "E5", "query": "同比变化大吗", "category": "模糊表述", "expected_route": "CLARIFY", "gold_chunk_ids": {"600519_yoy"}, "history": None},

    # F. 复合问题
    {"id": "F1", "query": "查茅台营收，然后和行业平均对比，再计算超额收益", "category": "复合意图", "expected_route": "MULTI_STEP", "gold_chunk_ids": {"600519_revenue"}, "history": None},
    {"id": "F2", "query": "帮我算一下茅台ROE，然后判断这个水平在行业里算什么档次", "category": "复合意图", "expected_route": "MULTI_STEP", "gold_chunk_ids": {"600519_roe"}, "history": None},
    {"id": "F3", "query": "五粮液和茅台谁更赚钱？给出具体数据支撑结论", "category": "复合意图", "expected_route": "MULTI_STEP", "gold_chunk_ids": {"maotai_vs_wuliangye"}, "history": None},
    {"id": "F4", "query": "茅台的市值现在是多少？换算成美元呢？", "category": "复合意图", "expected_route": "MULTI_STEP", "gold_chunk_ids": {"600519_market_cap"}, "history": None},
    {"id": "F5", "query": "把茅台近5年的营收做成表格，然后分析趋势", "category": "复合意图", "expected_route": "MULTI_STEP", "gold_chunk_ids": {"maotai_5yr_revenue"}, "history": None},

]

REWRITE_MODES = ["rule", "llm", "hyde", "expand", "hyde_expand"]

# ─────────────────────────────────────────────────────────────────────────────
# 数据类
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RoutingResult:
    query_id: str
    query: str
    expected_route: str
    actual_route: str
    selected_tools: list[str]
    confidence: float
    is_correct: bool
    error: str | None  # "500" / "529" / None


@dataclass
class LatencyResult:
    query_id: str
    rewrite_ms: int
    retrieval_ms: int
    rerank_ms: int
    total_ms: int
    hits_count: int
    rewritten_query: str
    error: str | None


@dataclass
class RewriteAblationResult:
    query_id: str
    mode: str
    recall_at_3: float
    mrr: float
    ndcg_at_3: float
    hit_overlap_with_rule: float | None
    rewritten_query: str
    error: str | None


@dataclass
class QualityResult:
    query_id: str
    question: str
    generated_answer: str
    accuracy: float
    completeness: float
    relevance: float
    citation_correctness: float
    hallucination: float
    overall_quality: float
    reasoning: str
    error: str | None


# ─────────────────────────────────────────────────────────────────────────────
# 指标计算
# ─────────────────────────────────────────────────────────────────────────────

def _recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & gold) / len(gold)


def _mrr(retrieved: list[str], gold: set[str]) -> float:
    for i, rid in enumerate(retrieved, start=1):
        if rid in gold:
            return 1.0 / i
    return 0.0


def _ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, rid in enumerate(retrieved[:k], 1) if rid in gold)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = int(len(s) * p)
    return s[min(idx, len(s) - 1)]


def aggregate_routing_results(results: list[RoutingResult]) -> dict[str, Any]:
    valid = [r for r in results if r.error is None]
    errors_500529 = [r for r in results if r.error is not None]

    if not valid:
        return {
            "total": len(results),
            "valid": 0,
            "errors_500_529": len(errors_500529),
            "accuracy": 0.0,
            "by_category": {},
            "error_samples": [{"query_id": r.query_id, "error": r.error} for r in errors_500529[:3]],
        }

    correct = [r for r in valid if r.is_correct]

    by_category: dict[str, dict] = {}
    for r in valid:
        cat = r.query_id[0]  # A/B/C/D/E/F
        if cat not in by_category:
            by_category[cat] = {"correct": 0, "total": 0}
        by_category[cat]["total"] += 1
        if r.is_correct:
            by_category[cat]["correct"] += 1

    return {
        "total": len(results),
        "valid": len(valid),
        "errors_500_529": len(errors_500529),
        "accuracy": round(len(correct) / len(valid), 4) if valid else 0.0,
        "correct_count": len(correct),
        "by_category": {
            cat: round(v["correct"] / v["total"], 4) if v["total"] > 0 else 0
            for cat, v in by_category.items()
        },
        "error_samples": [{"query_id": r.query_id, "error": r.error} for r in errors_500529[:3]],
    }


def aggregate_latency_results(results: list[LatencyResult]) -> dict[str, Any]:
    valid = [r for r in results if r.error is None]
    errors = len(results) - len(valid)

    if not valid:
        return {"valid_count": 0, "error_count": errors}

    totals = [float(r.total_ms) for r in valid]
    retrievals = [float(r.retrieval_ms) for r in valid]
    rewrites = [float(r.rewrite_ms) for r in valid]
    reranks = [float(r.rerank_ms) for r in valid]

    return {
        "valid_count": len(valid),
        "error_count": errors,
        "p50_total_ms": round(percentile(totals, 0.5), 1),
        "p95_total_ms": round(percentile(totals, 0.95), 1),
        "p99_total_ms": round(percentile(totals, 0.99), 1),
        "p50_retrieval_ms": round(percentile(retrievals, 0.5), 1),
        "p95_retrieval_ms": round(percentile(retrievals, 0.95), 1),
        "avg_rewrite_ms": round(sum(rewrites) / len(rewrites), 1),
        "avg_rerank_ms": round(sum(reranks) / len(reranks), 1),
        "min_total_ms": round(min(totals), 1),
        "max_total_ms": round(max(totals), 1),
    }


def aggregate_rewrite_ablation(results: list[RewriteAblationResult]) -> dict[str, Any]:
    by_mode: dict[str, list[RewriteAblationResult]] = {}
    for r in results:
        if r.error is None:
            by_mode.setdefault(r.mode, []).append(r)

    summary: dict[str, Any] = {}
    for mode, mode_results in by_mode.items():
        recalls = [r.recall_at_3 for r in mode_results]
        mrrs = [r.mrr for r in mode_results]
        ndcgs = [r.ndcg_at_3 for r in mode_results]
        overlaps = [r.hit_overlap_with_rule for r in mode_results if r.hit_overlap_with_rule is not None]

        summary[mode] = {
            "mean_recall_at_3": round(sum(recalls) / len(recalls), 4),
            "mean_mrr": round(sum(mrrs) / len(mrrs), 4),
            "mean_ndcg_at_3": round(sum(ndcgs) / len(ndcgs), 4),
            "mean_hit_overlap_with_rule": round(sum(overlaps) / len(overlaps), 4) if overlaps else 1.0,
            "count": len(mode_results),
        }

    # 相对 rule 的提升
    if "rule" in summary:
        rule_base = summary["rule"]
        for mode, stats in summary.items():
            if mode != "rule":
                stats["recall_improvement"] = round(stats["mean_recall_at_3"] - rule_base["mean_recall_at_3"], 4)
                stats["mrr_improvement"] = round(stats["mean_mrr"] - rule_base["mean_mrr"], 4)
                stats["ndcg_improvement"] = round(stats["mean_ndcg_at_3"] - rule_base["mean_ndcg_at_3"], 4)

    return summary


def aggregate_quality_results(results: list[QualityResult]) -> dict[str, Any]:
    valid = [r for r in results if r.error is None]
    errors = len(results) - len(valid)

    if not valid:
        return {"valid": 0, "error_count": errors}

    dims = ["accuracy", "completeness", "relevance", "citation_correctness", "hallucination", "overall_quality"]
    summary: dict[str, Any] = {"valid": len(valid), "error_count": errors}

    for dim in dims:
        vals = [getattr(r, dim) for r in valid]
        summary[f"mean_{dim}"] = round(sum(vals) / len(vals), 3)
        summary[f"min_{dim}"] = round(min(vals), 3)
        summary[f"max_{dim}"] = round(max(vals), 3)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: 环境验证
# ─────────────────────────────────────────────────────────────────────────────

def phase1_env_check() -> dict[str, Any]:
    """验证所有组件可用性"""
    print("\n" + "=" * 60)
    print("Phase 1: 环境验证")
    print("=" * 60)

    config = AgentConfig.from_env()
    checks: dict[str, Any] = {}

    # 1. MiniMax API
    print("\n[1/5] 检查 MiniMax API...")
    try:
        provider = AnthropicCompatibleProvider(
            base_url=config.anthropic_base_url,
            api_key=config.anthropic_api_key,
            model_name=config.model_name,
        )
        resp = provider.generate(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.1,
            max_tokens=10,
        )
        checks["minimax"] = {"status": "OK", "response_preview": resp[:50]}
        print(f"  [OK] MiniMax API 可用，响应: {resp[:30]}...")
    except Exception as e:
        err_msg = str(e)
        checks["minimax"] = {"status": "FAIL", "error": err_msg[:100]}
        print(f"  [FAIL] MiniMax API 失败: {err_msg[:80]}")

    # 2. Ollama embedding (qwen3-embedding)
    print("\n[2/5] 检查 Ollama Embedding (qwen3-embedding)...")
    try:
        embed_provider = OllamaEmbeddingProvider(
            base_url=config.ollama_base_url,
            model_name=config.embedding_model,
        )
        vecs = embed_provider.embed_texts(["茅台"])
        checks["embedding"] = {"status": "OK", "embedding_dim": len(vecs[0]) if vecs else 0}
        print(f"  [OK] Embedding 可用，向量维度: {len(vecs[0]) if vecs else 'N/A'}")
    except Exception as e:
        err_msg = str(e)
        checks["embedding"] = {"status": "FAIL", "error": err_msg[:100]}
        print(f"  [FAIL] Embedding 失败: {err_msg[:80]}")

    # 3. Ollama BGE Reranker
    print("\n[3/5] 检查 Ollama BGE Reranker...")
    try:
        reranker = BGEReranker(
            base_url=config.rerank_base_url,
            model="dengcao/bge-reranker-v2-m3",  # Ollama 实际模型名
        )
        healthy = reranker._health_check()
        checks["reranker"] = {"status": "OK" if healthy else "DEGRADED", "healthy": healthy}
        print(f"  {'[OK]' if healthy else '[WARN]'} Reranker 健康检查: {'通过' if healthy else '失败，将降级为 SimpleReranker'}")
    except Exception as e:
        checks["reranker"] = {"status": "FAIL", "error": str(e)[:100]}
        print(f"  [FAIL] Reranker 检查失败: {str(e)[:80]}")

    # 4. 检索索引
    print("\n[4/5] 检查检索索引...")
    try:
        retriever = InMemoryHybridRetriever(
            lexical_weight=0.35,
            tfidf_weight=0.25,
            embedding_weight=0.4,
        )
        if embed_provider:
            retriever._embedding_provider = embed_provider

        index_dir = config.data_dir / "retrieval_index"
        loaded = False
        if index_dir.exists():
            loaded = retriever.load_index(index_dir)

        chunk_count = len(retriever._chunks)
        if loaded and chunk_count > 0:
            checks["retrieval_index"] = {"status": "OK", "chunk_count": chunk_count}
            print(f"  [OK] 索引已加载，共 {chunk_count} chunks")
        else:
            checks["retrieval_index"] = {"status": "NOT_LOADED", "chunk_count": 0}
            print(f"  [WARN] 索引未加载，chunk_count=0（需要先运行数据导入）")
    except Exception as e:
        checks["retrieval_index"] = {"status": "ERROR", "error": str(e)[:100]}
        print(f"  [FAIL] 检索索引检查失败: {str(e)[:80]}")

    # 5. 整体配置
    print("\n[5/5] 配置汇总...")
    checks["config"] = {
        "model_provider": config.model_provider,
        "model_name": config.model_name,
        "embedding_model": config.embedding_model,
        "rerank_model": config.rerank_model,
        "retrieval_top_k": config.retrieval_top_k,
        "rerank_top_k": config.rerank_top_k,
        "retrieval_fusion_mode": config.retrieval_fusion_mode,
        "lexical_weight": config.retrieval_lexical_weight,
        "tfidf_weight": config.retrieval_tfidf_weight,
        "embedding_weight": config.retrieval_embedding_weight,
    }
    print(f"  模型: {config.model_provider}/{config.model_name}")
    print(f"  Embedding: {config.embedding_model}")
    print(f"  Reranker: {config.rerank_model}")
    print(f"  检索权重: lexical={config.retrieval_lexical_weight}, tfidf={config.retrieval_tfidf_weight}, embedding={config.retrieval_embedding_weight}")

    return checks


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: 路由准确性测试
# ─────────────────────────────────────────────────────────────────────────────

def run_routing_test(config: AgentConfig) -> list[RoutingResult]:
    """测试路由准确性（真实 LLM 调用）"""
    print("\n" + "=" * 60)
    print("Phase 2: 路由准确性测试")
    print("=" * 60)

    try:
        provider = AnthropicCompatibleProvider(
            base_url=config.anthropic_base_url,
            api_key=config.anthropic_api_key,
            model_name=config.model_name,
        )
    except Exception as e:
        print(f"  [FAIL] 无法初始化 MiniMax provider: {e}")
        return []

    router = LLMRouterAgent(model_provider=provider, config={})
    results = []

    for q in COMPLEX_QUESTIONS:
        query = q["query"]
        expected = q.get("expected_route", "SINGLE_STEP")

        # 串行 + 重试 + 间隔（防止 MiniMax 限流）
        last_err = ""
        for attempt in range(3):
            try:
                routing_context = {"available_tools": [], "history": q.get("history")}
                decision = router.route(query, context=routing_context)
                actual = decision.route_type.value.upper()
                is_correct = actual == expected.upper()
                results.append(RoutingResult(
                    query_id=q["id"],
                    query=query,
                    expected_route=expected,
                    actual_route=actual,
                    selected_tools=decision.selected_tools,
                    confidence=decision.confidence,
                    is_correct=is_correct,
                    error=None,
                ))
                status = "[OK]" if is_correct else "[FAIL]"
                print(f"  {status} [{q['id']}] {query[:30]}... → 期望:{expected} 实际:{actual} (conf={decision.confidence:.2f})")
                break  # 成功，跳出重试循环
            except Exception as e:
                last_err = str(e)
                is_api_error = "500" in last_err or "529" in last_err or "status_code" in last_err
                if attempt < 2 and is_api_error:
                    wait = 2 ** (attempt + 1)
                    print(f"  [RETRY] [{q['id']}] API 错误，{wait}s 后重试 ({attempt+1}/3): {last_err[:50]}")
                    time.sleep(wait)
                    continue
                results.append(RoutingResult(
                    query_id=q["id"],
                    query=query,
                    expected_route=expected,
                    actual_route="ERROR",
                    selected_tools=[],
                    confidence=0.0,
                    is_correct=False,
                    error=last_err[:100] if is_api_error else None,
                ))
                if is_api_error:
                    print(f"  [WARN] [{q['id']}] API 错误 (500/529): {last_err[:60]}")
                else:
                    print(f"  [FAIL] [{q['id']}] 异常: {last_err[:60]}")
                break  # 非 API 错误或已达最大重试次数

        time.sleep(1)  # 每次请求间隔 1s，避免突发限流

    agg = aggregate_routing_results(results)
    print(f"\n  路由准确率: {agg['accuracy']*100:.1f}% ({agg['correct_count']}/{agg['valid']} 有效请求)")
    print(f"  API 错误 (500/529): {agg['errors_500_529']}/{agg['total']}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2b: 端到端延迟测试
# ─────────────────────────────────────────────────────────────────────────────

def run_latency_test(config: AgentConfig, retriever: InMemoryHybridRetriever | None = None) -> list[LatencyResult]:
    """测试端到端延迟（真实组件）"""
    print("\n" + "=" * 60)
    print("Phase 2b: 端到端延迟测试")
    print("=" * 60)

    try:
        provider = AnthropicCompatibleProvider(
            base_url=config.anthropic_base_url,
            api_key=config.anthropic_api_key,
            model_name=config.model_name,
        )
    except Exception as e:
        print(f"  [FAIL] 无法初始化 provider: {e}")
        return []

    if retriever is None:
        retriever = InMemoryHybridRetriever(
            lexical_weight=config.retrieval_lexical_weight,
            tfidf_weight=config.retrieval_tfidf_weight,
            embedding_weight=config.retrieval_embedding_weight,
        )
        # 尝试加载索引
        index_dir = config.data_dir / "retrieval_index"
        if index_dir.exists():
            retriever.load_index(index_dir)
            print(f"  索引已加载: {len(retriever._chunks)} chunks")
        else:
            print(f"  [WARN] 索引目录不存在: {index_dir}")

    # 选择 reranker
    if config.rerank_provider == "ollama":
        reranker = BGEReranker(
            base_url=config.rerank_base_url,
            model="dengcao/bge-reranker-v2-m3",  # Ollama 实际模型名
        )
        if not reranker._health_check():
            print("  [WARN] BGE Reranker 健康检查失败，降级为 SimpleReranker")
            reranker = SimpleReranker()
    else:
        reranker = SimpleReranker()

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
    for q in COMPLEX_QUESTIONS:
        query = q["query"]
        try:
            result = rag.execute(query, rewrite_mode="hybrid")
            results.append(LatencyResult(
                query_id=q["id"],
                rewrite_ms=result["latency"]["rewrite_ms"],
                retrieval_ms=result["latency"]["retrieval_ms"],
                rerank_ms=result["latency"]["rerank_ms"],
                total_ms=result["latency"]["total_ms"],
                hits_count=len(result["hits"]),
                rewritten_query=result.get("rewritten_query", ""),
                error=None,
            ))
            lat = result["latency"]
            print(f"  [OK] [{q['id']}] total={lat['total_ms']}ms (rewrite={lat['rewrite_ms']}ms, retrieval={lat['retrieval_ms']}ms, rerank={lat['rerank_ms']}ms)")
        except Exception as e:
            err = str(e)
            results.append(LatencyResult(
                query_id=q["id"], rewrite_ms=0, retrieval_ms=0,
                rerank_ms=0, total_ms=0, hits_count=0,
                rewritten_query="", error=err[:100],
            ))
            print(f"  [FAIL] [{q['id']}] 错误: {err[:60]}")

    agg = aggregate_latency_results(results)
    print(f"\n  P50 端到端延迟: {agg['p50_total_ms']}ms")
    print(f"  P95 端到端延迟: {agg['p95_total_ms']}ms")
    print(f"  P99 端到端延迟: {agg['p99_total_ms']}ms")
    print(f"  P50 检索延迟: {agg['p50_retrieval_ms']}ms")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Query Rewrite Ablation
# ─────────────────────────────────────────────────────────────────────────────

def run_rewrite_ablation(config: AgentConfig, retriever: InMemoryHybridRetriever | None = None) -> list[RewriteAblationResult]:
    """测试各 rewrite 模式有效性"""
    print("\n" + "=" * 60)
    print("Phase 3: Query Rewrite Ablation")
    print("=" * 60)

    try:
        provider = AnthropicCompatibleProvider(
            base_url=config.anthropic_base_url,
            api_key=config.anthropic_api_key,
            model_name=config.model_name,
        )
    except Exception as e:
        print(f"  [FAIL] 无法初始化 provider: {e}")
        return []

    if retriever is None:
        retriever = InMemoryHybridRetriever(
            lexical_weight=config.retrieval_lexical_weight,
            tfidf_weight=config.retrieval_tfidf_weight,
            embedding_weight=config.retrieval_embedding_weight,
        )
        index_dir = config.data_dir / "retrieval_index"
        if index_dir.exists():
            retriever.load_index(index_dir)

    reranker = SimpleReranker()

    # 先跑 rule 模式作为 baseline
    print("  预热 rule baseline...")
    rule_hits: dict[str, list[str]] = {}
    for q in COMPLEX_QUESTIONS:
        try:
            rag = RAGChain(config={}, retriever=retriever, reranker=reranker, model_provider=None)
            result = rag.execute(q["query"], rewrite_mode="rule")
            rule_hits[q["id"]] = [h.chunk_id for h in result["hits"]]
        except Exception:
            rule_hits[q["id"]] = []

    results = []
    for mode in REWRITE_MODES:
        print(f"\n  测试模式: {mode}")
        model = provider if mode != "rule" else None

        for q in COMPLEX_QUESTIONS:
            gold_ids = set(q.get("gold_chunk_ids", []))
            try:
                rag = RAGChain(config={}, retriever=retriever, reranker=reranker, model_provider=model)
                result = rag.execute(q["query"], rewrite_mode=mode)
                hits_ids = [h.chunk_id for h in result["hits"]]

                recall = _recall_at_k(hits_ids, gold_ids, k=3)
                mrr = _mrr(hits_ids, gold_ids)
                ndcg = _ndcg_at_k(hits_ids, gold_ids, k=3)
                overlap = len(set(hits_ids) & set(rule_hits[q["id"]])) / max(len(hits_ids), 1)

                results.append(RewriteAblationResult(
                    query_id=q["id"],
                    mode=mode,
                    recall_at_3=recall,
                    mrr=mrr,
                    ndcg_at_3=ndcg,
                    hit_overlap_with_rule=overlap,
                    rewritten_query=result.get("rewritten_query", ""),
                    error=None,
                ))
                print(f"    [{q['id']}] recall_at_3={recall:.3f} mrr={mrr:.3f} ndcg_at_3={ndcg:.3f}")
            except Exception as e:
                results.append(RewriteAblationResult(
                    query_id=q["id"], mode=mode,
                    recall_at_3=0.0, mrr=0.0, ndcg_at_3=0.0,
                    hit_overlap_with_rule=None,
                    rewritten_query="", error=str(e)[:100],
                ))

    agg = aggregate_rewrite_ablation(results)
    print(f"\n  === Ablation 汇总 ===")
    print(f"  {'模式':<15} {'Recall@3':<10} {'MRR':<8} {'NDCG@3':<8} {'相对rule提升':<12}")
    print(f"  {'-'*55}")
    for mode in REWRITE_MODES:
        if mode in agg:
            s = agg[mode]
            improvement = s.get("recall_improvement", 0)
            imp_str = f"+{improvement:.3f}" if improvement >= 0 else f"{improvement:.3f}"
            print(f"  {mode:<15} {s['mean_recall_at_3']:<10.3f} {s['mean_mrr']:<8.3f} {s['mean_ndcg_at_3']:<8.3f} {imp_str:<12}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: 回答质量测试
# ─────────────────────────────────────────────────────────────────────────────

def run_quality_test(config: AgentConfig, retriever: InMemoryHybridRetriever | None = None) -> list[QualityResult]:
    """测试回答质量（LLM-as-Judge）"""
    print("\n" + "=" * 60)
    print("Phase 4: 回答质量测试 (LLM-as-Judge)")
    print("=" * 60)

    try:
        provider = AnthropicCompatibleProvider(
            base_url=config.anthropic_base_url,
            api_key=config.anthropic_api_key,
            model_name=config.model_name,
        )
    except Exception as e:
        print(f"  [FAIL] 无法初始化 provider: {e}")
        return []

    judge_provider = provider  # 用同一个 provider 做 judge

    if retriever is None:
        retriever = InMemoryHybridRetriever(
            lexical_weight=config.retrieval_lexical_weight,
            tfidf_weight=config.retrieval_tfidf_weight,
            embedding_weight=config.retrieval_embedding_weight,
        )
        index_dir = config.data_dir / "retrieval_index"
        if index_dir.exists():
            retriever.load_index(index_dir)

    reranker = SimpleReranker()

    JUDGE_PROMPT = """你是一个 RAG 系统质量评估专家。
问题: {question}
期望答案关键点: {expected}
生成回答: {generated_answer}
检索到的证据: {evidence_texts}

请从以下 5 个维度对回答评分（1-5 分，5 分最好）：
1. accuracy: 回答的事实是否正确？有无错误数据？
2. completeness: 是否回答了问题的所有部分？
3. relevance: 回答是否专注在问题上？有无偏题？
4. citation_correctness: 有无捏造引用？证据是否支持？
5. hallucination: 有无证据不支持的信息？5=无幻觉，1=大量幻觉

输出 JSON（不要有 markdown 代码块）：
{{"accuracy": 1-5, "completeness": 1-5, "relevance": 1-5, "citation_correctness": 1-5, "hallucination": 1-5, "overall_quality": 1-5, "reasoning": "评分说明"}}
"""

    results = []
    for q in COMPLEX_QUESTIONS:
        query = q["query"]
        gold_keywords = list(q.get("gold_chunk_ids", []))
        try:
            # RAG 检索
            rag = RAGChain(config={}, retriever=retriever, reranker=reranker, model_provider=provider)
            result = rag.execute(query, rewrite_mode="hybrid")
            hits = result["hits"]
            evidence_texts = [h.text for h in hits]
            evidence_str = "\n".join(evidence_texts)[:500]

            # 生成回答
            answer = provider.generate(
                messages=[
                    {"role": "system", "content": "你是一个回答助手，基于证据回答用户问题。"},
                    {"role": "user", "content": f"问题：{query}\n证据：{evidence_str}"}
                ],
                temperature=0.1,
                max_tokens=512,
            )

            # LLM-as-Judge 评分
            judge_prompt = JUDGE_PROMPT.format(
                question=query,
                expected=",".join(gold_keywords),
                generated_answer=answer,
                evidence_texts=evidence_str,
            )
            judge_resp = judge_provider.generate(
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0,
                max_tokens=512,
            )

            # 解析 JSON
            import re as _re
            json_match = _re.search(r"\{[\s\S]*\}", judge_resp)
            if json_match:
                parsed = json.loads(json_match.group())
                results.append(QualityResult(
                    query_id=q["id"],
                    question=query,
                    generated_answer=answer,
                    accuracy=parsed.get("accuracy", 0),
                    completeness=parsed.get("completeness", 0),
                    relevance=parsed.get("relevance", 0),
                    citation_correctness=parsed.get("citation_correctness", 0),
                    hallucination=parsed.get("hallucination", 0),
                    overall_quality=parsed.get("overall_quality", 0),
                    reasoning=parsed.get("reasoning", ""),
                    error=None,
                ))
                print(f"  [OK] [{q['id']}] overall={parsed.get('overall_quality', '?')}/5 acc={parsed.get('accuracy', '?')} comp={parsed.get('completeness', '?')}")
            else:
                results.append(QualityResult(
                    query_id=q["id"], question=query, generated_answer=answer,
                    accuracy=0, completeness=0, relevance=0, citation_correctness=0,
                    hallucination=0, overall_quality=0, reasoning="",
                    error=f"Judge 响应无法解析: {judge_resp[:80]}",
                ))
                print(f"  [FAIL] [{q['id']}] Judge 响应解析失败")

        except Exception as e:
            err = str(e)
            is_api = "500" in err or "529" in err or "status_code" in err
            results.append(QualityResult(
                query_id=q["id"], question=query, generated_answer="",
                accuracy=0, completeness=0, relevance=0, citation_correctness=0,
                hallucination=0, overall_quality=0, reasoning="",
                error=err[:100] if is_api else None,
            ))
            if is_api:
                print(f"  [WARN] [{q['id']}] API 错误 (500/529)")
            else:
                print(f"  [FAIL] [{q['id']}] 异常: {err[:60]}")

    agg = aggregate_quality_results(results)
    print(f"\n  === 质量评分汇总 ===")
    dims = ["accuracy", "completeness", "relevance", "citation_correctness", "hallucination", "overall_quality"]
    print(f"  {'维度':<20} {'均值':<8} {'最低':<8} {'最高':<8}")
    print(f"  {'-'*50}")
    for dim in dims:
        print(f"  {dim:<20} {agg[f'mean_{dim}']:<8.3f} {agg[f'min_{dim}']:<8.3f} {agg[f'max_{dim}']:<8.3f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="路由与 RAG 系统严格评估")
    parser.add_argument("--env-check", action="store_true", help="仅环境验证")
    parser.add_argument("--routing-only", action="store_true", help="仅路由测试")
    parser.add_argument("--latency-only", action="store_true", help="仅延迟测试")
    parser.add_argument("--rewrite-ablation", action="store_true", help="仅 Rewrite Ablation")
    parser.add_argument("--quality-only", action="store_true", help="仅质量测试")
    parser.add_argument("--all", action="store_true", help="全量测试")
    parser.add_argument("--output", default="data/eval/rigorous_results.json", help="输出路径")
    parser.add_argument("--routing-model", default="minimax", help="路由测试使用的模型: minimax/ollama")
    args = parser.parse_args()

    config = AgentConfig.from_env()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "questions": [{"id": q["id"], "query": q["query"], "category": q["category"]} for q in COMPLEX_QUESTIONS],
        "rewrite_modes": REWRITE_MODES,
    }

    # Phase 1: 环境验证
    env_results = phase1_env_check()
    all_results["env_check"] = env_results

    if args.env_check:
        output_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已保存至 {output_path}")
        return

    # 构建 retriever（共享）
    retriever: InMemoryHybridRetriever | None = None
    try:
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
            print(f"\n共享 retriever 已加载: {len(retriever._chunks)} chunks")
    except Exception as e:
        print(f"\n[WARN] Retriever 初始化失败: {e}")

    # 执行测试
    if args.all or (not args.routing_only and not args.latency_only and not args.rewrite_ablation and not args.quality_only and not args.env_check):
        # 默认全量
        all_results["routing"] = aggregate_routing_results(run_routing_test(config))
        all_results["latency"] = aggregate_latency_results(run_latency_test(config, retriever))
        all_results["rewrite_ablation"] = aggregate_rewrite_ablation(config, retriever)
        all_results["quality"] = aggregate_quality_results(config, retriever)
    else:
        if args.routing_only:
            all_results["routing"] = aggregate_routing_results(run_routing_test(config))
        if args.latency_only:
            all_results["latency"] = aggregate_latency_results(run_latency_test(config, retriever))
        if args.rewrite_ablation:
            all_results["rewrite_ablation"] = aggregate_rewrite_ablation(run_rewrite_ablation(config, retriever))
        if args.quality_only:
            all_results["quality"] = aggregate_quality_results(config, retriever)

    output_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"结果已保存至 {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
