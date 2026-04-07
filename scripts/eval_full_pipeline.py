#!/usr/bin/env python3
"""
scripts/eval_full_pipeline.py

完整评估框架：运行三个维度的评估并生成汇总报告。

评估维度：
1. RAG 检索有效性：Recall@K / HitRate@K / MRR
2. RAG 生成质量：拒答率 / 锚点覆盖率 / 幻觉率
3. 多 Agent 编排有效性：Coordinator vs SimpleAgent（速度、准确率、步骤数）

用法：
    python scripts/eval_full_pipeline.py
    python scripts/eval_full_pipeline.py --skip-retrieval
    python scripts/eval_full_pipeline.py --skip-generation
    python scripts/eval_full_pipeline.py --skip-coordinator
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from agent.config import AgentConfig
from agent.core.evaluation import (
    GroundednessEvaluator,
    RelevanceEvaluator,
    load_retrieval_test_set,
    recall_at_k,
    hit_rate_at_k,
    mean_reciprocal_rank,
)
from agent.core.retrieval import InMemoryHybridRetriever
from agent.llm.embeddings import build_embedding_provider
from agent.llm.providers import build_model_provider
from agent.application.rag_agent_service import RagAgentService
from agent.application.agent_service import AgentService


# ── helpers ────────────────────────────────────────────────────────────────
def ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load_raw_docs(data_raw: Path):
    docs = []
    for company_dir in sorted(data_raw.iterdir()):
        if not company_dir.is_dir():
            continue
        for ext in ("*.md", "*.txt"):
            for p in sorted(company_dir.glob(ext)):
                docs.append((p.stem, f"{company_dir.name}/{p.name}", p.read_text(encoding="utf-8")))
    return docs


# ══════════════════════════════════════════════════════════════════════════════
# 维度一：检索有效性
# ══════════════════════════════════════════════════════════════════════════════
def run_retrieval_eval(config: AgentConfig) -> dict[str, Any]:
    print(f"\n{'='*60}")
    print("维度一：RAG 检索有效性评估")
    print(f"{'='*60}")

    embedding_provider = build_embedding_provider(config)
    actual_weight = config.retrieval_embedding_weight
    embedding_healthy = False
    if embedding_provider:
        try:
            test = embedding_provider.embed_texts(["健康检查"])
            if not test or not test[0] or all(v == 0.0 for v in test[0]):
                raise ValueError("全零向量")
            embedding_healthy = True
        except Exception as exc:
            print(f"  [WARN] Embedding 不可用，降级为纯词频检索: {exc}")
            embedding_provider = None
            actual_weight = 0.0
    else:
        print(f"  [WARN] Embedding provider 未配置，使用纯词频检索")

    retriever = InMemoryHybridRetriever(
        embedding_provider=embedding_provider,
        fusion_mode=config.retrieval_fusion_mode,
        lexical_weight=config.retrieval_lexical_weight,
        tfidf_weight=config.retrieval_tfidf_weight,
        embedding_weight=actual_weight,
        embedding_top_k=config.embedding_top_k,
    )

    index_dir = config.data_dir / "retrieval_index"
    retriever.load_index(index_dir)

    test_set_path = ROOT / "data" / "eval" / "retrieval_test_set.json"
    test_records = load_retrieval_test_set(test_set_path)

    if not test_records:
        print("[WARN] 检索评估集为空")
        return {}

    k = config.retrieval_top_k
    records_out = []
    recall_scores, hit_scores, mrr_scores = [], [], []

    for rec in test_records:
        hits, _ = retriever.search_with_debug(rec.query, top_k=config.retrieval_top_k)
        hit_texts = [hit.text for hit in hits]
        rk = recall_at_k(hit_texts, rec.expected_answers, k)
        hr = hit_rate_at_k(hit_texts, rec.expected_answers, k)
        mrr_val = mean_reciprocal_rank(hit_texts, rec.expected_answers)
        recall_scores.append(rk)
        hit_scores.append(hr)
        mrr_scores.append(mrr_val)
        records_out.append({
            "query": rec.query,
            "recall@k": round(rk, 4),
            "hit_rate@k": round(hr, 4),
            "mrr": round(mrr_val, 4),
            "hits_count": len(hits),
        })
        print(f"  recall@{k}={rk:.2f}  hit@{k}={hr:.2f}  mrr={mrr_val:.2f}  | {rec.query[:50]}")

    n = len(records_out)
    summary = {
        f"recall@{k}": round(sum(recall_scores) / n, 4) if n else 0.0,
        f"hit_rate@{k}": round(sum(hit_scores) / n, 4) if n else 0.0,
        "mrr": round(sum(mrr_scores) / n, 4) if n else 0.0,
        "total": n,
        "embedding_enabled": embedding_provider is not None,
        "embedding_model": config.embedding_model,
        "fusion_mode": config.retrieval_fusion_mode,
        "weights": {
            "lexical": config.retrieval_lexical_weight,
            "tfidf": config.retrieval_tfidf_weight,
            "embedding": actual_weight,
        },
    }

    print(f"\n  汇总（{n} 条）:")
    print(f"    Recall@{k}:  {summary[f'recall@{k}']:.4f}")
    print(f"    HitRate@{k}: {summary[f'hit_rate@{k}']:.4f}")
    print(f"    MRR:         {summary['mrr']:.4f}")
    print(f"    Embedding:   {'启用' if embedding_healthy else '禁用（降级为纯词频）'} ({config.embedding_model})")

    return {
        "summary": summary,
        "records": records_out,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 维度二：RAG 生成质量
# ══════════════════════════════════════════════════════════════════════════════
def run_generation_eval(config: AgentConfig) -> dict[str, Any]:
    print(f"\n{'='*60}")
    print("维度二：RAG 生成质量评估")
    print(f"{'='*60}")

    # 强制禁用 cascade reranker（避免 HuggingFace 模型下载超时），使用简单重排
    config.rerank_cascade = False

    model = build_model_provider(config)
    rag = RagAgentService(config=config, model=model)

    # 摄入文档
    data_raw = ROOT / "data" / "raw" / "finance"
    if data_raw.exists():
        for doc_id, source, content in _load_raw_docs(data_raw):
            rag.ingest_document(doc_id=doc_id, source=source, content=content)

    eval_set_path = ROOT / "data" / "eval" / "week1_eval_set.jsonl"
    eval_items = []
    if eval_set_path.exists():
        for line in eval_set_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                eval_items.append(json.loads(line))

    if not eval_items:
        print("[WARN] 生成评估集为空")
        return {}

    groundedness_eval = GroundednessEvaluator(model)
    relevance_eval = RelevanceEvaluator(model)

    rows = []
    refusal_count = 0
    groundedness_scores = []
    relevance_scores = []

    for item in eval_items:
        q = str(item["question"])
        expected = str(item.get("expected_answer", ""))
        result = rag.answer(q, append_to_eval_store=False)
        answer = str(result.get("answer", ""))
        hits = list(result.get("retrieval_hits", []))
        refusal = bool(result.get("refusal", False))
        reason = str(result.get("reason", ""))

        if refusal:
            refusal_count += 1
            groundedness_scores.append(0.0)
            relevance_scores.append(0.0)
        else:
            evidence = [hit.get("text_preview", "") or hit.get("text", "") for hit in hits]
            g_score = groundedness_eval.evaluate(answer, evidence)
            r_score = relevance_eval.evaluate(q, answer)
            groundedness_scores.append(g_score)
            relevance_scores.append(r_score)

        rows.append({
            "question": q,
            "expected_answer": expected,
            "answer_preview": answer[:200],
            "refusal": refusal,
            "reason": reason,
            "retrieval_hit_count": len(hits),
            "groundedness": round(groundedness_scores[-1], 4),
            "relevance": round(relevance_scores[-1], 4),
        })
        print(f"  {'[拒答]' if refusal else '[  OK ]'}  g={groundedness_scores[-1]:.2f}  r={relevance_scores[-1]:.2f}  | {q[:45]}")

    n = len(rows)
    summary = {
        "total": n,
        "refusal_rate": round(refusal_count / n, 4) if n else 0.0,
        "groundedness_avg": round(sum(groundedness_scores) / n, 4) if n else 0.0,
        "relevance_avg": round(sum(relevance_scores) / n, 4) if n else 0.0,
        "refusal_breakdown": _refusal_breakdown(rows),
    }

    print(f"\n  汇总（{n} 条）:")
    print(f"    拒答率:        {summary['refusal_rate']:.2%}")
    print(f"    Groundedness: {summary['groundedness_avg']:.4f}")
    print(f"    Relevance:    {summary['relevance_avg']:.4f}")

    return {
        "summary": summary,
        "rows": rows,
    }


def _refusal_breakdown(rows: list[dict]) -> dict[str, int]:
    reasons: dict[str, int] = {}
    for r in rows:
        if r["refusal"]:
            reason = r.get("reason", "unknown") or "unknown"
            reasons[reason] = reasons.get(reason, 0) + 1
    return reasons


# ══════════════════════════════════════════════════════════════════════════════
# 维度三：多 Agent 编排有效性（Coordinator vs SimpleAgent）
# ══════════════════════════════════════════════════════════════════════════════
def run_coordinator_eval(config: AgentConfig) -> dict[str, Any]:
    print(f"\n{'='*60}")
    print("维度三：多 Agent 编排有效性评估")
    print(f"{'='*60}")

    # 强制开启 Coordinator 用于对比
    config.use_coordinator = True

    model = build_model_provider(config)
    rag = RagAgentService(config=config, model=model)

    # 摄入文档
    data_raw = ROOT / "data" / "raw" / "finance"
    if data_raw.exists():
        for doc_id, source, content in _load_raw_docs(data_raw):
            rag.ingest_document(doc_id=doc_id, source=source, content=content)

    eval_set_path = ROOT / "data" / "eval" / "week1_eval_set.jsonl"
    eval_items = []
    if eval_set_path.exists():
        for line in eval_set_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                eval_items.append(json.loads(line))

    if not eval_items:
        print("[WARN] 评估集为空，跳过 Coordinator 评估")
        return {}

    # 测试 Coordinator 模式
    agent = AgentService(config=config)
    coord_rows = []
    simple_rows = []

    # 简单问题（不需要 Coordinator）
    simple_questions = [item for item in eval_items[:5]]
    # 复杂问题（适合 Coordinator）
    complex_questions = [item for item in eval_items[5:15]] if len(eval_items) > 5 else eval_items[5:]

    print("\n  [Coordinator 模式]")
    for item in complex_questions:
        q = str(item["question"])
        expected = str(item.get("expected_answer", ""))
        start = time.perf_counter()
        result = agent.chat(session_id="eval", user_message=q)
        latency_ms = int((time.perf_counter() - start) * 1000)
        answer = str(result.get("answer", ""))
        steps = int(result.get("steps_used", 0))

        # 计算准确率（子串匹配）
        match = expected in answer if expected else False
        coord_rows.append({
            "question": q,
            "answer_preview": answer[:150],
            "expected": expected,
            "substring_match": match,
            "latency_ms": latency_ms,
            "steps_used": steps,
        })
        print(f"  [{'✓' if match else '✗'}]  {latency_ms}ms  {steps}步  | {q[:45]}")

    # 对比：相同问题用 SimpleAgent
    print("\n  [SimpleAgent 模式 — 对比]")
    config.use_coordinator = False
    agent_simple = AgentService(config=config)
    for item in complex_questions:
        q = str(item["question"])
        expected = str(item.get("expected_answer", ""))
        start = time.perf_counter()
        result = agent_simple.chat(session_id="eval", user_message=q)
        latency_ms = int((time.perf_counter() - start) * 1000)
        answer = str(result.get("answer", ""))
        steps = int(result.get("steps_used", 0))
        match = expected in answer if expected else False
        simple_rows.append({
            "question": q,
            "answer_preview": answer[:150],
            "expected": expected,
            "substring_match": match,
            "latency_ms": latency_ms,
            "steps_used": steps,
        })
        print(f"  [{'✓' if match else '✗'}]  {latency_ms}ms  {steps}步  | {q[:45]}")

    n = len(coord_rows)
    coord_acc = sum(1 for r in coord_rows if r["substring_match"]) / n if n else 0.0
    simple_acc = sum(1 for r in simple_rows if r["substring_match"]) / n if n else 0.0
    coord_avg_latency = sum(r["latency_ms"] for r in coord_rows) / n if n else 0
    simple_avg_latency = sum(r["latency_ms"] for r in simple_rows) / n if n else 0
    coord_avg_steps = sum(r["steps_used"] for r in coord_rows) / n if n else 0.0
    simple_avg_steps = sum(r["steps_used"] for r in simple_rows) / n if n else 0.0

    summary = {
        "total_questions": n,
        "coordinator": {
            "accuracy": round(coord_acc, 4),
            "avg_latency_ms": round(coord_avg_latency, 2),
            "avg_steps": round(coord_avg_steps, 2),
        },
        "simple_agent": {
            "accuracy": round(simple_acc, 4),
            "avg_latency_ms": round(simple_avg_latency, 2),
            "avg_steps": round(simple_avg_steps, 2),
        },
        "improvement": {
            "accuracy_delta": round(coord_acc - simple_acc, 4),
            "latency_ratio": round(coord_avg_latency / max(simple_avg_latency, 1), 2),
            "steps_ratio": round(coord_avg_steps / max(simple_avg_steps, 0.01), 2),
        },
    }

    print(f"\n  汇总（{n} 条复杂问题）:")
    print(f"    Coordinator:  准确率 {summary['coordinator']['accuracy']:.2%}  "
          f"延迟 {summary['coordinator']['avg_latency_ms']:.0f}ms  "
          f"平均 {summary['coordinator']['avg_steps']:.1f} 步")
    print(f"    SimpleAgent:  准确率 {summary['simple_agent']['accuracy']:.2%}  "
          f"延迟 {summary['simple_agent']['avg_latency_ms']:.0f}ms  "
          f"平均 {summary['simple_agent']['avg_steps']:.1f} 步")
    print(f"    提升:         准确率 {summary['improvement']['accuracy_delta']:+.2%}  "
          f"延迟比 {summary['improvement']['latency_ratio']:.2f}x  "
          f"步数比 {summary['improvement']['steps_ratio']:.2f}x")

    return {
        "summary": summary,
        "coordinator_rows": coord_rows,
        "simple_rows": simple_rows,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="完整评估框架")
    parser.add_argument("--skip-retrieval", action="store_true", help="跳过检索评估")
    parser.add_argument("--skip-generation", action="store_true", help="跳过生成质量评估")
    parser.add_argument("--skip-coordinator", action="store_true", help="跳过多 Agent 编排评估")
    parser.add_argument("--output", type=Path, default=ROOT / "runtime" / "eval_full",
                        help="输出目录")
    args = parser.parse_args()

    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    config = AgentConfig.from_env()

    results: dict[str, Any] = {
        "generated_at": ts(),
        "embedding_model": config.embedding_model,
        "model": config.model_name,
    }

    # 维度一
    if not args.skip_retrieval:
        results["retrieval"] = run_retrieval_eval(config)
    else:
        print("\n[跳过] 检索有效性评估")

    # 维度二
    if not args.skip_generation:
        results["generation"] = run_generation_eval(config)
    else:
        print("\n[跳过] RAG 生成质量评估")

    # 维度三
    if not args.skip_coordinator:
        results["coordinator"] = run_coordinator_eval(config)
    else:
        print("\n[跳过] 多 Agent 编排评估")

    # 保存结果
    full_path = out_dir / "full_evaluation.json"
    with full_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n完整报告: {full_path}")

    # 打印摘要
    _print_summary(results)


def _print_summary(results: dict[str, Any]) -> None:
    print(f"\n{'#'*60}")
    print("# 评估摘要")
    print(f"{'#'*60}")
    if "retrieval" in results and results["retrieval"]:
        r = results["retrieval"]["summary"]
        # Dynamically find recall@K and hit_rate@K since K comes from config
        recall_key = next((k for k in r if k.startswith("recall@")), None)
        hit_key = next((k for k in r if k.startswith("hit_rate@")), None)
        recall_val = r.get(recall_key, "N/A")
        hit_val = r.get(hit_key, "N/A")
        mrr_val = r.get("mrr", "N/A")
        k_suffix = recall_key.replace("recall@", "") if recall_key else "?"
        print(f"  检索 Recall@{k_suffix}:  {recall_val:.4f}" if isinstance(recall_val, float) else f"  检索 Recall@{k_suffix}:  {recall_val}")
        print(f"  检索 HitRate@{k_suffix}: {hit_val:.4f}" if isinstance(hit_val, float) else f"  检索 HitRate@{k_suffix}: {hit_val}")
        print(f"  检索 MRR:       {mrr_val:.4f}" if isinstance(mrr_val, float) else f"  检索 MRR:       {mrr_val}")
    if "generation" in results and results["generation"]:
        g = results["generation"]["summary"]
        print(f"  生成 Groundedness: {g.get('groundedness_avg', 'N/A'):.4f}")
        print(f"  生成 Relevance:    {g.get('relevance_avg', 'N/A'):.4f}")
        print(f"  生成 拒答率:       {g.get('refusal_rate', 'N/A'):.2%}")
    if "coordinator" in results and results["coordinator"]:
        c = results["coordinator"]["summary"]
        print(f"  Coordinator 准确率: {c['coordinator']['accuracy']:.2%}")
        print(f"  SimpleAgent 准确率: {c['simple_agent']['accuracy']:.2%}")
        print(f"  编排提升:            {c['improvement']['accuracy_delta']:+.2%}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
