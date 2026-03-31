#!/usr/bin/env python3
"""
scripts/eval_retrieval.py

批量运行检索评估：遍历 retrieval_test_set.json，
对每条 query 调用 RAG pipeline，评估 Recall@K / HitRate@K / MRR，
输出汇总表格。

用法：
    python scripts/eval_retrieval.py [--k 3] [--output data/eval/eval_records/retrieval_results.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# 确保 src 在 path 中
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.agent.config import AgentConfig
from src.agent.core.evaluation import (
    RetrievalEvalRecord,
    load_retrieval_test_set,
    recall_at_k,
    hit_rate_at_k,
    mean_reciprocal_rank,
)
from src.agent.core.retrieval import InMemoryHybridRetriever
from src.agent.llm.embeddings import build_embedding_provider
from src.agent.llm.providers import build_model_provider


def run_retrieval_eval(
    test_set_path: Path,
    output_path: Path | None,
    k: int = 3,
) -> dict[str, float]:
    config = AgentConfig.from_env()

    # 构建检索器
    embedding_provider = build_embedding_provider(config)
    actual_weight = config.retrieval_embedding_weight
    if embedding_provider:
        try:
            test = embedding_provider.embed_texts(["健康检查"])
            if not test or not test[0] or all(v == 0.0 for v in test[0]):
                raise ValueError("全零向量")
        except Exception:
            embedding_provider = None
            actual_weight = 0.0

    retriever = InMemoryHybridRetriever(
        embedding_provider=embedding_provider,
        fusion_mode=config.retrieval_fusion_mode,
        lexical_weight=config.retrieval_lexical_weight,
        tfidf_weight=config.retrieval_tfidf_weight,
        embedding_weight=actual_weight,
        embedding_top_k=config.embedding_top_k,
    )

    # 尝试加载已有索引
    index_dir = config.data_dir / "retrieval_index"
    retriever.load_index(index_dir)

    test_records = load_retrieval_test_set(test_set_path)
    if not test_records:
        print(f"[WARN] 评估集为空或不存在: {test_set_path}")
        return {}

    records_out: list[dict] = []
    recall_scores, hit_rate_scores, mrr_scores = [], [], []

    for rec in test_records:
        hits, _ = retriever.search_with_debug(rec.query, top_k=config.retrieval_top_k)
        hit_texts = [hit.text for hit in hits]

        rk = recall_at_k(hit_texts, rec.expected_answers, k)
        hr = hit_rate_at_k(hit_texts, rec.expected_answers, k)
        mrr = mean_reciprocal_rank(hit_texts, rec.expected_answers)

        recall_scores.append(rk)
        hit_rate_scores.append(hr)
        mrr_scores.append(mrr)

        row = {
            "query": rec.query,
            "expected_answers": rec.expected_answers,
            "recall@k": round(rk, 4),
            "hit_rate@k": round(hr, 4),
            "mrr": round(mrr, 4),
            "hits_count": len(hits),
        }
        records_out.append(row)
        print(f"[{len(records_out):02d}] recall@{k}={rk:.2f}  hit@{k}={hr:.2f}  mrr={mrr:.2f}  | {rec.query[:60]}")

    # 汇总
    n = len(records_out)
    summary = {
        f"recall@{k}": round(sum(recall_scores) / n, 4) if n else 0.0,
        f"hit_rate@{k}": round(sum(hit_rate_scores) / n, 4) if n else 0.0,
        "mrr": round(sum(mrr_scores) / n, 4) if n else 0.0,
        "total": n,
        "timestamp": datetime.utcnow().isoformat(),
    }

    print(f"\n{'='*60}")
    print(f"检索评估汇总（共 {n} 条）")
    print(f"{'='*60}")
    print(f"  Recall@{k}:  {summary[f'recall@{k}']:.4f}")
    print(f"  HitRate@{k}: {summary[f'hit_rate@{k}']:.4f}")
    print(f"  MRR:         {summary['mrr']:.4f}")
    print(f"{'='*60}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for row in records_out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\n结果已写入: {output_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量检索评估")
    parser.add_argument(
        "--test-set",
        type=Path,
        default=Path("data/eval/retrieval_test_set.json"),
        help="评估集路径（默认 data/eval/retrieval_test_set.json）",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Recall/HitRate 的 K 值（默认 3）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/eval_records/retrieval_results.jsonl"),
        help="结果输出路径",
    )
    args = parser.parse_args()

    run_retrieval_eval(args.test_set, args.output, args.k)
