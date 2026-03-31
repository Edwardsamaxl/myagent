#!/usr/bin/env python3
"""
scripts/run_ablation.py

Ablation 实验：对 RAG pipeline 的不同配置做横向对比。

实验组：
    baseline:      EMBEDDING_ENABLED=true,  RERANK_ENABLED=true
    no-embedding:  EMBEDDING_ENABLED=false, RERANK_ENABLED=true
    no-rerank:     EMBEDDING_ENABLED=true,  RERANK_ENABLED=false
    full-off:      EMBEDDING_ENABLED=false, RERANK_ENABLED=false

用法：
    python scripts/run_ablation.py [--test-set data/eval/retrieval_test_set.json] [--k 3]
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.agent.config import AgentConfig
from src.agent.core.evaluation import (
    load_retrieval_test_set,
    recall_at_k,
    hit_rate_at_k,
    mean_reciprocal_rank,
)
from src.agent.core.retrieval import InMemoryHybridRetriever
from src.agent.llm.embeddings import build_embedding_provider


EXPERIMENTS = [
    {
        "name": "baseline",
        "embedding_enabled": True,
        "rerank_enabled": True,
        "label": "完整配置 (embedding+rerank)",
    },
    {
        "name": "no-embedding",
        "embedding_enabled": False,
        "rerank_enabled": True,
        "label": "无 embedding (纯词频+rerank)",
    },
    {
        "name": "no-rerank",
        "embedding_enabled": True,
        "rerank_enabled": False,
        "label": "无 rerank (embedding+词频，不重排)",
    },
    {
        "name": "full-off",
        "embedding_enabled": False,
        "rerank_enabled": False,
        "label": "全关 (纯词频，不重排)",
    },
]


def run_experiment(
    exp: dict,
    test_set_path: Path,
    k: int,
) -> dict[str, float]:
    """用指定实验配置跑检索评估。"""
    # 用环境变量模拟配置（不影响全局 config）
    os.environ["EMBEDDING_ENABLED"] = "true" if exp["embedding_enabled"] else "false"
    os.environ["RERANK_ENABLED"] = "true" if exp["rerank_enabled"] else "false"

    config = AgentConfig.from_env()
    embedding_provider = build_embedding_provider(config)
    actual_weight = config.retrieval_embedding_weight

    if not exp["embedding_enabled"]:
        embedding_provider = None
        actual_weight = 0.0
    elif embedding_provider:
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

    index_dir = config.data_dir / "retrieval_index"
    retriever.load_index(index_dir)

    test_records = load_retrieval_test_set(test_set_path)
    if not test_records:
        return {}

    recall_scores, hit_rate_scores, mrr_scores = [], [], []

    for rec in test_records:
        hits, _ = retriever.search_with_debug(rec.query, top_k=config.retrieval_top_k)
        hit_texts = [hit.text for hit in hits]
        recall_scores.append(recall_at_k(hit_texts, rec.expected_answers, k))
        hit_rate_scores.append(hit_rate_at_k(hit_texts, rec.expected_answers, k))
        mrr_scores.append(mean_reciprocal_rank(hit_texts, rec.expected_answers))

    n = len(test_records)
    return {
        f"recall@{k}": round(sum(recall_scores) / n, 4) if n else 0.0,
        f"hit_rate@{k}": round(sum(hit_rate_scores) / n, 4) if n else 0.0,
        "mrr": round(sum(mrr_scores) / n, 4) if n else 0.0,
        "total": n,
    }


def main(test_set_path: Path, k: int) -> None:
    test_records = load_retrieval_test_set(test_set_path)
    if not test_records:
        print(f"[ERROR] 评估集为空: {test_set_path}")
        return

    results: list[dict] = []
    for exp in EXPERIMENTS:
        print(f"\n>>> 实验: {exp['name']} ({exp['label']})")
        metrics = run_experiment(exp, test_set_path, k)
        if not metrics:
            print(f"  [SKIP] 无数据")
            continue
        print(f"  Recall@{k}: {metrics[f'recall@{k}']:.4f}")
        print(f"  HitRate@{k}: {metrics[f'hit_rate@{k}']:.4f}")
        print(f"  MRR: {metrics['mrr']:.4f}")
        results.append({**exp, **metrics})

    # 打印汇总表格
    print(f"\n{'='*70}")
    print(f"Ablation 汇总（K={k}，共 {len(test_records)} 条）")
    print(f"{'='*70}")
    header = f"{'实验名':<20} {'Recall@K':>10} {'HitRate@K':>12} {'MRR':>8} {'说明'}"
    print(header)
    print("-" * 70)
    for r in results:
        print(
            f"{r['name']:<20} {r[f'recall@{k}']:>10.4f} "
            f"{r[f'hit_rate@{k}']:>12.4f} {r['mrr']:>8.4f}  {r['label']}"
        )
    print(f"{'='*70}")

    # 写入文件
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"data/eval/eval_records/ablation_{timestamp}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Ablation 实验")
    parser.add_argument(
        "--test-set",
        type=Path,
        default=Path("data/eval/retrieval_test_set.json"),
    )
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    main(args.test_set, args.k)
