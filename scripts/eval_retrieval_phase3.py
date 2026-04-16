#!/usr/bin/env python3
"""
Phase 3.2: 检索有效性测试（基于实际数据重新设计）

验证检索结果与问题的相关性，评估 Top-K 召回率。

注意：测试问题基于实际索引数据设计，确保 chunk 中包含相关内容。

用法:
    python scripts/eval_retrieval_phase3.py
"""

from __future__ import annotations

import codecs
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "backslashreplace")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "backslashreplace")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from agent.config import AgentConfig
from agent.llm.embeddings import build_embedding_provider
from agent.rag.retrieval import InMemoryHybridRetriever


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3.2 测试用例（基于实际数据设计）
# ─────────────────────────────────────────────────────────────────────────────
RETRIEVAL_TEST_CASES = [
    # 茅台相关（茅台数据丰富）
    {"id": "R1", "query": "茅台是哪家公司？", "gold_source_pattern": "茅台", "reason": "公司介绍"},
    {"id": "R2", "query": "茅台2024年的营收是多少？", "gold_source_pattern": "茅台", "reason": "2024年报有营收数据"},
    {"id": "R3", "query": "茅台过去3年的营收增长情况", "gold_source_pattern": "茅台", "reason": "多年营收数据"},
    {"id": "R4", "query": "茅台的主营业务是什么", "gold_source_pattern": "茅台", "reason": "公司主营业务介绍"},
    {"id": "R5", "query": "茅台的每股收益多少", "gold_source_pattern": "茅台", "reason": "财务指标"},
    {"id": "R6", "query": "茅台的现金流情况如何", "gold_source_pattern": "茅台", "reason": "现金流量表"},
    {"id": "R7", "query": "茅台的负债情况", "gold_source_pattern": "茅台", "reason": "资产负债表"},
    {"id": "R8", "query": "贵州茅台的行业地位", "gold_source_pattern": "茅台", "reason": "行业分析"},
    {"id": "R9", "query": "茅台和五粮液的对比", "gold_source_pattern": "茅台", "reason": "可同时召回两者"},
    {"id": "R10", "query": "五粮液的主营业务是什么", "gold_source_pattern": "五粮液", "reason": "五粮液公司介绍"},
    # 五粮液相关（五粮液2025H有数据）
    {"id": "R11", "query": "五粮液2025年上半年营收情况", "gold_source_pattern": "五粮液", "reason": "2025半年度报告"},
    {"id": "R12", "query": "五粮液的利润增长", "gold_source_pattern": "五粮液", "reason": "利润数据"},
    {"id": "R13", "query": "五粮液的白酒产品", "gold_source_pattern": "五粮液", "reason": "产品介绍"},
    {"id": "R14", "query": "五粮液和茅台谁更赚钱", "gold_source_pattern": "五粮液", "reason": "两者对比"},
    # 行业相关
    {"id": "R15", "query": "白酒行业的发展情况", "gold_source_pattern": "茅台", "reason": "行业分析在茅台报告中"},
    {"id": "R16", "query": "白酒行业竞争格局", "gold_source_pattern": "茅台", "reason": "行业竞争"},
    # 抽象概念（索引中可能没有）
    {"id": "R17", "query": "茅台的护城河是什么", "gold_source_pattern": "茅台", "reason": "护城河分析"},
    {"id": "R18", "query": "茅台的股价现在多少", "gold_source_pattern": "无", "reason": "无股价数据-预期低召回"},
    {"id": "R19", "query": "五粮液的股价历史走势", "gold_source_pattern": "无", "reason": "无股价数据-预期低召回"},
    {"id": "R20", "query": "茅台2024年的分红方案", "gold_source_pattern": "茅台", "reason": "分红相关"},
]


# ─────────────────────────────────────────────────────────────────────────────
# 数据类
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    query_id: str
    query: str
    gold_source_pattern: str
    top_sources: list[str]
    hits_count: int
    # 命中评估
    hit: float  # 0 or 1, 是否在top5有命中
    mrr: float
    error: str | None


# ─────────────────────────────────────────────────────────────────────────────
# 指标计算
# ─────────────────────────────────────────────────────────────────────────────

def calculate_hit(sources: list[str], gold_pattern: str) -> float:
    """计算 Hit@5：查询是否在 top5 中有至少一个命中（二值）"""
    if gold_pattern == "无":
        return 0.0  # 预期无数据
    for s in sources[:5]:
        if gold_pattern in s:
            return 1.0
    return 0.0


def calculate_mrr(sources: list[str], gold_pattern: str) -> float:
    """计算 MRR：首个命中结果的位置倒数"""
    if gold_pattern == "无":
        return 0.0
    for i, s in enumerate(sources, 1):
        if gold_pattern in s:
            return 1.0 / i
    return 0.0


def aggregate_results(results: list[RetrievalResult]) -> dict:
    valid = [r for r in results if r.error is None]

    # 按 gold_pattern 分组
    with_data = [r for r in valid if r.gold_source_pattern != "无"]
    without_data = [r for r in valid if r.gold_source_pattern == "无"]

    def stats(res_list):
        if not res_list:
            return {"hit": 0, "mrr": 0, "count": 0}
        hits = [r.hit for r in res_list]
        mrrs = [r.mrr for r in res_list]
        return {
            "hit": round(sum(hits) / len(hits), 4),
            "mrr": round(sum(mrrs) / len(mrrs), 4),
            "count": len(res_list),
        }

    return {
        "total": len(valid),
        "with_data": stats(with_data),
        "without_data": stats(without_data),
        "overall": stats(valid),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 主测试逻辑
# ─────────────────────────────────────────────────────────────────────────────

def run_retrieval_test(config: AgentConfig, top_k: int = 5) -> list[RetrievalResult]:
    print("\n" + "=" * 60)
    print("Phase 3.2: 检索有效性测试（基于实际数据）")
    print("=" * 60)

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
        print(f"  [FAIL] 索引目录不存在: {index_dir}")
        return []

    results = []
    for tc in RETRIEVAL_TEST_CASES:
        query = tc["query"]
        gold = tc["gold_source_pattern"]
        print(f"\n  [{tc['id']}] {query[:40]}...")

        try:
            hits, _ = retriever.search_with_debug(query, top_k=top_k)

            sources = [h.source for h in hits]
            hit = calculate_hit(sources, gold)
            mrr = calculate_mrr(sources, gold)

            results.append(RetrievalResult(
                query_id=tc["id"],
                query=query,
                gold_source_pattern=gold,
                top_sources=sources,
                hits_count=len(hits),
                hit=hit,
                mrr=mrr,
                error=None,
            ))

            status = "✓" if hit > 0 else "✗"
            print(f"      {status} Hit={hit:.0f} MRR={mrr:.3f} | {sources[:3]}")

        except Exception as e:
            err = str(e)
            results.append(RetrievalResult(
                query_id=tc["id"],
                query=query,
                gold_source_pattern=gold,
                top_sources=[],
                hits_count=0,
                hit=0.0,
                mrr=0.0,
                error=err[:100],
            ))
            print(f"      [FAIL] {err[:60]}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 输出汇总报告
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: list[RetrievalResult], agg: dict) -> None:
    print("\n" + "=" * 60)
    print("Phase 3.2 汇总报告")
    print("=" * 60)

    print(f"\n【总体统计】")
    print(f"  总测试: {agg['overall']['count']} 条")

    print(f"\n【有数据的问题 (n={agg['with_data']['count']})】")
    print(f"  Hit@5: {agg['with_data']['hit']:.2%}")
    print(f"  MRR:   {agg['with_data']['mrr']:.3f}")

    print(f"\n【无数据的问题 (n={agg['without_data']['count']})】")
    print(f"  (预期无召回，用于验证数据覆盖)")
    for r in results:
        if r.gold_source_pattern == "无":
            print(f"    {r.query_id}: Hit={r.hit:.0f} - {r.query[:40]}")

    # 达标检查
    print(f"\n【目标检查】")
    target_hit = 0.70
    target_mrr = 0.60
    actual_hit = agg['with_data']['hit']
    actual_mrr = agg['with_data']['mrr']
    print(f"  Hit@5: {actual_hit:.2%} (目标 ≥{target_hit:.0%}) {'✓ 通过' if actual_hit >= target_hit else '✗ 未达标'}")
    print(f"  MRR:   {actual_mrr:.3f} (目标 ≥{target_mrr:.1f}) {'✓ 通过' if actual_mrr >= target_mrr else '✗ 未达标'}")

    # 低召回问题
    low_recall = [(r.query_id, r.query, r.hit) for r in results if r.hit < 0.5 and r.gold_source_pattern != "无"]
    if low_recall:
        print(f"\n【低召回问题 (Hit = 0)】")
        for rid, query, hit in low_recall:
            print(f"  {rid}: {query[:40]}... (Hit={hit:.0f})")


def main():
    config = AgentConfig.from_env()
    output_path = Path("data/eval/retrieval_phase3_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 运行测试
    results = run_retrieval_test(config, top_k=5)

    # 聚合统计
    agg = aggregate_results(results)

    # 打印汇总
    print_summary(results, agg)

    # 保存 JSON
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "test_cases": RETRIEVAL_TEST_CASES,
        "summary": agg,
        "details": [asdict(r) for r in results],
    }
    output_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
