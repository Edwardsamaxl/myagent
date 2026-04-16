#!/usr/bin/env python3
"""
Phase 3.3: 多 Agent 协作模型有效性测试

验证 Multi-Agent 协作机制（路由 Agent + 检索 Agent + 生成 Agent）的有效性。
测试问题基于实际 chunk 数据设计，确保答案有据可查。

用法:
    python scripts/eval_agent_collaboration_phase3.py
"""

from __future__ import annotations

import codecs
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "backslashreplace")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "backslashreplace")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from agent.config import AgentConfig
from agent.llm.providers import AnthropicCompatibleProvider, build_model_provider
from agent.llm.embeddings import build_embedding_provider
from agent.rag.retrieval import InMemoryHybridRetriever
from agent.rag.rerank import SimpleReranker
from agent.core.routing.llm_router import LLMRouterAgent
from agent.core.routing.rag_chain import RAGChain


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3.3 测试用例（基于实际 chunk 数据设计）
# 所有问题均可从索引数据中检索到正确答案
# ─────────────────────────────────────────────────────────────────────────────
COLLABORATION_TEST_CASES = [
    {
        "id": "M1",
        "query": "贵州茅台的股票代码是多少？",
        "route_type": "SINGLE_STEP",
        "expected_answer": "600519",
        "expected_company": "贵州茅台",
        "reason": "公司基本信息，单点查询",
    },
    {
        "id": "M2",
        "query": "贵州茅台2025年上半年归属于上市公司股东的净利润是多少？",
        "route_type": "SINGLE_STEP",
        "expected_answer": "45402962298.10",
        "expected_company": "贵州茅台",
        "reason": "2025H1财务指标，单点查询",
    },
    {
        "id": "M3",
        "query": "贵州茅台2025年上半年营收同比2024年同期增长了多少？",
        "route_type": "MULTI_STEP",
        "expected_answer": "9.10",
        "expected_company": "贵州茅台",
        "reason": "跨期对比：需检索2025H1和2024H1数据并计算增长率",
    },
    {
        "id": "M4",
        "query": "贵州茅台和五粮液2025年上半年净利润对比，谁赚得更多？",
        "route_type": "MULTI_STEP",
        "expected_answer": "贵州茅台",
        "expected_company": "贵州茅台",
        "reason": "双公司对比：贵州茅台454亿 vs 五粮液195亿",
    },
    {
        "id": "M5",
        "query": "贵州茅台2024年每10股派发现金红利是多少元？",
        "route_type": "SINGLE_STEP",
        "expected_answer": "276.24",
        "expected_company": "贵州茅台",
        "reason": "分红政策，单点查询",
    },
    {
        "id": "M6",
        "query": "贵州茅台2025年上半年经营活动产生的现金流量净额同比2024年同期变化多少？",
        "route_type": "MULTI_STEP",
        "expected_answer": "减少64.18",
        "expected_company": "贵州茅台",
        "reason": "跨期现金流对比：2025H1比2024H1减少64.18%",
    },
    {
        "id": "M7",
        "query": "贵州茅台2024年基本每股收益是多少元？",
        "route_type": "SINGLE_STEP",
        "expected_answer": "68.64",
        "expected_company": "贵州茅台",
        "reason": "2024年报财务指标，单点查询",
    },
    {
        "id": "M8",
        "query": "贵州茅台和五粮液2025年上半年基本每股收益分别是多少？",
        "route_type": "MULTI_STEP",
        "expected_answer": "贵州茅台36.18 五粮液5.0216",
        "expected_company": "贵州茅台",
        "reason": "双公司财务指标对比：贵州茅台36.18 vs 五粮液5.0216",
    },
    {
        "id": "M9",
        "query": "贵州茅台2024年营业收入变动的主要原因是什么？",
        "route_type": "SINGLE_STEP",
        "expected_answer": "销量增加",
        "expected_company": "贵州茅台",
        "reason": "年报管理层讨论：销量增加及茅台酒主要产品销售价格调整",
    },
    {
        "id": "M10",
        "query": "五粮液2025年上半年营业收入是多少元？",
        "route_type": "SINGLE_STEP",
        "expected_answer": "52770984383.52",
        "expected_company": "五粮液",
        "reason": "五粮液2025H1财务指标，单点查询",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# 答案质量评估工具
# ─────────────────────────────────────────────────────────────────────────────

def normalize_number(text: str) -> str:
    """提取文本中的数字，移除逗号和空格"""
    text = re.sub(r'[\s,，：:（）()"]', '', text)
    return text


def extract_number(text: str) -> str:
    """从文本中提取数值，优先匹配长数字（更精确）"""
    text = normalize_number(text)
    # 优先匹配较长数字（更精确，避免截取短数字）
    matches = re.findall(r'(\d+\.?\d*)', text)
    if not matches:
        return ""
    # 返回最长匹配（通常是要找的实际数字）
    return max(matches, key=len)


def _to_float(s: str) -> float | None:
    """尝试将字符串转为浮点数"""
    try:
        return float(s)
    except ValueError:
        return None


def _numbers_close(a: float, b: float) -> bool:
    """判断两个数值是否足够接近（考虑单位缩放）"""
    if a == b:
        return True
    # 5% 容差
    if b != 0 and abs(a - b) / b < 0.05:
        return True
    # 考虑缩放关系（亿=10^8，万=10^4，%=/100）
    scale_factors = [1e8, 1e4, 1e2, 1e-2, 1e-4, 1e-8]
    for scale in scale_factors:
        if b != 0 and abs(a - b * scale) / (b * scale) < 0.05:
            return True
        if a != 0 and abs(a * scale - b) / a < 0.05:
            return True
    return False


def check_answer_match(response: str, expected: str, case_id: str) -> dict[str, Any]:
    """
    检查回答是否包含期望答案
    返回: {"correct": bool, "score": float, "detail": str}

    评估标准：
    1. 数值匹配：考虑单位缩放（亿/万/%），5% 容差
    2. 关键词匹配：包含期望关键词即可
    3. 语义匹配：减少/增长等方向性词汇
    """
    response_lower = response.lower()
    expected_lower = expected.lower()

    # 数值匹配（考虑单位转换）
    if any(c in expected for c in '0123456789'):
        resp_num = extract_number(response)
        exp_num = extract_number(expected)
        if resp_num and exp_num:
            r = _to_float(resp_num)
            e = _to_float(exp_num)
            if r is not None and e is not None:
                if _numbers_close(r, e):
                    return {"correct": True, "score": 1.0, "detail": f"数值匹配: {r} ≈ {e}"}
                else:
                    return {"correct": False, "score": 0.0, "detail": f"数值偏差: 回答{r} vs 期望{e}"}

    # 语义方向匹配（减少/增长/上升/下降）
    decline_keywords = ["减少", "下降", "降低", "下滑", "负增长", "回落"]
    increase_keywords = ["增加", "增长", "上升", "提高", "上升", "扩"]
    if any(kw in expected_lower for kw in decline_keywords):
        if any(kw in response_lower for kw in decline_keywords):
            return {"correct": True, "score": 1.0, "detail": "语义匹配: 减少方向正确"}
    if any(kw in expected_lower for kw in increase_keywords):
        if any(kw in response_lower for kw in increase_keywords):
            return {"correct": True, "score": 1.0, "detail": "语义匹配: 增长方向正确"}

    # 关键词匹配
    keywords = expected.split()
    matched = sum(1 for kw in keywords if kw in response_lower)
    score = matched / len(keywords) if keywords else 0.0

    # 部分匹配逻辑
    if score >= 0.5:
        return {"correct": True, "score": score, "detail": f"关键词匹配率 {score:.0%}"}
    elif expected_lower in response_lower:
        return {"correct": True, "score": 1.0, "detail": "精确包含"}
    else:
        return {"correct": False, "score": score, "detail": f"未匹配期望: {expected}"}


def generate_answer(query: str, hits: list, provider: Any, model: str) -> str:
    """使用 LLM 根据检索结果生成答案"""
    if not hits:
        return "抱歉，未检索到相关信息。"

    # 构建上下文
    context_chunks = []
    for i, hit in enumerate(hits[:3]):
        source = hit.source if hasattr(hit, "source") else ""
        text = (hit.text if hasattr(hit, "text") else "")[:500]
        context_chunks.append(f"[{i+1}] 来源: {source}\n内容: {text}")

    context = "\n\n".join(context_chunks)

    prompt = f"""你是一个财务分析助手。请根据以下检索到的信息回答用户问题。

检索到的信息：
{context}

用户问题：{query}

要求：
1. 只基于检索到的信息回答，不要编造数据
2. 如果信息不足以回答，说明情况
3. 回答要简洁准确，列出具体数字和来源

回答："""

    try:
        response = provider.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
        )
        return response.strip()
    except Exception as e:
        return f"[生成失败: {str(e)}]"


# ─────────────────────────────────────────────────────────────────────────────
# 数据类
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CollaborationResult:
    query_id: str
    query: str
    expected_route: str
    actual_route: str
    route_correct: bool
    hits_count: int
    response: str
    answer_correct: bool
    answer_score: float
    evaluation_detail: str
    latency_ms: int
    error: str | None


# ─────────────────────────────────────────────────────────────────────────────
# 聚合统计
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_collaboration(results: list[CollaborationResult]) -> dict:
    valid = [r for r in results if r.error is None]
    errors = len(results) - len(valid)

    if not valid:
        return {"valid": 0, "errors": errors}

    route_correct = sum(1 for r in valid if r.route_correct)
    answer_correct = sum(1 for r in valid if r.answer_correct)
    answer_scores = [r.answer_score for r in valid]

    return {
        "valid": len(valid),
        "errors": errors,
        "route_accuracy": round(route_correct / len(valid), 4),
        "answer_accuracy": round(answer_correct / len(valid), 4),
        "answer_score_avg": round(sum(answer_scores) / len(answer_scores), 4),
        "by_route_type": _aggregate_by_route(valid),
    }


def _aggregate_by_route(results: list[CollaborationResult]) -> dict:
    by_type = {}
    for r in results:
        key = r.expected_route
        if key not in by_type:
            by_type[key] = {"total": 0, "route_correct": 0, "answer_correct": 0, "scores": []}
        by_type[key]["total"] += 1
        if r.route_correct:
            by_type[key]["route_correct"] += 1
        if r.answer_correct:
            by_type[key]["answer_correct"] += 1
        by_type[key]["scores"].append(r.answer_score)

    for key, v in by_type.items():
        n = v["total"]
        v["route_accuracy"] = round(v["route_correct"] / n, 4)
        v["answer_accuracy"] = round(v["answer_correct"] / n, 4)
        v["answer_score_avg"] = round(sum(v["scores"]) / n, 4)
        del v["scores"]
    return by_type


# ─────────────────────────────────────────────────────────────────────────────
# 主测试逻辑
# ─────────────────────────────────────────────────────────────────────────────

def run_collaboration_test(config: AgentConfig) -> list[CollaborationResult]:
    print("\n" + "=" * 60)
    print("Phase 3.3: 多 Agent 协作有效性测试")
    print("=" * 60)

    # 初始化 Provider（根据配置自动选择）
    try:
        provider = build_model_provider(config)
        print(f"  [OK] Provider: {config.model_provider}/{config.model_name}")
    except Exception as e:
        print(f"  [FAIL] Provider 初始化失败: {e}")
        return []

    # 初始化 Router
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
        print(f"  [FAIL] 索引目录不存在: {index_dir}")
        return []

    # 初始化 Reranker
    reranker = SimpleReranker()

    # 初始化 RAG Chain
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
    for tc in COLLABORATION_TEST_CASES:
        query = tc["query"]
        expected_route = tc["route_type"]
        expected_answer = tc["expected_answer"]
        print(f"\n  [{tc['id']}] {query[:40]}...")

        try:
            t0 = time.time()

            # 1. 路由决策
            routing_context = {"available_tools": [], "history": None}
            decision = router.route(query, context=routing_context)
            actual_route = decision.route_type.value.upper()
            route_correct = (actual_route == expected_route)

            # 2. RAG Chain（检索）
            rag_result = rag.execute(query, rewrite_mode="hybrid")
            hits = rag_result["hits"]
            hits_count = len(hits)

            # 3. 生成答案
            response = generate_answer(query, hits, provider, config.model_name)

            # 4. 评估答案质量
            eval_result = check_answer_match(response, expected_answer, tc["id"])
            answer_correct = eval_result["correct"]
            answer_score = eval_result["score"]

            latency_ms = int((time.time() - t0) * 1000)

            results.append(CollaborationResult(
                query_id=tc["id"],
                query=query,
                expected_route=expected_route,
                actual_route=actual_route,
                route_correct=route_correct,
                hits_count=hits_count,
                response=response[:200] + "..." if len(response) > 200 else response,
                answer_correct=answer_correct,
                answer_score=answer_score,
                evaluation_detail=eval_result["detail"],
                latency_ms=latency_ms,
                error=None,
            ))

            route_icon = "✓" if route_correct else "✗"
            answer_icon = "✓" if answer_correct else "✗"
            print(f"      路由: {route_icon} {actual_route}/{expected_route} | "
                  f"答案: {answer_icon} ({eval_result['detail'][:30]}) | "
                  f"延迟: {latency_ms}ms | 检索: {hits_count}条")

        except Exception as e:
            err = str(e)
            results.append(CollaborationResult(
                query_id=tc["id"],
                query=query,
                expected_route=expected_route,
                actual_route="ERROR",
                route_correct=False,
                hits_count=0,
                response="",
                answer_correct=False,
                answer_score=0.0,
                evaluation_detail="",
                latency_ms=0,
                error=err[:100],
            ))
            print(f"      [FAIL] {err[:60]}")
            continue

        time.sleep(1.5)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 输出汇总报告
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: list[CollaborationResult], agg: dict) -> None:
    print("\n" + "=" * 60)
    print("Phase 3.3 汇总报告")
    print("=" * 60)

    print(f"\n【总体统计】")
    print(f"  有效测试: {agg['valid']}/10 条")
    if agg['errors'] > 0:
        print(f"  失败: {agg['errors']} 条")

    if agg['valid'] == 0:
        print("\n[SKIP] 无有效测试结果，跳过统计汇总")
        failures = [(r.query_id, r.query, r.error) for r in results]
        print("\n【失败详情】")
        for fid, query, err in failures:
            print(f"  {fid}: {query[:40]}... -> {err}")
        return

    print(f"\n【协作准确性 - 路由】")
    print(f"  路由准确率: {agg['route_accuracy']:.2%} (目标 >85%)")
    route_status = "✓ 通过" if agg['route_accuracy'] > 0.85 else "✗ 未达标"
    print(f"  状态: {route_status}")

    print(f"\n【最终答案质量】")
    print(f"  答案准确率: {agg['answer_accuracy']:.2%} (目标 >90%)")
    print(f"  平均得分: {agg['answer_score_avg']:.2%}")
    answer_status = "✓ 通过" if agg['answer_accuracy'] > 0.90 else "✗ 未达标"
    print(f"  状态: {answer_status}")

    print(f"\n【按路由类型分析】")
    print(f"  {'类型':<12} {'数量':>6} {'路由准确率':>12} {'答案准确率':>12} {'平均得分':>10}")
    print("  " + "-" * 56)
    for route_type, stats in agg["by_route_type"].items():
        print(f"  {route_type:<12} {stats['total']:>6} "
              f"{stats['route_accuracy']:>11.2%} {stats['answer_accuracy']:>11.2%} "
              f"{stats['answer_score_avg']:>9.2%}")

    # 失败案例
    failures = [(r.query_id, r.query, r.expected_route, r.actual_route,
                 r.evaluation_detail) for r in results if not r.answer_correct or not r.route_correct]
    if failures:
        print(f"\n【失败案例】")
        for fid, query, exp, act, detail in failures:
            print(f"  {fid}: 期望路由={exp} 实际路由={act}")
            print(f"       问题: {query[:40]}...")
            print(f"       评估: {detail}")

    # 验收标准
    print(f"\n【验收标准检查】")
    checks = [
        ("路由准确率 > 85%", agg['route_accuracy'] > 0.85, f"{agg['route_accuracy']:.2%}"),
        ("答案准确率 > 90%", agg['answer_accuracy'] > 0.90, f"{agg['answer_accuracy']:.2%}"),
    ]
    for name, passed, value in checks:
        status = "✓ 通过" if passed else "✗ 未达标"
        print(f"  {name}: {value} {status}")


def main():
    config = AgentConfig.from_env()
    output_path = Path("data/eval/collaboration_phase3_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 运行测试
    results = run_collaboration_test(config)

    # 聚合统计
    agg = aggregate_collaboration(results)

    # 打印汇总
    print_summary(results, agg)

    # 保存 JSON
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "test_cases": COLLABORATION_TEST_CASES,
        "summary": agg,
        "details": [
            {
                "query_id": r.query_id,
                "query": r.query,
                "expected_route": r.expected_route,
                "actual_route": r.actual_route,
                "route_correct": r.route_correct,
                "hits_count": r.hits_count,
                "response": r.response,
                "answer_correct": r.answer_correct,
                "answer_score": r.answer_score,
                "evaluation_detail": r.evaluation_detail,
                "latency_ms": r.latency_ms,
                "error": r.error,
            }
            for r in results
        ],
    }
    output_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
