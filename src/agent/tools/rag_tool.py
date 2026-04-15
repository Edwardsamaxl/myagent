"""RAG search_knowledge_base tool: retrieves evidence without LLM generation.

This module provides the search_knowledge_base tool function that the Agent
can call on-demand. It implements the retrieval pipeline (intent classify,
query rewrite, retrieval, rerank) and returns formatted evidence blocks.
"""

from __future__ import annotations

import os
import re
from typing import Any

from ..core.dialogue import classify_intent, rewrite_for_rag, IntentResult
from ..core.dialogue.query_rewrite import QueryRewriteResult
from ..rag.evidence_format import format_evidence_block_from_hits
from ..rag.schemas import RetrievalHit

# Reference coverage threshold for quality warning
_REFUSE_COVERAGE_THRESHOLD = float(os.getenv("AGENT_REFUSE_COVERAGE_THRESHOLD", "0.10"))

# Anchor token extraction for coverage evaluation
_ANCHOR_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{2,}|\d+(?:\.\d+)?%?")
_STOPWORDS = {
    "公司", "多少", "什么", "如何", "是否", "以及", "这个", "那个", "根据", "报告", "数据", "其中",
}

# Inline router functions to avoid import shadowing issue (router/ package vs router.py)
from enum import Enum


class QueryType(Enum):
    NUMERIC = "numeric"
    ENTITY = "entity"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


_ENTITY_VALUE_KEYWORDS = {
    "营业收入", "净利润", "归属于上市公司股东净利润", "基本每股收益",
    "加权平均净资产收益率", "每10股派发现金红利", "总资产", "净资产",
    "经营活动产生的现金流量净额", "总股本", "注册资本",
    "利润总额", "息税前利润", "毛利润", "毛利率",
    "同比增长", "同比变化", "同比增长率为", "增长率为",
    "资产负债率", "毛利率", "净利率", " ROE", "投入资本回报率",
    "持股", "股利的", "分红", "派息", "送股", "转增",
    "法定代表人", "注册地址", "股票代码", "董事会秘书", "会计师事务所",
    "上市", "交易所", "公司名称", "公司地址", "办公地址",
}

_SEMANTIC_VERBS = {
    "解释", "原因", "如何", "怎么样", "为什么", "说明",
    "分析", "评估", "判断", "对比", "比较", "区别",
    "介绍", "概述", "总结", "归纳", "梳理",
    "是否", "能否", "可否", "能不能", "会不会",
    "有什么", "有哪些", "是什么", "指什么",
}

_NUMERIC_PATTERNS = [
    re.compile(r"\d+[%％]"),
    re.compile(r"\d+[多多少]?[亿万元]"),
    re.compile(r"\d+[.．]\d+[%％]?"),
    re.compile(r"\d{4}年\d{1,2}月"),
    re.compile(r"\d{4}年"),
    re.compile(r"第\d+[多多少]?[亿万元]?"),
    re.compile(r"同比增长\s*\d+"),
    re.compile(r"下降\s*\d+"),
]

_WEIGHT_PRESETS: dict[QueryType, dict[str, float]] = {
    QueryType.NUMERIC: {"lexical": 0.40, "tfidf": 0.25, "embedding": 0.25, "numeric": 0.10},
    QueryType.ENTITY: {"lexical": 0.35, "tfidf": 0.25, "embedding": 0.35, "numeric": 0.05},
    QueryType.SEMANTIC: {"lexical": 0.20, "tfidf": 0.15, "embedding": 0.60, "numeric": 0.05},
    QueryType.HYBRID: {"lexical": 0.35, "tfidf": 0.25, "embedding": 0.40, "numeric": 0.00},
}


def classify_query_type(query: str) -> QueryType:
    """Classify query type for retrieval weight selection."""
    q = query.strip()
    numeric_matches = sum(1 for p in _NUMERIC_PATTERNS if p.search(q))
    entity_kw_hits = [kw for kw in _ENTITY_VALUE_KEYWORDS if kw in q]
    has_number = bool(re.search(r"\d", q))
    has_percent = bool(re.search(r"[%％]", q))

    if numeric_matches >= 2 or (has_number and has_percent and numeric_matches >= 1):
        return QueryType.NUMERIC
    if has_number and entity_kw_hits:
        return QueryType.NUMERIC
    if entity_kw_hits:
        return QueryType.ENTITY
    if any(verb in q for verb in _SEMANTIC_VERBS):
        return QueryType.SEMANTIC
    return QueryType.HYBRID


def get_weights_for_type(qtype: QueryType) -> dict[str, float]:
    """Return recommended weights for query type."""
    return _WEIGHT_PRESETS.get(qtype, _WEIGHT_PRESETS[QueryType.HYBRID]).copy()


def extract_numeric_indicators(query: str) -> set[str]:
    """Extract financial indicator keywords from query."""
    indicators: set[str] = set()
    for kw in _ENTITY_VALUE_KEYWORDS:
        if kw in query:
            indicators.add(kw)
    for year in re.findall(r"\d{4}年?", query):
        indicators.add(year)
    return indicators


def _extract_anchor_tokens(text: str) -> list[str]:
    """Extract key anchor tokens (numbers, years, percentages, key entities)."""
    tokens: list[str] = []
    for m in _ANCHOR_TOKEN_RE.finditer(text):
        token = m.group(0).strip()
        if len(token) <= 1:
            continue
        if token in _STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _evaluate_query_coverage(question: str, hit_texts: list[str]) -> tuple[float, dict[str, int]]:
    """Calculate query anchor coverage in evidence texts."""
    evidence_text = "\n".join(hit_texts).lower()
    q_anchors = _extract_anchor_tokens(question)
    if not q_anchors:
        return 1.0, {"anchors": 0, "covered": 0}
    covered = sum(1 for token in q_anchors if token.lower() in evidence_text)
    return covered / len(q_anchors), {"anchors": len(q_anchors), "covered": covered}


def search_knowledge_base_impl(
    query: str,
    retriever: Any,
    reranker: Any,
    config: Any,
) -> str:
    """Search knowledge base and return formatted evidence block.

    This function implements the retrieval pipeline without LLM generation:
    1. Classify intent and rewrite query for RAG
    2. Classify query type and apply dynamic weights
    3. Execute retrieval
    4. Rerank hits
    5. Format and return evidence block

    Args:
        query: The search query string
        retriever: InMemoryHybridRetriever instance
        reranker: Reranker instance (SimpleReranker or similar)
        config: AgentConfig with retrieval_top_k, rerank_top_k settings

    Returns:
        Formatted evidence block string, or empty/insufficient result message
    """
    if not query.strip():
        return "查询字符串为空。"

    try:
        top_k = config.retrieval_top_k
        rerank_top_k = config.rerank_top_k

        # Step 1: Classify intent and rewrite query for RAG
        intent: IntentResult = classify_intent(query, None)
        rewritten = rewrite_for_rag(
            turn_text=query,
            history=None,
            intent=intent,
            model=None,  # No LLM rewrite in tool mode
        )
        rewritten_query = str(rewritten) if isinstance(rewritten, QueryRewriteResult) else rewritten

        # Step 2: Classify query type and apply dynamic weights
        qtype = classify_query_type(rewritten_query)
        weights = get_weights_for_type(qtype)
        retriever.set_weights(
            lexical=weights.get("lexical"),
            tfidf=weights.get("tfidf"),
            embedding=weights.get("embedding"),
            numeric=weights.get("numeric"),
        )

        # Step 3: Extract numeric indicators for keyword boost
        numeric_indicators = extract_numeric_indicators(rewritten_query)

        # Step 4: Execute retrieval with rewritten query and numeric indicators
        hits, retrieval_debug = retriever.search_with_debug(
            rewritten_query,
            top_k=top_k,
            numeric_indicators=numeric_indicators,
        )

        if not hits:
            return "检索结果为空，无法找到相关信息。"

        # Step 5: Rerank hits
        reranked_hits, _ = reranker.rerank_with_debug(
            query=query,
            hits=hits,
            top_k=rerank_top_k,
        )

        if not reranked_hits:
            return "检索结果为空，无法找到相关信息。"

        # Step 6: Format evidence block
        evidence_block = format_evidence_block_from_hits(reranked_hits)

        # Step 7: Coverage quality check
        hit_texts = [hit.text for hit in reranked_hits]
        coverage, detail = _evaluate_query_coverage(query, hit_texts)
        refuse_flag = ""
        if coverage < _REFUSE_COVERAGE_THRESHOLD:
            refuse_flag = (
                f"\n\n[引用合规警告] 证据覆盖度不足（{detail['covered']}/{detail['anchors']}），"
                "低质量证据已标记。\n__refuse__"
            )

        # Build formatted evidence output
        lines = []
        for i, hit in enumerate(reranked_hits[:5], 1):
            preview = hit.text[:300]
            source = hit.source or "unknown"
            score = hit.score
            lines.append(f"[{i}] 来源: {source} (相关度: {score:.2f})")
            lines.append(f"内容: {preview}")
            lines.append("")
        return "\n".join(lines) + refuse_flag

    except Exception as exc:  # noqa: BLE001
        return f"检索失败: {exc}"


def create_search_knowledge_base_tool(
    rag_service: Any,
) -> tuple[str, str, Any]:
    """Create the search_knowledge_base tool.

    Returns a tuple of (name, description, func) compatible with the Tool dataclass.
    The tool function wraps search_knowledge_base_impl with the rag_service's
    retriever and reranker.
    """
    def search_knowledge_base(query: str) -> str:
        """Search the knowledge base for relevant information.

        This tool retrieves evidence from the knowledge base for factual queries
        such as data, metrics, and events. Returns formatted evidence blocks.

        Args:
            query: The search query string

        Returns:
            Formatted evidence block with sources and relevance scores,
            or an empty/insufficient result message.
        """
        return search_knowledge_base_impl(
            query=query,
            retriever=rag_service.retriever,
            reranker=rag_service.reranker,
            config=rag_service.config,
        )

    name = "search_knowledge_base"
    description = (
        "在知识库中检索相关信息。输入查询字符串，返回相关文档片段（格式化证据块）。"
        "用于回答需要事实依据的问题，如数据、指标、事件等。返回格式：[序号] 来源: xxx "
        "(相关度: x.xx) 内容: xxx。"
    )
    return name, description, search_knowledge_base
