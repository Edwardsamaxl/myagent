"""Router module: 查询类型分类 + Agent 路由决策"""
from .agent_router import AgentRouter, Route, RouterDecision

# Query type classification functions (moved from src/agent/core/router.py)
# to avoid import conflicts between router/ package and router.py sibling module
import re
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


def classify_query_type(query: str) -> QueryType:
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


_WEIGHT_PRESETS: dict[QueryType, dict[str, float]] = {
    QueryType.NUMERIC: {"lexical": 0.40, "tfidf": 0.25, "embedding": 0.25, "numeric": 0.10},
    QueryType.ENTITY: {"lexical": 0.35, "tfidf": 0.25, "embedding": 0.35, "numeric": 0.05},
    QueryType.SEMANTIC: {"lexical": 0.20, "tfidf": 0.15, "embedding": 0.60, "numeric": 0.05},
    QueryType.HYBRID: {"lexical": 0.35, "tfidf": 0.25, "embedding": 0.40, "numeric": 0.00},
}


def get_weights_for_type(qtype: QueryType) -> dict[str, float]:
    return _WEIGHT_PRESETS.get(qtype, _WEIGHT_PRESETS[QueryType.HYBRID]).copy()


def extract_numeric_indicators(query: str) -> set[str]:
    indicators: set[str] = set()
    for kw in _ENTITY_VALUE_KEYWORDS:
        if kw in query:
            indicators.add(kw)
    for year in re.findall(r"\d{4}年?", query):
        indicators.add(year)
    return indicators


__all__ = [
    "AgentRouter",
    "Route",
    "RouterDecision",
    "QueryType",
    "classify_query_type",
    "get_weights_for_type",
    "extract_numeric_indicators",
]
