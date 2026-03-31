from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import requests

from ..config import AgentConfig
from .retrieval import _tokenize
from .schemas import RetrievalHit

logger = logging.getLogger(__name__)


class SimpleReranker:
    """Rerank v2: 基础分 + keyword/length/metadata/numeric 规则，与 retrieval 形成 recall+rerank 分层。"""

    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int = 3) -> list[RetrievalHit]:
        reranked, _ = self.rerank_with_debug(query=query, hits=hits, top_k=top_k)
        return reranked

    def rerank_with_debug(
        self, query: str, hits: list[RetrievalHit], top_k: int = 3
    ) -> tuple[list[RetrievalHit], list[dict[str, float | str]]]:
        if not hits:
            return [], []
        keyword_on = _env_bool("RERANK_KEYWORD_BONUS_ENABLED", True)
        length_penalty_on = _env_bool("RERANK_LENGTH_PENALTY_ENABLED", True)
        metadata_on = _env_bool("RERANK_METADATA_BONUS_ENABLED", True)
        numeric_on = _env_bool("RERANK_NUMERIC_BONUS_ENABLED", True)
        keyword_unit_bonus = 0.015
        length_penalty_value = 0.05
        query_tokens = set(_tokenize(query))
        query_nums = self._extract_numbers(query)
        constraints = self._extract_query_constraints(query)

        scored: list[tuple[RetrievalHit, float, dict[str, float | str]]] = []
        for hit in hits:
            final_score, breakdown = self._score_hit(
                query=query,
                hit=hit,
                query_tokens=query_tokens,
                query_nums=query_nums,
                constraints=constraints,
                keyword_on=keyword_on,
                keyword_unit_bonus=keyword_unit_bonus,
                length_penalty_on=length_penalty_on,
                length_penalty_value=length_penalty_value,
                metadata_on=metadata_on,
                numeric_on=numeric_on,
            )
            scored.append((hit, final_score, breakdown))

        # 排序稳定性：先按 final，再按 base，再按 chunk_id，保证 C 侧消费顺序可复现。
        scored.sort(
            key=lambda item: (
                item[1],
                float(item[2].get("base", 0.0)),
                str(item[0].chunk_id),
            ),
            reverse=True,
        )
        picked = scored[: max(1, top_k)]
        reranked_hits = [item[0] for item in picked]
        rerank_breakdown: list[dict[str, float | str]] = []
        for idx, item in enumerate(picked, start=1):
            row = dict(item[2])
            row["rank"] = str(idx)
            rerank_breakdown.append(row)
        return reranked_hits, rerank_breakdown

    def _score_hit(
        self,
        *,
        query: str,
        hit: RetrievalHit,
        query_tokens: set[str],
        query_nums: set[str],
        constraints: dict[str, object],
        keyword_on: bool,
        keyword_unit_bonus: float,
        length_penalty_on: bool,
        length_penalty_value: float,
        metadata_on: bool,
        numeric_on: bool,
    ) -> tuple[float, dict[str, float | str]]:
        base = hit.score
        if keyword_on:
            keyword_bonus = sum(keyword_unit_bonus for t in query_tokens if t and t in hit.text.lower())
        else:
            keyword_bonus = 0.0
        intent_bonus = self._intent_bonus(query, hit.text)
        if length_penalty_on:
            length_penalty = 0.0 if 120 <= len(hit.text) <= 800 else length_penalty_value
        else:
            length_penalty = 0.0
        metadata_bonus = (
            self._metadata_bonus(query, hit.metadata, constraints)
            if metadata_on
            else 0.0
        )
        numeric_bonus = self._numeric_bonus(query_nums, hit.text) if numeric_on else 0.0
        lexical_score = keyword_bonus + intent_bonus
        semantic_score = base + metadata_bonus + numeric_bonus
        final_score = semantic_score + lexical_score - length_penalty
        breakdown: dict[str, float | str] = {
            "chunk_id": hit.chunk_id,
            "source": hit.source,
            "base": round(base, 6),
            "lexical": round(lexical_score, 6),
            "semantic": round(semantic_score, 6),
            "length_penalty": round(length_penalty, 6),
            "final": round(final_score, 6),
        }
        return final_score, breakdown

    @staticmethod
    def _metadata_bonus(query: str, metadata: dict[str, str], constraints: dict[str, object]) -> float:
        """从 retrieval 迁入：date/doc_type 软约束；可选 company 匹配。"""
        if not metadata:
            return 0.0
        q = query
        bonus = 0.0
        years = constraints.get("years", set())
        date_val = metadata.get("date", "")
        if years and date_val:
            if any(y in date_val for y in years):
                bonus += 0.08
            else:
                bonus -= 0.1

        # doc_type：半年报 / 年报
        bonus += SimpleReranker._doc_type_bonus(
            doc_type=metadata.get("doc_type", ""),
            expect_doc_type=str(constraints.get("doc_type", "")),
            is_profile_query=bool(constraints.get("is_profile_query", False)),
        )

        # 可选：company 与 query 主体匹配
        company = metadata.get("company", "")
        if company and company in q:
            bonus += 0.03

        return bonus

    @staticmethod
    def _doc_type_bonus(*, doc_type: str, expect_doc_type: str, is_profile_query: bool) -> float:
        if expect_doc_type:
            return 0.05 if expect_doc_type in doc_type else -0.08
        if not is_profile_query:
            return 0.0
        # 公司基本面无显式年份时，优先年报，抑制半年报误命中。
        if "年报" in doc_type:
            return 0.08
        if "半年报" in doc_type:
            return -0.08
        return 0.0

    @staticmethod
    def _extract_query_constraints(query: str) -> dict[str, object]:
        years = set(re.findall(r"\b(?:19|20)\d{2}\b", query))
        if not years:
            years = set(re.findall(r"(?:19|20)\d{2}", query))
        doc_type = ""
        if "半年报" in query or "上半年" in query:
            doc_type = "半年报"
        elif "年报" in query or "年度报告" in query:
            doc_type = "年报"
        profile_keys = ("法定代表人", "注册地址", "董事会秘书", "会计师事务所", "办公地址")
        is_profile_query = any(k in query for k in profile_keys)
        return {
            "years": years,
            "doc_type": doc_type,
            "is_profile_query": is_profile_query,
        }

    @staticmethod
    def _intent_bonus(query: str, chunk_text: str) -> float:
        # 对高价值短语做精确匹配，帮助同源文档中的实体字段优先。
        anchors = ("法定代表人", "注册地址", "董事会秘书", "会计师事务所", "营业收入", "变动原因")
        bonus = 0.0
        for phrase in anchors:
            if phrase in query and phrase in chunk_text:
                bonus += 0.04
        return bonus

    @staticmethod
    def _extract_numbers(text: str) -> set[str]:
        """提取 query 中的数字（股票代码、金额、比率等），用于 numeric_bonus。"""
        nums: set[str] = set()
        # 整数（含 6 位股票代码、4 位年份等）
        for m in re.finditer(r"\d+", text):
            nums.add(m.group())
        # 小数、比率（如 15.71、12.5%）
        for m in re.finditer(r"\d+\.\d+%?|\d+%", text):
            nums.add(m.group())
        return nums

    @staticmethod
    def _numeric_bonus(query_nums: set[str], chunk_text: str) -> float:
        """query 中数字在 chunk 文本中出现，+0.04/类，每类最多一次。"""
        if not query_nums:
            return 0.0
        bonus = 0.0
        for num in query_nums:
            if num in chunk_text:
                bonus += 0.04
        return bonus


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class RerankResult:
    """BGE Reranker 返回的单个重排结果。"""
    hit: RetrievalHit
    rerank_score: float


class BGEReranker:
    """基于 Ollama BGE Reranker v2-m3 的文档重排序。

    使用 cross-encoder 架构，将 query + passage 一起编码，
    输出 0~1 相关度分数，按分数降序排列。

    部署要求：
        ollama pull BAAI/bge-reranker-v2-m3

    API：POST /api/rerank
        {
          "model": "BAAI/bge-reranker-v2-m3",
          "query": "...",
          "documents": ["...", "..."]
        }

    返回：{"results": [{"index": 0, "relevance_score": 0.95}, ...]}
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "BAAI/bge-reranker-v2-m3",
        timeout: int = 60,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._healthy = True
        self._health_checked = False

    def _health_check(self) -> bool:
        """启动时验证 Ollama rerank API 是否可用。"""
        if self._health_checked:
            return self._healthy
        self._health_checked = True
        try:
            # 用最短的 query + 1 个空 doc 验证连通性
            resp = requests.post(
                f"{self.base_url}/api/rerank",
                json={"model": self.model, "query": "健康检查", "documents": ["测试"]},
                timeout=10,
            )
            resp.raise_for_status()
            self._healthy = True
        except Exception as exc:
            self._healthy = False
            logger.warning(
                "[BGEReranker] 健康检查失败，将降级为 SimpleReranker。"
                f"原因: {exc}。"
                "修复: 确认已运行 'ollama pull BAAI/bge-reranker-v2-m3'。"
            )
        return self._healthy

    def rerank(
        self, query: str, hits: list[RetrievalHit], top_k: int = 3
    ) -> list[RetrievalHit]:
        reranked, _ = self.rerank_with_debug(query=query, hits=hits, top_k=top_k)
        return reranked

    def rerank_with_debug(
        self, query: str, hits: list[RetrievalHit], top_k: int = 3
    ) -> tuple[list[RetrievalHit], list[dict[str, float | str]]]:
        if not hits:
            return [], []

        if not self._health_check():
            logger.warning("[BGEReranker] 未通过健康检查，降级为 SimpleReranker。")
            return [], []

        documents = [hit.text for hit in hits]

        try:
            resp = requests.post(
                f"{self.base_url}/api/rerank",
                json={"model": self.model, "query": query, "documents": documents},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            results: list[dict[str, Any]] = resp.json().get("results", [])
        except requests.RequestException as exc:
            logger.warning(f"[BGEReranker] API 调用失败: {exc}，跳过重排。")
            return hits, []

        # 构建 index → score 映射
        score_map: dict[int, float] = {}
        for item in results:
            idx = item.get("index", -1)
            score = item.get("relevance_score", 0.0)
            if 0 <= idx < len(hits):
                score_map[idx] = score

        # 按 score 降序排列
        sorted_indices = sorted(
            range(len(hits)),
            key=lambda i: score_map.get(i, 0.0),
            reverse=True,
        )

        reranked_hits = [hits[i] for i in sorted_indices[:top_k]]
        rerank_breakdown: list[dict[str, float | str]] = []
        for rank, idx in enumerate(sorted_indices[:top_k], start=1):
            rerank_breakdown.append({
                "chunk_id": hits[idx].chunk_id,
                "source": hits[idx].source,
                "rerank_score": round(score_map.get(idx, 0.0), 6),
                "original_rank": str(idx + 1),
                "new_rank": str(rank),
            })

        return reranked_hits, rerank_breakdown


def build_reranker(config: AgentConfig) -> SimpleReranker | BGEReranker:
    """根据配置构建合适的 reranker。

    RERANK_ENABLED=false → SimpleReranker（直接返回）
    RERANK_ENABLED=true + RERANKER_PROVIDER=ollama → BGEReranker（健康检查失败则降级）
    """
    if not config.rerank_enabled:
        return SimpleReranker()

    if config.rerank_provider == "ollama":
        reranker = BGEReranker(
            base_url=config.rerank_base_url,
            model=config.rerank_model,
        )
        if not reranker._health_check():
            logger.warning("[RAG] BGEReranker 健康检查未通过，使用 SimpleReranker 作为 fallback。")
            return SimpleReranker()
        return reranker

    # 其他 provider fallback
    return SimpleReranker()
