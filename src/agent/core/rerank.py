from __future__ import annotations

import os
import re

from .retrieval import _tokenize
from .schemas import RetrievalHit


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
