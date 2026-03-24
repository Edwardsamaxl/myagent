from __future__ import annotations

import os
import re

from ..llm.providers import Message, ModelProvider
from .evidence_format import (
    citations_are_valid,
    citation_ids_for_hits,
    contains_citation_marker,
    evaluate_anchor_coverage,
    extract_citation_indices,
    format_citation_lines,
    format_evidence_block_from_hits,
    normalize_structured_answer,
    pick_key_evidence_snippet,
    select_evidence_hits,
)
from .schemas import GenerationResult, RetrievalHit


class GroundedGenerator:
    """基于检索命中生成回答：引用编号与证据块序号一致，拒答与 `GenerationResult.reason` 可观测。"""

    # 若检索返回结构或证据拼接方式变化，请先与检索侧对齐，再改本段（见 evidence_format 模块说明）。
    SYSTEM_PROMPT = """你是金融知识助手。回答必须遵循：
1) 只基于给定证据回答，不允许臆测；
2) 输出结构必须包含三段：结论、关键依据、引用编号；
3) 证据块已按 [1]、[2]… 编号；回答正文中引用时使用相同编号，如「……[1][2]」；
4) 不允许无引用硬答；
5) 若证据不足或问题超出证据范围，请只输出一行：拒答：<简短原因>（不要输出其它段落）；
6) 若能回答，不要用「拒答：」句式；
7) 关键依据优先给出可核对数字或事实短句，避免泛泛分析。"""

    def __init__(self, model: ModelProvider, temperature: float, max_tokens: int) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # 便于离线 AB：0=旧策略，1=严格证据策略（默认开启）
        self.strict_policy = os.getenv("GENERATION_STRICT_POLICY", "1").strip().lower() not in {
            "0",
            "false",
            "off",
        }
        self.refuse_coverage_threshold = float(
            os.getenv("GENERATION_REFUSE_COVERAGE_THRESHOLD", "0.10")
        )
        self.cautious_coverage_threshold = float(
            os.getenv("GENERATION_CAUTIOUS_COVERAGE_THRESHOLD", "0.22")
        )

    def generate(self, question: str, hits: list[RetrievalHit]) -> GenerationResult:
        if not hits:
            return GenerationResult(
                answer="当前检索不到足够证据，无法给出可靠结论。",
                citations=[],
                refusal=True,
                reason="no_retrieval_hit",
            )

        # 固化证据选取：仅对选中的证据生成回答与引用，减少“有命中但引用弱关联”。
        evidence_hits = select_evidence_hits(question=question, hits=hits, max_evidence=3)
        context = format_evidence_block_from_hits(evidence_hits)
        citations = format_citation_lines(evidence_hits)

        prompt = (
            f"用户问题:\n{question}\n\n"
            f"可用证据（字段与检索结果一致：chunk_id、score、source；正文在「内容」下）：\n{context}\n\n"
            "请按系统指令作答。"
        )
        messages: list[Message] = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        raw = self.model.generate(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ).strip()
        answer, refusal, reason = self._postprocess_answer(
            raw=raw,
            question=question,
            hits=evidence_hits,
            citation_ids=citation_ids_for_hits(evidence_hits),
        )
        return GenerationResult(answer=answer, citations=citations, refusal=refusal, reason=reason)

    _REF_LINE_RE = re.compile(r"\n引用[:：]\s*[\[\]\d,\s]+\s*$")
    _KEY_EVIDENCE_PREFIX = "关键依据："
    _INSUFFICIENT_PATTERNS = (
        "证据不足",
        "信息不足",
        "无法基于当前证据",
        "无法根据当前证据",
        "无法判断",
        "无法确定",
        "无法给出可靠结论",
        "超出证据范围",
    )
    def _postprocess_answer(
        self,
        raw: str,
        question: str,
        hits: list[RetrievalHit],
        citation_ids: list[str],
    ) -> tuple[str, bool, str]:
        """统一拒答判定与结构规范化；引用列表仍以 hits 为准，不依赖模型解析。"""
        text = raw.strip()
        if not text:
            return "拒答：证据不足，无法给出可靠结论。", True, "insufficient_evidence"
        if text.startswith(("拒答：", "拒答:")):
            return text, True, "insufficient_evidence"
        first_line = text.split("\n", 1)[0].strip()
        if first_line.startswith("当前检索不到"):
            # 注意：本分支只发生在“有命中但模型仍给空检索式拒答”的输出，语义统一归为证据不足。
            return text, True, "insufficient_evidence"
        if any(pattern in text for pattern in self._INSUFFICIENT_PATTERNS):
            normalized = text
            if not normalized.startswith(("拒答：", "拒答:")):
                normalized = f"拒答：{first_line or '证据不足，无法给出可靠结论。'}"
            return normalized, True, "insufficient_evidence"
        # 去掉可选的文末「引用：」行，避免与 citations 重复堆叠
        cleaned = self._REF_LINE_RE.sub("", text).strip()
        structured = normalize_structured_answer(cleaned, citation_ids)
        # 旧策略：仅维持基础约束，便于 AB 对比。
        if not self.strict_policy:
            if not contains_citation_marker(structured):
                return "拒答：证据不足，无法建立可引用的回答。", True, "insufficient_evidence"
            return self._ensure_key_evidence_line(structured, hits), False, ""

        # 1) 引用合规校验
        if not contains_citation_marker(structured) or not citations_are_valid(structured, len(hits)):
            return "拒答：回答缺少有效引用编号，无法保证可追溯性。", True, "citation_missing"

        # 2) 锚点覆盖校验（防“有引用但不对题”）
        coverage, detail = evaluate_anchor_coverage(question=question, answer=structured, hits=hits)
        if coverage < self.refuse_coverage_threshold:
            return (
                f"拒答：证据与问题关键锚点匹配不足（{detail['covered']}/{detail['anchors']}），无法给出可靠结论。",
                True,
                "insufficient_evidence",
            )
        if coverage < self.cautious_coverage_threshold:
            cautious = self._cautious_template(question=question, hits=hits, citation_ids=citation_ids)
            return cautious, False, ""
        return self._ensure_key_evidence_line(structured, hits), False, ""

    @staticmethod
    def _ensure_key_evidence_line(answer: str, hits: list[RetrievalHit]) -> str:
        """关键依据段至少含一条可定位事实片段。"""
        if GroundedGenerator._KEY_EVIDENCE_PREFIX not in answer:
            return answer
        snippet = pick_key_evidence_snippet("", hits)
        if not snippet:
            return answer
        lines = answer.splitlines()
        out: list[str] = []
        replaced = False
        for line in lines:
            if line.startswith(GroundedGenerator._KEY_EVIDENCE_PREFIX):
                content = line[len(GroundedGenerator._KEY_EVIDENCE_PREFIX) :].strip()
                if len(content) < 8:
                    out.append(f"{GroundedGenerator._KEY_EVIDENCE_PREFIX}{snippet}")
                    replaced = True
                    continue
            out.append(line)
        if not replaced:
            return answer
        return "\n".join(out)

    @staticmethod
    def _cautious_template(question: str, hits: list[RetrievalHit], citation_ids: list[str]) -> str:
        """匹配度偏低时的降级回答模板（非硬拒答）。"""
        cited = extract_citation_indices("".join(citation_ids))
        ref_text = "".join(f"[{i}]" for i in cited) if cited else "".join(citation_ids)
        snippet = pick_key_evidence_snippet(question, hits) or "证据片段信息有限。"
        return (
            "结论：基于现有证据可给出谨慎估计，建议结合后续披露数据复核。\n"
            f"关键依据：证据显示 {snippet}\n"
            f"引用编号：{ref_text}"
        )
