from __future__ import annotations

import re

from ..llm.providers import Message, ModelProvider
from .evidence_format import format_citation_lines, format_evidence_block_from_hits
from .schemas import GenerationResult, RetrievalHit


class GroundedGenerator:
    """基于检索命中生成回答：引用编号与证据块序号一致，拒答与 `GenerationResult.reason` 可观测。"""

    # 若检索返回结构或证据拼接方式变化，请先与检索侧对齐，再改本段（见 evidence_format 模块说明）。
    SYSTEM_PROMPT = """你是金融知识助手。回答必须遵循：
1) 只基于给定证据回答，不允许臆测；
2) 证据块已按 [1]、[2]… 编号；回答正文中引用时使用相同编号，如「……[1][2]」；
3) 若证据不足或问题超出证据范围，请只输出一行：拒答：<简短原因>（不要输出其它段落）；
4) 若能回答，先给出结论与推理；不要用「拒答：」句式；文末可单独一行写「引用：[1][2]」以便核对（可选）。"""

    def __init__(self, model: ModelProvider, temperature: float, max_tokens: int) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, question: str, hits: list[RetrievalHit]) -> GenerationResult:
        if not hits:
            return GenerationResult(
                answer="当前检索不到足够证据，无法给出可靠结论。",
                citations=[],
                refusal=True,
                reason="no_retrieval_hit",
            )

        context = format_evidence_block_from_hits(hits)
        citations = format_citation_lines(hits)

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
        answer, refusal, reason = self._postprocess_answer(raw)
        return GenerationResult(answer=answer, citations=citations, refusal=refusal, reason=reason)

    _REF_LINE_RE = re.compile(r"\n引用[:：]\s*[\[\]\d,\s]+\s*$")

    @classmethod
    def _postprocess_answer(cls, raw: str) -> tuple[str, bool, str]:
        """统一拒答判定与文末引用行清理；引用列表仍以 hits 为准，不依赖模型解析。"""
        text = raw.strip()
        if text.startswith(("拒答：", "拒答:")):
            return text, True, "insufficient_evidence"
        first_line = text.split("\n", 1)[0].strip()
        if first_line.startswith("无法基于当前证据") or first_line.startswith("当前检索不到"):
            return text, True, "insufficient_evidence"
        # 去掉可选的文末「引用：」行，避免与 citations 重复堆叠
        cleaned = cls._REF_LINE_RE.sub("", text).strip()
        return cleaned, False, ""
