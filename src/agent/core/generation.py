from __future__ import annotations

from ..llm.providers import Message, ModelProvider
from .schemas import GenerationResult, RetrievalHit


class GroundedGenerator:
    """Generate answers with explicit citation constraints."""

    SYSTEM_PROMPT = """你是金融知识助手。回答必须遵循：
1) 只基于给定证据回答，不允许臆测；
2) 回答后给出引用来源编号，如 [1][2]；
3) 若证据不足，明确拒答并说明缺失信息。
"""

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

        context_lines = []
        citations: list[str] = []
        for idx, hit in enumerate(hits, start=1):
            citations.append(f"[{idx}] {hit.source}")
            context_lines.append(f"[{idx}] 来源: {hit.source}\n内容: {hit.text}")
        context = "\n\n".join(context_lines)

        prompt = (
            f"用户问题:\n{question}\n\n"
            f"可用证据:\n{context}\n\n"
            "请给出简洁专业回答，最后单独一行输出引用编号。"
        )
        messages: list[Message] = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        answer = self.model.generate(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ).strip()
        refusal = "无法" in answer and "证据" in answer
        return GenerationResult(answer=answer, citations=citations, refusal=refusal)

