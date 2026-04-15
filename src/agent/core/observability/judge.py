"""LLM-as-judge 答案质量评估。"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from ...llm.providers import ModelProvider
from .trace_record import RouteQuality, TraceRecord


SYSTEM_PROMPT = """你是一个专业的 AI 答案质量评估专家。请评估给定的 AI 答案质量。

## 评估维度
1. 准确性（Accuracy）：答案是否正确回答了用户问题
2. 完整性（Completeness）：答案是否涵盖了问题的所有方面
3. 引用正确性（Citation）：答案中的引用是否来自提供的文档
4. 清晰度（Clarity）：答案是否条理清晰、易于理解

## 评分标准
- excellent (4分)：答案准确、完整，引用恰当，表达清晰
- good (3分)：答案基本正确，有轻微遗漏或表述不清
- fair (2分)：答案有部分错误或重大遗漏
- poor (1分)：答案严重错误或完全不相关

## 输出格式
请以JSON格式输出评估结果：
{
  "quality": "excellent|good|fair|poor",
  "score": 1-4的数值分数,
  "reasoning": "评估理由（中文，100字以内）"
}"""


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


@dataclass
class JudgeResult:
    """Judge 评估结果。"""

    trace_id: str
    quality: RouteQuality
    score: float
    reasoning: str
    latency_ms: int


class JudgeEvaluator:
    """LLM-as-judge 答案质量评估器。"""

    def __init__(
        self,
        model: ModelProvider,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def evaluate(self, record: TraceRecord) -> JudgeResult:
        start_ms = int(time.perf_counter() * 1000)
        citations_text = "\n".join(f"- {c}" for c in record.citations[:5]) if record.citations else "（无引用）"
        user_prompt = f"""## 用户问题
{record.query}

## AI 答案
{record.answer[:1500]}

## 引用文档
{citations_text}"""

        try:
            raw = self.model.generate(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ).strip()
        except Exception as exc:
            elapsed_ms = int(time.perf_counter() * 1000) - start_ms
            return JudgeResult(
                trace_id=record.trace_id,
                quality=RouteQuality.FAIR,
                score=2.0,
                reasoning=f"LLM调用失败: {exc}",
                latency_ms=elapsed_ms,
            )

        obj = _extract_json(raw)
        if not obj:
            elapsed_ms = int(time.perf_counter() * 1000) - start_ms
            return JudgeResult(
                trace_id=record.trace_id,
                quality=RouteQuality.FAIR,
                score=2.0,
                reasoning="LLM输出解析失败，回退到fair",
                latency_ms=elapsed_ms,
            )

        quality_str = obj.get("quality", "fair")
        try:
            quality = RouteQuality(quality_str)
        except ValueError:
            quality = RouteQuality.FAIR

        score = float(obj.get("score", 2.0))
        reasoning = str(obj.get("reasoning", ""))
        elapsed_ms = int(time.perf_counter() * 1000) - start_ms

        return JudgeResult(
            trace_id=record.trace_id,
            quality=quality,
            score=score,
            reasoning=reasoning,
            latency_ms=elapsed_ms,
        )
