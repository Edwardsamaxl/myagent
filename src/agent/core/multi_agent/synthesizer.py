from __future__ import annotations

import logging

from ...config import AgentConfig
from ...llm.providers import ModelProvider
from .worker_result import WorkerResult

logger = logging.getLogger(__name__)


SYNTHESIZER_SYSTEM_PROMPT = """你是一个专业的金融知识助手。
根据各步骤的执行结果，汇总生成最终回复。

要求：
1. 优先使用已有数据，避免编造
2. 关键数据需要引用来源
3. 保持回答简洁专业
4. 如果某些步骤失败，在回复中说明
"""


class Synthesizer:
    """Synthesizes final response from all Worker execution results."""

    def __init__(self, model: ModelProvider, config: AgentConfig) -> None:
        self.model = model
        self.config = config

    def synthesize(
        self,
        user_input: str,
        worker_results: dict[str, WorkerResult],
        rag_hits: list[dict] | None = None,
    ) -> str:
        """Generate final response from all Worker results.

        Args:
            user_input: Original user query.
            worker_results: Dict mapping worker_id -> WorkerResult.
            rag_hits: Optional pre-fetched RAG hits.

        Returns:
            Final synthesized response string.
        """
        context_parts = []

        # Add pre-fetched RAG hits if available
        if rag_hits:
            from ...rag.evidence_format import format_evidence_block_from_api_dicts

            evidence_block = format_evidence_block_from_api_dicts(rag_hits)
            context_parts.append(
                f"【预检索的证据】（来自 AgentService 层，已包含以下文档内容，无需重复检索）\n{evidence_block}"
            )

        # Add worker execution results
        for worker_id, result in worker_results.items():
            status_icon = "✓" if result.success else "✗"
            context_parts.append(
                f"{status_icon} [{result.task_type.upper()}] Worker-{worker_id}\n"
                f"结果: {result.output if result.success else result.error}"
            )

        context = "\n\n".join(context_parts)

        # If all workers failed, degrade gracefully
        if not any(r.success for r in worker_results.values()):
            return "抱歉，我无法完成您的请求，请尝试重新描述您的问题。"

        messages: list[dict] = [
            {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"用户问题：{user_input}\n\n执行详情：\n{context}\n\n请生成最终回复。",
            },
        ]

        try:
            return self.model.generate(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        except Exception as exc:
            logger.error(f"Synthesizer generation failed: {exc}")
            return "抱歉，生成回复时出现问题。"
