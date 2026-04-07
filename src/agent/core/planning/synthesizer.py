from __future__ import annotations

import logging

from ...config import AgentConfig
from ...llm.providers import ModelProvider
from .plan_schema import PlanArtifact, PlanStepAction
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
    """结果汇总生成器"""

    def __init__(self, model: ModelProvider, config: AgentConfig) -> None:
        self.model = model
        self.config = config

    def synthesize(
        self,
        user_input: str,
        plan: PlanArtifact,
        worker_results: dict[str, WorkerResult],
    ) -> str:
        """根据计划和各 worker 结果生成最终回复"""

        # 构建汇总上下文
        context_parts = []
        for step in plan.steps:
            if step.id in worker_results:
                result = worker_results[step.id]
                status_icon = "✓" if result.is_success() else "✗"
                context_parts.append(
                    f"{status_icon} [{step.action.value.upper()}] {step.detail}\n"
                    f"结果: {result.output if result.is_success() else result.error}"
                )

        context = "\n\n".join(context_parts)

        # 如果所有步骤都失败，降级为直接生成
        if not any(r.is_success() for r in worker_results.values()):
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
            logger.error(f"Synthesizer 生成失败: {exc}")
            return "抱歉，生成回复时出现问题。"
