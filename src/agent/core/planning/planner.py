from __future__ import annotations

import uuid
from datetime import datetime

from ..dialogue.intent_schema import IntentKind, IntentResult
from .plan_schema import PlanArtifact, PlanStep, PlanStepAction, PlanStepStatus


def build_turn_plan(
    *,
    intent: IntentResult,
    rag_will_run: bool,
) -> PlanArtifact:
    """P0：单轮计划仅用于可观测摘要，不改变 max_steps 语义。"""
    pid = f"p_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"
    goal = intent.normalized_query or ""
    steps: list[PlanStep] = []

    if intent.intent == IntentKind.AMBIGUOUS:
        steps.append(
            PlanStep("s1", PlanStepAction.CLARIFY, "返回澄清问句", PlanStepStatus.DONE)
        )
        return PlanArtifact(pid, goal, steps)

    if rag_will_run:
        steps.append(
            PlanStep(
                "s1",
                PlanStepAction.RAG,
                "检索+重排（同路径内 grounded 生成供 trace；对话终答由 Agent 用证据辅助）",
                PlanStepStatus.PENDING,
            )
        )
    sid = "s2" if steps else "s1"
    steps.append(
        PlanStep(
            sid,
            PlanStepAction.AGENT_LOOP,
            "SimpleAgent 工具循环或自然语言终答",
            PlanStepStatus.PENDING,
        )
    )
    return PlanArtifact(pid, goal, steps)
