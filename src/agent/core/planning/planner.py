from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from ..dialogue.intent_schema import IntentKind, IntentResult
from .plan_schema import PlanArtifact, PlanStep, PlanStepAction, PlanStepStatus

if TYPE_CHECKING:
    from ..router import Route, RouterDecision


def build_turn_plan(
    *,
    intent: IntentResult,
    rag_will_run: bool,
    router_decision: RouterDecision | None = None,
) -> PlanArtifact:
    """基于实际路由决策生成计划。

    Args:
        intent: 意图分类结果
        rag_will_run: 是否会运行 RAG
        router_decision: AgentRouter 的决策结果（可选，用于生成准确的 plan）
    """
    pid = f"p_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"
    goal = intent.normalized_query or ""
    steps: list[PlanStep] = []

    # 无 router_decision 时退化为原有逻辑（向后兼容）
    if router_decision is None:
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

    # 基于 router_decision.route 生成对应 plan
    route = router_decision.route

    if route.value == "clarify":
        steps.append(
            PlanStep(
                "s1",
                PlanStepAction.CLARIFY,
                router_decision.reasoning,
                PlanStepStatus.DONE,
            )
        )
    elif route.value == "coordinator":
        steps.append(
            PlanStep(
                "s1",
                PlanStepAction.COORDINATOR,
                f"多步规划（预估{router_decision.estimated_steps}步）",
                PlanStepStatus.PENDING,
            )
        )
    elif route.value == "react":
        if rag_will_run:
            steps.append(
                PlanStep(
                    "s1",
                    PlanStepAction.RAG,
                    "文档检索 + 重排",
                    PlanStepStatus.PENDING,
                )
            )
        sid = f"s{len(steps) + 1}" if steps else "s1"
        steps.append(
            PlanStep(
                sid,
                PlanStepAction.REACT,
                router_decision.reasoning,
                PlanStepStatus.PENDING,
            )
        )
    else:
        # 兜底：不应该到达这里
        steps.append(
            PlanStep("s1", PlanStepAction.AGENT_LOOP, "工具循环", PlanStepStatus.PENDING)
        )

    return PlanArtifact(pid, goal, steps)
