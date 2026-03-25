from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PlanStepAction(str, Enum):
    RAG = "rag"
    AGENT_LOOP = "agent_loop"
    CLARIFY = "clarify"


class PlanStepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class PlanStep:
    id: str
    action: PlanStepAction
    detail: str
    status: PlanStepStatus = PlanStepStatus.PENDING
    depends_on: list[str] = field(default_factory=list)


@dataclass
class PlanArtifact:
    plan_id: str
    goal: str
    steps: list[PlanStep]
    version: int = 1

    def summary(self) -> str:
        parts = [f"{s.id}:{s.action.value}:{s.status.value}" for s in self.steps]
        return "; ".join(parts)
