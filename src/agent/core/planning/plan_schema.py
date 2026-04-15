from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PlanStepAction(str, Enum):
    """任务步骤动作类型"""
    RAG = "rag"                  # 文档检索
    CALC = "calc"                # 算术计算
    WEB = "web"                  # 网络搜索
    MEMORY = "memory"            # 记忆操作
    SYNTHESIZE = "synthesize"    # 汇总生成
    AGENT_LOOP = "agent_loop"    # 保留：通用工具循环（兼容旧代码）
    CLARIFY = "clarify"          # 澄清问题
    REACT = "react"               # 单步 ReAct 工具调用
    COORDINATOR = "coordinator"   # 多步规划协调


class PlanStepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class PlanStep:
    id: str                           # 步骤 ID，如 "s1"
    action: PlanStepAction            # 动作类型
    detail: str                       # 详细描述
    status: PlanStepStatus = PlanStepStatus.PENDING
    depends_on: list[str] = field(default_factory=list)       # 依赖的步骤 ID
    parallel_with: list[str] = field(default_factory=list)    # 可并行的步骤 ID


@dataclass
class PlanArtifact:
    plan_id: str
    goal: str
    steps: list[PlanStep]
    version: int = 1

    def summary(self) -> str:
        parts = [f"{s.id}:{s.action.value}:{s.status.value}" for s in self.steps]
        return "; ".join(parts)
