from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .plan_schema import PlanStepAction


@dataclass
class WorkerResult:
    """Worker 执行结果"""
    step_id: str                                    # 对应 PlanStep.id
    action: PlanStepAction                          # 动作类型
    status: Literal["pending", "running", "success", "failed"]
    output: str                                     # 执行结果文本
    error: str | None = None                       # 错误信息
    metadata: dict | None = None                   # 额外信息（如检索的 chunk 数量）

    def is_success(self) -> bool:
        return self.status == "success"
