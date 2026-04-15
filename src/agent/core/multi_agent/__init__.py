"""Multi-Agent Coordination Module.

Refactored from src/agent/core/planning/ to separate concerns:
- Coordinator: task assignment and result collection (no planning)
- Worker: task execution with independent trace
- Synthesizer: final response generation
- TaskNotification: inter-component communication
- WorkerResult: execution result data structure
"""

from .coordinator import Coordinator, CoordinatorResult
from .synthesizer import Synthesizer
from .task_notification import NotificationType, TaskNotification
from .worker import Worker, WorkerResult
from .worker_result import WorkerResult as WorkerResultData

__all__ = [
    "Coordinator",
    "CoordinatorResult",
    "NotificationType",
    "Synthesizer",
    "TaskNotification",
    "Worker",
    "WorkerResult",
    "WorkerResultData",
]
