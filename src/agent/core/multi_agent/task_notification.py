from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NotificationType(str, Enum):
    """TaskNotification types for Worker-Coordinator communication."""
    COMPLETED = "completed"                 # Worker execution completed
    ERROR = "error"                          # Execution error
    STATUS_UPDATE = "status_update"         # Status update (e.g., started)


@dataclass
class TaskNotification:
    """Notification sent from Worker to Coordinator."""
    sender: str                      # Worker ID (step_id)
    type: NotificationType            # Notification type
    payload: Any | None = None       # Result data (e.g., rag_hits)
    dependencies_met: list[str] = field(default_factory=list)  # Which dependencies are satisfied

    def to_dict(self) -> dict:
        return {
            "sender": self.sender,
            "type": self.type.value,
            "payload": self.payload,
            "dependencies_met": self.dependencies_met,
        }
