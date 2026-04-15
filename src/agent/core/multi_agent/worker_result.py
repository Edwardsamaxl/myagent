from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class WorkerResult:
    """Result of a Worker execution."""
    worker_id: str                                  # Worker identifier
    task_type: str                                  # Task type (e.g., "rag", "calc", "web")
    output: str                                     # Execution result text
    latency_ms: int                                 # Execution time in milliseconds
    success: bool                                   # Whether execution succeeded
    error: str | None = None                       # Error message if failed
    metadata: dict[str, Any] | None = None  # Additional info (e.g., chunk count)

    def to_dict(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "task_type": self.task_type,
            "output": self.output,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }
