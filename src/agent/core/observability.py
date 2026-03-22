from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class TraceEvent:
    trace_id: str
    stage: str
    message: str
    payload: dict[str, str] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class TraceLogger:
    def __init__(self, trace_file: Path) -> None:
        self.trace_file = trace_file
        self.trace_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.trace_file.exists():
            self.trace_file.write_text("", encoding="utf-8")

    @staticmethod
    def new_trace_id() -> str:
        return uuid.uuid4().hex[:12]

    def log(self, event: TraceEvent) -> None:
        line = json.dumps(
            {
                "trace_id": event.trace_id,
                "stage": event.stage,
                "message": event.message,
                "payload": event.payload,
                "timestamp": event.timestamp,
            },
            ensure_ascii=False,
        )
        with self.trace_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

