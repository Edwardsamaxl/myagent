from __future__ import annotations

"""Trace 为 JSONL：每行一个 JSON 对象。

字段：`trace_id`（12 位 hex，与同日 `eval_records.jsonl` 中同次问答一致）、
`stage`（见 `TraceStage`）、`message`、`payload`（键值均为字符串）、`timestamp`（UTC ISO）。

关联方式：对一次 RAG 问答，用 `trace_id` 在 `traces.jsonl` 与 `eval_records.jsonl` 中过滤同一请求。
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


class TraceStage:
    """与 `RagAgentService` 写入的 `stage` 取值保持一致，便于检索与仪表盘分组。"""

    INGESTION_START = "ingestion_start"
    INGESTION_DONE = "ingestion_done"
    RAG_ANSWER = "rag_answer"


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

