from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class EvalRecord:
    """在线请求一条一行写入 `eval_records.jsonl`；与离线批评估行字段对齐（见 evaluation 模块）。"""

    trace_id: str
    question: str
    answer: str
    references: list[str]
    latency_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    run_mode: str = "online"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
