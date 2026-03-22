from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .schemas import EvalRecord


class EvaluationStore:
    """Store online eval records and return lightweight aggregated metrics."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.file_path.write_text("", encoding="utf-8")

    def append(self, record: EvalRecord) -> None:
        line = json.dumps(asdict(record), ensure_ascii=False)
        with self.file_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def summary(self) -> dict[str, float | int]:
        lines = [ln for ln in self.file_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            return {
                "total_requests": 0,
                "avg_latency_ms": 0.0,
                "avg_estimated_cost_usd": 0.0,
            }
        rows = [json.loads(ln) for ln in lines]
        total = len(rows)
        avg_latency = sum(float(r.get("latency_ms", 0)) for r in rows) / total
        avg_cost = sum(float(r.get("estimated_cost_usd", 0.0)) for r in rows) / total
        return {
            "total_requests": total,
            "avg_latency_ms": round(avg_latency, 2),
            "avg_estimated_cost_usd": round(avg_cost, 6),
        }

