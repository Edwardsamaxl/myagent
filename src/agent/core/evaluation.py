from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .schemas import EvalRecord

# 在线 / 离线聚合共用同一套键名（口径统一）
METRIC_TOTAL_REQUESTS = "total_requests"
METRIC_AVG_LATENCY_MS = "avg_latency_ms"
METRIC_AVG_ESTIMATED_COST_USD = "avg_estimated_cost_usd"
METRIC_SUBSTRING_MATCH_RATE = "substring_match_rate"


def _row_has_substring_match(row: dict[str, Any]) -> bool | None:
    v = row.get("substring_match")
    if v is None:
        return None
    return bool(v)


def aggregate_eval_rows(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """从 JSON 行字典列表聚合指标；在线 `eval_records.jsonl` 与离线批评估输出共用此函数。"""
    if not rows:
        return {
            METRIC_TOTAL_REQUESTS: 0,
            METRIC_AVG_LATENCY_MS: 0.0,
            METRIC_AVG_ESTIMATED_COST_USD: 0.0,
            METRIC_SUBSTRING_MATCH_RATE: None,
        }
    total = len(rows)
    avg_latency = sum(float(r.get("latency_ms", 0)) for r in rows) / total
    avg_cost = sum(float(r.get("estimated_cost_usd", 0.0)) for r in rows) / total
    match_flags = [_row_has_substring_match(r) for r in rows]
    explicit = [m for m in match_flags if m is not None]
    match_rate: float | None = None
    if explicit:
        match_rate = round(sum(1 for m in explicit if m) / len(explicit), 4)
    return {
        METRIC_TOTAL_REQUESTS: total,
        METRIC_AVG_LATENCY_MS: round(avg_latency, 2),
        METRIC_AVG_ESTIMATED_COST_USD: round(avg_cost, 6),
        METRIC_SUBSTRING_MATCH_RATE: match_rate,
    }


def load_eval_rows_from_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 评估记录（每行一个对象）。"""
    if not path.exists():
        return []
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


def substring_match(expected: str, actual: str) -> bool:
    """离线指标：参考答案是否以子串形式出现在模型答案中（去空白后比较）。"""
    e = expected.replace(" ", "").replace("\n", "").strip()
    a = actual.replace(" ", "").replace("\n", "").strip()
    if not e:
        return False
    return e in a


class EvaluationStore:
    """在线评估记录持久化；聚合指标与 `aggregate_eval_rows` 口径一致。"""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.file_path.write_text("", encoding="utf-8")

    def append(self, record: EvalRecord) -> None:
        line = json.dumps(asdict(record), ensure_ascii=False)
        with self.file_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def summary(self) -> dict[str, float | int | None]:
        rows = load_eval_rows_from_jsonl(self.file_path)
        return aggregate_eval_rows(rows)

