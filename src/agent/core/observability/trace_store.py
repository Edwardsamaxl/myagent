"""In-memory + JSONL 存储。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Iterator

from .trace_record import TraceRecord


class TraceStore:
    """追踪记录存储：内存缓存 + JSONL 持久化。"""

    DEFAULT_DIR = Path("data/routing_observability/traces")

    def __init__(self, trace_dir: Path | None = None) -> None:
        self._trace_dir = (trace_dir or self.DEFAULT_DIR)
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._memory: dict[str, TraceRecord] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        if not self._trace_dir.exists():
            return
        cutoff = datetime.utcnow() - timedelta(days=7)
        for f in self._trace_dir.glob("*.jsonl"):
            try:
                for line in f.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        ts = obj.get("timestamp", "")
                        if ts:
                            try:
                                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                                if dt < cutoff:
                                    continue
                            except ValueError:
                                pass
                        record = self._dict_to_record(obj)
                        if record:
                            self._memory[record.trace_id] = record
                    except Exception:
                        continue
            except Exception:
                continue

    def _dict_to_record(self, d: dict[str, Any]) -> TraceRecord | None:
        try:
            from .trace_record import LatencyRecord, RouteQuality
            lat = LatencyRecord(**d.get("latency", {}))
            quality = None
            q = d.get("quality")
            if q:
                try:
                    quality = RouteQuality(q)
                except ValueError:
                    pass
            record = TraceRecord(
                trace_id=d.get("trace_id", ""),
                timestamp=d.get("timestamp", ""),
                query=d.get("query", ""),
                history=d.get("history", []),
                route_type=d.get("route_type", ""),
                route_reasoning=d.get("route_reasoning", ""),
                route_confidence=float(d.get("route_confidence", 0.0)),
                selected_tools=d.get("selected_tools", []),
                rag_config=d.get("rag_config"),
                latency=lat,
                retrieval_hits_count=int(d.get("retrieval_hits_count", 0)),
                retrieval_top_score=float(d.get("retrieval_top_score", 0.0)),
                answer=d.get("answer", ""),
                citations=d.get("citations", []),
                quality=quality,
                quality_score=float(d["quality_score"]) if d.get("quality_score") is not None else None,
                quality_reasoning=d.get("quality_reasoning"),
            )
            return record
        except Exception:
            return None

    def append(self, record: TraceRecord) -> None:
        with self._lock:
            self._memory[record.trace_id] = record
            self._flush_one(record)

    def _flush_one(self, record: TraceRecord) -> None:
        ts = record.timestamp[:10]
        trace_file = self._trace_dir / f"traces_{ts}.jsonl"
        with trace_file.open("a", encoding="utf-8") as f:
            f.write(record.to_jsonl() + "\n")

    def get(self, trace_id: str) -> TraceRecord | None:
        with self._lock:
            return self._memory.get(trace_id)

    def query(
        self,
        *,
        route_type: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        min_confidence: float | None = None,
        limit: int = 100,
    ) -> list[TraceRecord]:
        with self._lock:
            results: list[TraceRecord] = []
            for record in self._memory.values():
                if route_type and record.route_type != route_type:
                    continue
                if min_confidence is not None and record.route_confidence < min_confidence:
                    continue
                if start_time:
                    try:
                        ts = datetime.fromisoformat(record.timestamp.replace("Z", "+00:00"))
                        if ts < start_time:
                            continue
                    except ValueError:
                        pass
                if end_time:
                    try:
                        ts = datetime.fromisoformat(record.timestamp.replace("Z", "+00:00"))
                        if ts > end_time:
                            continue
                    except ValueError:
                        pass
                results.append(record)
            results.sort(key=lambda r: r.timestamp, reverse=True)
            return results[:limit]

    def iter_recent(self, days: int = 7) -> Iterator[TraceRecord]:
        cutoff = datetime.utcnow() - timedelta(days=days)
        with self._lock:
            for record in self._memory.values():
                try:
                    ts = datetime.fromisoformat(record.timestamp.replace("Z", "+00:00"))
                    if ts >= cutoff:
                        yield record
                except ValueError:
                    continue

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = len(self._memory)
            if total == 0:
                return {"total": 0}
            confidences = [r.route_confidence for r in self._memory.values() if r.route_confidence > 0]
            route_types: dict[str, int] = {}
            for r in self._memory.values():
                route_types[r.route_type] = route_types.get(r.route_type, 0) + 1
            latencies = [r.latency.total_ms for r in self._memory.values() if r.latency.total_ms > 0]
            return {
                "total": total,
                "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
                "route_types": route_types,
                "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
                "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            }
