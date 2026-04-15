"""路由质量分析器。"""

from __future__ import annotations

from typing import Any

from .trace_record import TraceRecord
from .trace_store import TraceStore


class RoutingAnalyzer:
    """路由质量分析器。"""

    def __init__(self, store: TraceStore) -> None:
        self._store = store

    def analyze_low_confidence(self, threshold: float = 0.5) -> dict[str, Any]:
        low_conf_records = self._store.query(min_confidence=0.0, limit=1000)
        low_conf_records = [r for r in low_conf_records if r.route_confidence < threshold]
        by_type: dict[str, list[TraceRecord]] = {}
        for r in low_conf_records:
            by_type.setdefault(r.route_type, []).append(r)
        suggestions: list[str] = []
        for rtype, records in by_type.items():
            if len(records) >= 3:
                suggestions.append(
                    f"route_type={rtype} 有 {len(records)} 条低置信度(<{threshold})记录，建议优化该类型的判断逻辑。"
                )
        return {
            "threshold": threshold,
            "total_low_confidence": len(low_conf_records),
            "by_route_type": {k: len(v) for k, v in by_type.items()},
            "suggestions": suggestions,
        }

    def analyze_route_type_distribution(self) -> dict[str, Any]:
        records = list(self._store.iter_recent())
        if not records:
            return {"error": "no recent records"}
        type_stats: dict[str, dict[str, Any]] = {}
        for r in records:
            rt = r.route_type or "unknown"
            if rt not in type_stats:
                type_stats[rt] = {"count": 0, "confidences": [], "latencies": [], "qualities": []}
            type_stats[rt]["count"] += 1
            type_stats[rt]["confidences"].append(r.route_confidence)
            type_stats[rt]["latencies"].append(r.latency.total_ms)
            if r.quality_score is not None:
                type_stats[rt]["qualities"].append(r.quality_score)
        result: dict[str, Any] = {}
        for rt, stats in type_stats.items():
            avg_conf = sum(stats["confidences"]) / len(stats["confidences"]) if stats["confidences"] else 0.0
            avg_lat = sum(stats["latencies"]) / len(stats["latencies"]) if stats["latencies"] else 0
            avg_q = sum(stats["qualities"]) / len(stats["qualities"]) if stats["qualities"] else None
            result[rt] = {
                "count": stats["count"],
                "avg_confidence": round(avg_conf, 4),
                "avg_latency_ms": round(avg_lat, 1),
                "avg_quality_score": round(avg_q, 4) if avg_q is not None else None,
            }
        return result

    def analyze_latency_outliers(self, p: float = 0.95) -> dict[str, Any]:
        records = list(self._store.iter_recent())
        latencies = [r.latency.total_ms for r in records if r.latency.total_ms > 0]
        if not latencies:
            return {"error": "no latency data"}
        sorted_lat = sorted(latencies)
        threshold = sorted_lat[int(len(sorted_lat) * p)]
        outliers = [r for r in records if r.latency.total_ms > threshold]
        stage_outliers: dict[str, int] = {"rewrite": 0, "retrieval": 0, "rerank": 0, "routing": 0}
        for r in outliers:
            if r.latency.rewrite_ms > 0 and r.latency.rewrite_ms > r.latency.retrieval_ms * 0.5:
                stage_outliers["rewrite"] += 1
            if r.latency.retrieval_ms > 500:
                stage_outliers["retrieval"] += 1
            if r.latency.rerank_ms > 200:
                stage_outliers["rerank"] += 1
            if r.latency.routing_ms > 1000:
                stage_outliers["routing"] += 1
        return {
            "p_threshold": p,
            "threshold_ms": threshold,
            "outlier_count": len(outliers),
            "outlier_rate": round(len(outliers) / len(records), 4),
            "stage_breakdown": stage_outliers,
        }

    def full_report(self) -> dict[str, Any]:
        return {
            "low_confidence": self.analyze_low_confidence(),
            "route_distribution": self.analyze_route_type_distribution(),
            "latency_outliers": self.analyze_latency_outliers(),
            "store_stats": self._store.stats(),
        }
