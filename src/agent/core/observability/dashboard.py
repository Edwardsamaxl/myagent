"""Dashboard API 端点（Flask Blueprint）。"""

from __future__ import annotations

try:
    from flask import Blueprint, jsonify, request
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

from .trace_store import TraceStore
from .trace_record import RouteQuality

dashboard_bp = Blueprint("observability_dashboard", __name__, url_prefix="/api/observability")

_store: TraceStore | None = None


def init_dashboard(store: TraceStore) -> None:
    global _store
    _store = store


def _require_store() -> TraceStore:
    if _store is None:
        raise RuntimeError("TraceStore not initialized. Call init_dashboard() first.")
    return _store


@dashboard_bp.route("/stats", methods=["GET"])
def get_stats():
    store = _require_store()
    return jsonify(store.stats())


@dashboard_bp.route("/traces", methods=["GET"])
def list_traces():
    store = _require_store()
    limit = min(int(request.args.get("limit", 50)), 200)
    route_type = request.args.get("route_type") or None
    min_conf = request.args.get("min_confidence")
    min_confidence = float(min_conf) if min_conf else None
    records = store.query(route_type=route_type, min_confidence=min_confidence, limit=limit)
    return jsonify({"total": len(records), "traces": [r.to_dict() for r in records]})


@dashboard_bp.route("/traces/<trace_id>", methods=["GET"])
def get_trace(trace_id: str):
    store = _require_store()
    record = store.get(trace_id)
    if record is None:
        return jsonify({"error": "trace not found"}), 404
    return jsonify(record.to_dict())


@dashboard_bp.route("/quality_distribution", methods=["GET"])
def quality_distribution():
    store = _require_store()
    dist: dict[str, int] = {"excellent": 0, "good": 0, "fair": 0, "poor": 0, "unrated": 0}
    for record in store.iter_recent():
        if record.quality is None:
            dist["unrated"] += 1
        else:
            dist[record.quality.value] += 1
    return jsonify(dist)


@dashboard_bp.route("/latency_breakdown", methods=["GET"])
def latency_breakdown():
    store = _require_store()
    rewrite_ms, retrieval_ms, rerank_ms, routing_ms, total_ms = [], [], [], [], []
    for record in store.iter_recent():
        if record.latency.total_ms <= 0:
            continue
        rewrite_ms.append(record.latency.rewrite_ms)
        retrieval_ms.append(record.latency.retrieval_ms)
        rerank_ms.append(record.latency.rerank_ms)
        routing_ms.append(record.latency.routing_ms)
        total_ms.append(record.latency.total_ms)

    def pct(lst: list[int], p: float) -> float:
        if not lst:
            return 0.0
        return round(float(sorted(lst)[int(len(lst) * p)]), 1)

    return jsonify({
        "rewrite_ms": {"p50": pct(rewrite_ms, 0.5), "p95": pct(rewrite_ms, 0.95), "count": len(rewrite_ms)},
        "retrieval_ms": {"p50": pct(retrieval_ms, 0.5), "p95": pct(retrieval_ms, 0.95), "count": len(retrieval_ms)},
        "rerank_ms": {"p50": pct(rerank_ms, 0.5), "p95": pct(rerank_ms, 0.95), "count": len(rerank_ms)},
        "routing_ms": {"p50": pct(routing_ms, 0.5), "p95": pct(routing_ms, 0.95), "count": len(routing_ms)},
        "total_ms": {"p50": pct(total_ms, 0.5), "p95": pct(total_ms, 0.95), "count": len(total_ms)},
    })


@dashboard_bp.route("/confidence_trend", methods=["GET"])
def confidence_trend():
    from datetime import datetime, timedelta
    store = _require_store()
    days = min(int(request.args.get("days", 14)), 30)
    cutoff = datetime.utcnow() - timedelta(days=days)
    daily: dict[str, list[float]] = {}
    for record in store.iter_recent():
        if record.route_confidence <= 0:
            continue
        try:
            day = record.timestamp[:10]
            dt = datetime.fromisoformat(record.timestamp.replace("Z", "+00:00"))
            if dt < cutoff:
                continue
            daily.setdefault(day, []).append(record.route_confidence)
        except ValueError:
            continue
    trend = [
        {"date": day, "avg_confidence": round(sum(vals) / len(vals), 4), "count": len(vals)}
        for day, vals in sorted(daily.items())
    ]
    return jsonify(trend)


@dashboard_bp.route("/routing/distribution", methods=["GET"])
def routing_distribution():
    """路由类型分布统计（按类型和置信度区间）。"""
    store = _require_store()
    from collections import defaultdict
    type_counts: dict[str, int] = defaultdict(int)
    conf_buckets: dict[str, dict[str, int]] = defaultdict(lambda: {"high": 0, "medium": 0, "low": 0})
    for record in store.iter_recent():
        if not record.route_type:
            continue
        type_counts[record.route_type] += 1
        conf = record.route_confidence
        bucket = "high" if conf >= 0.8 else ("medium" if conf >= 0.5 else "low")
        conf_buckets[record.route_type][bucket] += 1
    total = sum(type_counts.values()) or 1
    return jsonify({
        "counts": dict(type_counts),
        "confidence_buckets": dict(conf_buckets),
        "percentages": {k: round(v / total * 100, 1) for k, v in type_counts.items()},
    })


@dashboard_bp.route("/routing/accuracy_trend", methods=["GET"])
def routing_accuracy_trend():
    """路由准确率趋势（按天统计路由正确率）。"""
    from datetime import datetime, timedelta
    store = _require_store()
    days = min(int(request.args.get("days", 14)), 30)
    cutoff = datetime.utcnow() - timedelta(days=days)
    daily: dict[str, dict[str, int]] = {}
    for record in store.iter_recent():
        try:
            dt = datetime.fromisoformat(record.timestamp.replace("Z", "+00:00"))
            if dt < cutoff:
                continue
        except ValueError:
            continue
        day = record.timestamp[:10]
        if day not in daily:
            daily[day] = {"correct": 0, "total": 0}
        daily[day]["total"] += 1
        if record.quality in (RouteQuality.EXCELLENT, RouteQuality.GOOD):
            daily[day]["correct"] += 1
    trend = [
        {
            "date": day,
            "accuracy": round(d["correct"] / d["total"] * 100, 1) if d["total"] > 0 else 0.0,
            "correct": d["correct"],
            "total": d["total"],
        }
        for day, d in sorted(daily.items())
    ]
    return jsonify(trend)


@dashboard_bp.route("/routing/error_analysis", methods=["GET"])
def routing_error_analysis():
    """错误分析（低质量路由的详细分类）。"""
    store = _require_store()
    from collections import defaultdict
    error_by_type: dict[str, int] = defaultdict(int)
    low_confidence_count = 0
    low_confidence_by_type: dict[str, int] = defaultdict(int)
    no_tools_count = 0
    high_latency_count = 0
    for record in store.iter_recent():
        if record.quality in (RouteQuality.FAIR, RouteQuality.POOR):
            error_by_type[record.route_type or "unknown"] += 1
        if 0 < record.route_confidence < 0.5:
            low_confidence_count += 1
            low_confidence_by_type[record.route_type or "unknown"] += 1
        if not record.selected_tools:
            no_tools_count += 1
        if record.latency.total_ms > 5000:
            high_latency_count += 1
    return jsonify({
        "errors_by_route_type": dict(error_by_type),
        "low_confidence_count": low_confidence_count,
        "low_confidence_by_type": dict(low_confidence_by_type),
        "no_tools_selected_count": no_tools_count,
        "high_latency_count": high_latency_count,
    })
