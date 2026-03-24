"""Evaluate sparse channel ablation: tfidf vs bm25 vs tfidf_bm25.

Outputs:
- runtime/day3/bm25_ablation/A_tfidf_summary.json
- runtime/day3/bm25_ablation/B_bm25_summary.json
- runtime/day3/bm25_ablation/C_tfidf_bm25_summary.json
- runtime/day3/bm25_ablation/D_bm25_no_keyword_summary.json
- runtime/day3/bm25_ablation/overview.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent.application.rag_agent_service import RagAgentService
from agent.config import AgentConfig
from agent.llm.providers import MockProvider, build_model_provider

EVAL_SET = ROOT / "data" / "eval" / "week1_eval_set.jsonl"
DATA_RAW = ROOT / "data" / "raw" / "finance"
OUT_DIR = ROOT / "runtime" / "day3" / "bm25_ablation"


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _load_eval_rows() -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in EVAL_SET.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _load_raw_docs() -> list[tuple[str, str, str]]:
    docs: list[tuple[str, str, str]] = []
    for company_dir in sorted(DATA_RAW.iterdir()):
        if not company_dir.is_dir():
            continue
        for ext in ("*.md", "*.txt"):
            for p in sorted(company_dir.glob(ext)):
                docs.append((p.stem, f"{company_dir.name}/{p.name}", p.read_text(encoding="utf-8")))
    return docs


def _normalize_source(src: str) -> str:
    return src.replace(".txt", ".md").strip()


def _top1_hit(expected_source: str, retrieval_hits: list[dict[str, Any]]) -> bool:
    if not retrieval_hits:
        return False
    return _normalize_source(expected_source) == _normalize_source(str(retrieval_hits[0].get("source", "")))


def _run_variant(*, embedding_enabled: bool) -> dict[str, Any]:
    use_mock = os.getenv("EVAL_USE_MOCK", "false").strip().lower() in {"1", "true", "yes", "on"}
    if use_mock:
        os.environ["MODEL_PROVIDER"] = "mock"
        os.environ["MODEL_NAME"] = "mock-bm25-ablation"
    os.environ["EMBEDDING_ENABLED"] = "true" if embedding_enabled else "false"
    os.environ["RETRIEVAL_FUSION_MODE"] = "weighted_sum"
    os.environ["RETRIEVAL_LEXICAL_WEIGHT"] = "0.35"
    os.environ["RETRIEVAL_TFIDF_WEIGHT"] = "0.25"
    os.environ["RETRIEVAL_EMBEDDING_WEIGHT"] = "0.40"
    os.environ["EMBEDDING_TOP_K"] = "12"

    cfg = AgentConfig.from_env()
    model = MockProvider(model_name="mock-bm25-ablation") if use_mock else build_model_provider(cfg)
    svc = RagAgentService(config=cfg, model=model)
    for doc_id, source, content in _load_raw_docs():
        svc.ingest_document(doc_id=doc_id, source=source, content=content)

    rows: list[dict[str, Any]] = []
    for item in _load_eval_rows():
        q = item["question"]
        expected_source = str(item.get("source", ""))
        res = svc.answer(q, append_to_eval_store=False)
        hits = list(res.get("retrieval_hits", []))
        rows.append(
            {
                "question": q,
                "trace_id": str(res.get("trace_id", "")),
                "latency_ms": int(res.get("latency_ms", 0)),
                "refusal": bool(res.get("refusal", False)),
                "reason": str(res.get("reason", "")),
                "retrieval_hit_count": len(hits),
                "top1_source": str(hits[0].get("source", "")) if hits else "",
                "top1_hit": _top1_hit(expected_source, hits),
                "top3_sources": [str(h.get("source", "")) for h in hits[:3]],
            }
        )
    return {"rows": rows, "metrics": _calc_metrics(rows)}


def _calc_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    n = len(rows)
    retrieved_zero = sum(1 for r in rows if r["retrieval_hit_count"] == 0)
    top1_hits = sum(1 for r in rows if r["top1_hit"])
    refusals = sum(1 for r in rows if r["refusal"])
    avg_latency = round(sum(r["latency_ms"] for r in rows) / n, 2) if n else 0.0
    cross_drift = 0
    cross_base = 0
    for r in rows:
        q = r["question"]
        s = r["top1_source"]
        if not s:
            continue
        if "2024" in q:
            cross_base += 1
            if "2025" in s:
                cross_drift += 1
        elif "2025" in q:
            cross_base += 1
            if "2024" in s:
                cross_drift += 1
    return {
        "sample_size": n,
        "top1_hit_rate": round(top1_hits / n, 4) if n else 0.0,
        "retrieved_zero_rate": round(retrieved_zero / n, 4) if n else 0.0,
        "refusal_rate": round(refusals / n, 4) if n else 0.0,
        "avg_latency_ms": avg_latency,
        "cross_year_drift_rate": round(cross_drift / cross_base, 4) if cross_base else 0.0,
    }


def _diag_label(b: dict[str, Any], h: dict[str, Any]) -> str:
    b0 = b["retrieval_hit_count"] == 0
    h0 = h["retrieval_hit_count"] == 0
    if b0 and not h0:
        if any(y in b["question"] for y in ("2024", "2025")):
            return "year_number_miss"
        return "tokenization_gap"
    if b0 and h0:
        return "expression_gap"
    if not b0 and not h0 and not b["top1_hit"] and h["top1_hit"]:
        return "ranking_recover_by_hybrid"
    if not h0 and not h["top1_hit"]:
        return "semantic_drift"
    return "other"


def _build_failures(base_rows: list[dict[str, Any]], hyb_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_q_h = {r["question"]: r for r in hyb_rows}
    out: list[dict[str, Any]] = []
    for b in base_rows:
        h = by_q_h.get(b["question"])
        if not h:
            continue
        if b["retrieval_hit_count"] == 0 or not b["top1_hit"] or b["refusal"]:
            out.append(
                {
                    "question": b["question"],
                    "label": _diag_label(b, h),
                    "baseline_trace_id": b["trace_id"],
                    "hybrid_trace_id": h["trace_id"],
                    "baseline_top3_sources": b["top3_sources"],
                    "hybrid_top3_sources": h["top3_sources"],
                }
            )
    return out


def _add_failure_metrics(metrics: dict[str, Any], failures: list[dict[str, Any]]) -> dict[str, Any]:
    m = dict(metrics)
    m["failure_case_count"] = len(failures)
    m["semantic_drift_count"] = sum(1 for f in failures if f["label"] == "semantic_drift")
    m["ranking_recover_by_hybrid_count"] = sum(1 for f in failures if f["label"] == "ranking_recover_by_hybrid")
    return m


def _run_group(
    name: str,
    *,
    sparse_mode: str,
    keyword_bonus: bool,
    k1: float,
    b: float,
    alpha: float = 0.5,
) -> dict[str, Any]:
    os.environ["SPARSE_MODE"] = sparse_mode
    os.environ["BM25_K1"] = str(k1)
    os.environ["BM25_B"] = str(b)
    os.environ["TFIDF_BM25_ALPHA"] = str(alpha)
    os.environ["RERANK_KEYWORD_BONUS_ENABLED"] = "true" if keyword_bonus else "false"
    os.environ["RERANK_LENGTH_PENALTY_ENABLED"] = "true"
    os.environ["RERANK_METADATA_BONUS_ENABLED"] = "true"
    os.environ["RERANK_NUMERIC_BONUS_ENABLED"] = "true"

    base = _run_variant(embedding_enabled=False)
    hyb = _run_variant(embedding_enabled=True)
    failures = _build_failures(base["rows"], hyb["rows"])

    return {
        "group": name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(EVAL_SET.relative_to(ROOT)),
        "config": {
            "sparse_mode": sparse_mode,
            "keyword_bonus": keyword_bonus,
            "length_penalty": True,
            "metadata_bonus": True,
            "numeric_bonus": True,
            "bm25_k1": k1,
            "bm25_b": b,
            "tfidf_bm25_alpha": alpha,
        },
        "baseline": {
            "metrics": _add_failure_metrics(base["metrics"], failures),
            "rows": base["rows"],
        },
        "hybrid": {
            "metrics": _add_failure_metrics(hyb["metrics"], failures),
            "rows": hyb["rows"],
        },
        "failure_cases": failures,
        "typical_failures": failures[:2],
    }


def _score_for_selection(res: dict[str, Any]) -> tuple[float, float]:
    # primary: hybrid top1, secondary: lower hybrid semantic drift
    h = res["hybrid"]["metrics"]
    return float(h["top1_hit_rate"]), -float(h["semantic_drift_count"])


def _pick_best_sparse(
    *,
    sparse_mode: str,
    keyword_bonus: bool,
    alpha: float = 0.5,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for k1 in (1.2, 1.5, 1.8):
        for b in (0.5, 0.75, 0.9):
            candidates.append(
                _run_group(
                    "tmp",
                    sparse_mode=sparse_mode,
                    keyword_bonus=keyword_bonus,
                    k1=k1,
                    b=b,
                    alpha=alpha,
                )
            )
    best = max(candidates, key=_score_for_selection)
    best["grid_candidates"] = [
        {
            "k1": c["config"]["bm25_k1"],
            "b": c["config"]["bm25_b"],
            "baseline_top1_hit_rate": c["baseline"]["metrics"]["top1_hit_rate"],
            "hybrid_top1_hit_rate": c["hybrid"]["metrics"]["top1_hit_rate"],
            "hybrid_semantic_drift_count": c["hybrid"]["metrics"]["semantic_drift_count"],
        }
        for c in candidates
    ]
    return best


def main() -> None:
    load_dotenv(ROOT / ".env")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # A: tfidf baseline
    a = _run_group("A", sparse_mode="tfidf", keyword_bonus=True, k1=1.5, b=0.75)
    # B: bm25 best k1,b
    b_best = _pick_best_sparse(sparse_mode="bm25", keyword_bonus=True)
    b_best["group"] = "B"
    # C: tfidf_bm25 best k1,b (alpha fixed 0.5)
    c_best = _pick_best_sparse(sparse_mode="tfidf_bm25", keyword_bonus=True, alpha=0.5)
    c_best["group"] = "C"
    # D: bm25 no keyword (best k1,b)
    d_best = _pick_best_sparse(sparse_mode="bm25", keyword_bonus=False)
    d_best["group"] = "D"

    out_map = {
        "A_tfidf_summary.json": a,
        "B_bm25_summary.json": b_best,
        "C_tfidf_bm25_summary.json": c_best,
        "D_bm25_no_keyword_summary.json": d_best,
    }
    for fn, payload in out_map.items():
        (OUT_DIR / fn).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    table = []
    for g, fn in (
        ("A", "A_tfidf_summary.json"),
        ("B", "B_bm25_summary.json"),
        ("C", "C_tfidf_bm25_summary.json"),
        ("D", "D_bm25_no_keyword_summary.json"),
    ):
        payload = out_map[fn]
        table.append(
            {
                "group": g,
                "file": f"runtime/day3/bm25_ablation/{fn}",
                "config": payload["config"],
                "baseline": payload["baseline"]["metrics"],
                "hybrid": payload["hybrid"]["metrics"],
            }
        )

    overview = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_version": {
            "git_commit": _git_rev(),
            "script": "scripts/eval_bm25_ablation.py",
        },
        "repro_commands": [
            "python scripts/run_ingest.py",
            "python scripts/eval_bm25_ablation.py",
        ],
        "output_files": [f"runtime/day3/bm25_ablation/{k}" for k in out_map.keys()],
        "table": table,
    }
    (OUT_DIR / "overview.json").write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(OUT_DIR), "files": overview["output_files"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
