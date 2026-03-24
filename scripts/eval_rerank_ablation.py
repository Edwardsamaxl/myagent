"""Rerank ablation: A/B/C/D x baseline/hybrid.

Outputs:
- runtime/day3/ablation/A_summary.json
- runtime/day3/ablation/B_summary.json
- runtime/day3/ablation/C_summary.json
- runtime/day3/ablation/D_summary.json
- runtime/day3/ablation/overview.json
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
OUT_DIR = ROOT / "runtime" / "day3" / "ablation"


def _git_rev() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def _load_eval_rows() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in EVAL_SET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
        os.environ["MODEL_NAME"] = "mock-ablation"
    os.environ["EMBEDDING_ENABLED"] = "true" if embedding_enabled else "false"
    os.environ["RETRIEVAL_FUSION_MODE"] = "weighted_sum"
    os.environ["RETRIEVAL_LEXICAL_WEIGHT"] = "0.35"
    os.environ["RETRIEVAL_TFIDF_WEIGHT"] = "0.25"
    os.environ["RETRIEVAL_EMBEDDING_WEIGHT"] = "0.40"
    os.environ["EMBEDDING_TOP_K"] = "12"

    cfg = AgentConfig.from_env()
    model = MockProvider(model_name="mock-ablation") if use_mock else build_model_provider(cfg)
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

    n = len(rows)
    retrieved_zero = sum(1 for r in rows if r["retrieval_hit_count"] == 0)
    top1_hits = sum(1 for r in rows if r["top1_hit"])
    refusals = sum(1 for r in rows if r["refusal"])
    avg_latency = round(sum(r["latency_ms"] for r in rows) / n, 2) if n else 0.0

    cross_year_drift = 0
    cross_year_base = 0
    for r in rows:
        q = r["question"]
        s = r["top1_source"]
        if not s:
            continue
        if "2024" in q:
            cross_year_base += 1
            if "2025" in s:
                cross_year_drift += 1
        elif "2025" in q:
            cross_year_base += 1
            if "2024" in s:
                cross_year_drift += 1

    return {
        "rows": rows,
        "metrics": {
            "sample_size": n,
            "top1_hit_rate": round(top1_hits / n, 4) if n else 0.0,
            "retrieved_zero_rate": round(retrieved_zero / n, 4) if n else 0.0,
            "refusal_rate": round(refusals / n, 4) if n else 0.0,
            "avg_latency_ms": avg_latency,
            "cross_year_drift_rate": round(cross_year_drift / cross_year_base, 4) if cross_year_base else 0.0,
        },
    }


def _diag_label(base: dict[str, Any], hyb: dict[str, Any]) -> str:
    b0 = base["retrieval_hit_count"] == 0
    h0 = hyb["retrieval_hit_count"] == 0
    if b0 and not h0:
        q = base["question"]
        if any(y in q for y in ("2024", "2025")):
            return "year_number_miss"
        return "tokenization_gap"
    if b0 and h0:
        return "expression_gap"
    if not b0 and not h0 and not base["top1_hit"] and hyb["top1_hit"]:
        return "ranking_recover_by_hybrid"
    if not h0 and not hyb["top1_hit"]:
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
            label = _diag_label(b, h)
            out.append(
                {
                    "question": b["question"],
                    "label": label,
                    "baseline_trace_id": b["trace_id"],
                    "hybrid_trace_id": h["trace_id"],
                    "baseline_top3_sources": b["top3_sources"],
                    "hybrid_top3_sources": h["top3_sources"],
                }
            )
    return out


def _run_group(name: str, keyword_on: bool, length_on: bool) -> dict[str, Any]:
    os.environ["RERANK_KEYWORD_BONUS_ENABLED"] = "true" if keyword_on else "false"
    os.environ["RERANK_LENGTH_PENALTY_ENABLED"] = "true" if length_on else "false"
    os.environ["RERANK_METADATA_BONUS_ENABLED"] = "true"
    os.environ["RERANK_NUMERIC_BONUS_ENABLED"] = "true"

    base = _run_variant(embedding_enabled=False)
    hyb = _run_variant(embedding_enabled=True)
    failures = _build_failures(base["rows"], hyb["rows"])

    semantic_drift_n = sum(1 for f in failures if f["label"] == "semantic_drift")
    recover_n = sum(1 for f in failures if f["label"] == "ranking_recover_by_hybrid")

    result = {
        "group": name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(EVAL_SET.relative_to(ROOT)),
        "rerank_flags": {
            "keyword_bonus": keyword_on,
            "length_penalty": length_on,
            "metadata_bonus": True,
            "numeric_bonus": True,
        },
        "baseline": {
            "metrics": {
                **base["metrics"],
                "failure_case_count": len(failures),
                "semantic_drift_count": semantic_drift_n,
                "ranking_recover_by_hybrid_count": recover_n,
            },
            "rows": base["rows"],
        },
        "hybrid": {
            "metrics": {
                **hyb["metrics"],
                "failure_case_count": len(failures),
                "semantic_drift_count": semantic_drift_n,
                "ranking_recover_by_hybrid_count": recover_n,
            },
            "rows": hyb["rows"],
        },
        "failure_cases": failures,
        "typical_failures": failures[:2],
    }
    return result


def main() -> None:
    load_dotenv(ROOT / ".env")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    groups = {
        "A": (True, True),
        "B": (False, True),
        "C": (True, False),
        "D": (False, False),
    }
    outputs: dict[str, dict[str, Any]] = {}
    for g, (kw, lp) in groups.items():
        result = _run_group(g, kw, lp)
        outputs[g] = result
        (OUT_DIR / f"{g}_summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    overview_rows: list[dict[str, Any]] = []
    for g in ("A", "B", "C", "D"):
        r = outputs[g]
        overview_rows.append(
            {
                "group": g,
                "keyword_bonus": r["rerank_flags"]["keyword_bonus"],
                "length_penalty": r["rerank_flags"]["length_penalty"],
                "baseline": r["baseline"]["metrics"],
                "hybrid": r["hybrid"]["metrics"],
            }
        )
    overview = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_version": {
            "git_commit": _git_rev(),
            "script": "scripts/eval_rerank_ablation.py",
        },
        "repro_commands": [
            "python scripts/run_ingest.py",
            "python scripts/eval_rerank_ablation.py",
        ],
        "output_files": [f"runtime/day3/ablation/{g}_summary.json" for g in ("A", "B", "C", "D")],
        "table": overview_rows,
    }
    (OUT_DIR / "overview.json").write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(OUT_DIR), "files": overview["output_files"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
