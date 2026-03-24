"""Day3 baseline vs hybrid A/B evaluation.

Outputs:
- runtime/day3/embedding_ab_report.quick.json
- runtime/day3/summary.json
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent.application.rag_agent_service import RagAgentService
from agent.config import AgentConfig
from agent.llm.providers import MockProvider, build_model_provider

EVAL_SET = ROOT / "data" / "eval" / "week1_eval_set.jsonl"
DATA_RAW = ROOT / "data" / "raw" / "finance"
OUT_DIR = ROOT / "runtime" / "day3"
OUT_QUICK = OUT_DIR / "embedding_ab_report.quick.json"
OUT_SUMMARY = OUT_DIR / "summary.json"


def _git_rev() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
        return out
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
    top1 = str(retrieval_hits[0].get("source", ""))
    return _normalize_source(expected_source) == _normalize_source(top1)


def _run_variant(name: str, *, embedding_enabled: bool) -> dict[str, Any]:
    use_mock = os.getenv("EVAL_USE_MOCK", "false").strip().lower() in {"1", "true", "yes", "on"}
    if use_mock:
        os.environ["MODEL_PROVIDER"] = "mock"
        os.environ["MODEL_NAME"] = "mock-ab"
    os.environ["EMBEDDING_ENABLED"] = "true" if embedding_enabled else "false"
    # keep weights explicit for reproducibility
    os.environ["RETRIEVAL_FUSION_MODE"] = "weighted_sum"
    os.environ["RETRIEVAL_LEXICAL_WEIGHT"] = "0.35"
    os.environ["RETRIEVAL_TFIDF_WEIGHT"] = "0.25"
    os.environ["RETRIEVAL_EMBEDDING_WEIGHT"] = "0.40"
    os.environ["EMBEDDING_TOP_K"] = "12"

    config = AgentConfig.from_env()
    model = MockProvider(model_name="mock-ab") if use_mock else build_model_provider(config)
    service = RagAgentService(config=config, model=model)

    for doc_id, source, content in _load_raw_docs():
        service.ingest_document(doc_id=doc_id, source=source, content=content)

    eval_rows = _load_eval_rows()
    rows: list[dict[str, Any]] = []
    for item in eval_rows:
        q = item["question"]
        expected_source = str(item.get("source", ""))
        result = service.answer(q, append_to_eval_store=False)
        retrieval_hits = list(result.get("retrieval_hits", []))
        row = {
            "question": q,
            "expected_answer": item.get("expected_answer", ""),
            "expected_source": expected_source,
            "trace_id": result.get("trace_id"),
            "latency_ms": int(result.get("latency_ms", 0)),
            "refusal": bool(result.get("refusal", False)),
            "reason": str(result.get("reason", "")),
            "retrieval_hit_count": len(retrieval_hits),
            "top1_hit": _top1_hit(expected_source, retrieval_hits),
            "top3_sources": [str(h.get("source", "")) for h in retrieval_hits[:3]],
            "top3_previews": [str(h.get("text_preview", "")) for h in retrieval_hits[:3]],
        }
        rows.append(row)

    total = len(rows)
    retrieved_zero = sum(1 for r in rows if r["retrieval_hit_count"] == 0)
    top1_hits = sum(1 for r in rows if r["top1_hit"])
    refusals = sum(1 for r in rows if r["refusal"])
    avg_latency = round(sum(r["latency_ms"] for r in rows) / total, 2) if total else 0.0
    metrics = {
        "sample_size": total,
        "retrieved_zero_rate": round(retrieved_zero / total, 4) if total else 0.0,
        "top1_hit_rate": round(top1_hits / total, 4) if total else 0.0,
        "refusal_rate": round(refusals / total, 4) if total else 0.0,
        "avg_latency_ms": avg_latency,
    }
    return {
        "name": name,
        "config": {
            "embedding_enabled": embedding_enabled,
            "retrieval_fusion_mode": config.retrieval_fusion_mode,
            "retrieval_lexical_weight": config.retrieval_lexical_weight,
            "retrieval_tfidf_weight": config.retrieval_tfidf_weight,
            "retrieval_embedding_weight": config.retrieval_embedding_weight,
            "embedding_top_k": config.embedding_top_k,
            "embedding_provider": config.embedding_provider or config.model_provider,
            "embedding_model": config.embedding_model,
        },
        "metrics": metrics,
        "rows": rows,
    }


def _label_when_baseline_zero(base: dict[str, Any]) -> str:
    q = str(base["question"])
    if any(ch.isdigit() for ch in q) and any(s in q for s in ("2024", "2025")):
        return "year_number_miss"
    if "？" in q or "多少" in q or "是谁" in q:
        return "expression_gap"
    return "tokenization_gap"


def _diag_label(base: dict[str, Any], hyb: dict[str, Any]) -> str:
    b0 = base["retrieval_hit_count"] == 0
    h0 = hyb["retrieval_hit_count"] == 0
    if b0 and not h0:
        return _label_when_baseline_zero(base)
    if b0 and h0:
        return "evidence_missing_or_tokenization_gap"
    if not b0 and not h0 and not base["top1_hit"] and hyb["top1_hit"]:
        return "ranking_recover_by_hybrid"
    if not h0 and not hyb["top1_hit"]:
        return "semantic_drift"
    return "other"


def build_failures(baseline_rows: list[dict[str, Any]], hybrid_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_q_h = {r["question"]: r for r in hybrid_rows}
    failures: list[dict[str, Any]] = []
    for b in baseline_rows:
        h = by_q_h.get(b["question"])
        if not h:
            continue
        if b["retrieval_hit_count"] == 0 or not b["top1_hit"] or b["refusal"]:
            failures.append(
                {
                    "question": b["question"],
                    "label": _diag_label(b, h),
                    "baseline_trace_id": b["trace_id"],
                    "hybrid_trace_id": h["trace_id"],
                    "baseline": {
                        "retrieval_hit_count": b["retrieval_hit_count"],
                        "top1_hit": b["top1_hit"],
                        "refusal": b["refusal"],
                        "reason": b["reason"],
                        "top3_sources": b["top3_sources"],
                    },
                    "hybrid": {
                        "retrieval_hit_count": h["retrieval_hit_count"],
                        "top1_hit": h["top1_hit"],
                        "refusal": h["refusal"],
                        "reason": h["reason"],
                        "top3_sources": h["top3_sources"],
                    },
                }
            )
    return failures


def main() -> None:
    load_dotenv(ROOT / ".env")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline = _run_variant("baseline", embedding_enabled=False)
    hybrid = _run_variant("hybrid", embedding_enabled=True)
    failures = build_failures(baseline["rows"], hybrid["rows"])
    failures = failures[: max(5, len(failures))]

    quick = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(EVAL_SET.relative_to(ROOT)),
        "config_version": {
            "git_commit": _git_rev(),
            "script": "scripts/eval_embedding_ab.py",
        },
        "variants": {
            "baseline": {
                "config": baseline["config"],
                "metrics": baseline["metrics"],
            },
            "hybrid": {
                "config": hybrid["config"],
                "metrics": hybrid["metrics"],
            },
        },
        "failure_cases": failures,
        "files": {
            "quick_report": str(OUT_QUICK.relative_to(ROOT)),
            "summary": str(OUT_SUMMARY.relative_to(ROOT)),
        },
    }
    OUT_QUICK.write_text(json.dumps(quick, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "generated_at": quick["generated_at"],
        "dataset": quick["dataset"],
        "repro_commands": [
            "python scripts/run_ingest.py",
            "python scripts/eval_embedding_ab.py",
        ],
        "config_version": quick["config_version"],
        "result_files": quick["files"],
        "metrics": {
            "baseline": baseline["metrics"],
            "hybrid": hybrid["metrics"],
        },
        "failure_case_count": len(failures),
        "failure_case_preview": failures[:5],
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"quick": str(OUT_QUICK), "summary": str(OUT_SUMMARY)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
