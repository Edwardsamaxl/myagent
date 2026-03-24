"""Evaluate generation closure: baseline(strict off) vs current(strict on).

Outputs:
- runtime/day3/summary.json                (updated summary)
- runtime/day3/generation_failure_diagnosis.json
- runtime/day3/generation_samples_10.json
"""

from __future__ import annotations

import json
import os
import re
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
from agent.llm.providers import build_model_provider

EVAL_SET = ROOT / "data" / "eval" / "week1_eval_set.jsonl"
DATA_RAW = ROOT / "data" / "raw" / "finance"
OUT_DIR = ROOT / "runtime" / "day3"
OUT_SUMMARY = OUT_DIR / "summary.json"
OUT_DIAG = OUT_DIR / "generation_failure_diagnosis.json"
OUT_SAMPLES = OUT_DIR / "generation_samples_10.json"


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _normalize_source(src: str) -> str:
    return str(src).replace(".txt", ".md").strip()


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


def _contains_citation(answer: str) -> bool:
    return bool(re.search(r"\[\d+\]", answer))


def _citation_ok(answer: str, citations: list[str], refusal: bool) -> bool:
    if refusal:
        return False
    return bool(citations) and _contains_citation(answer)


def _top1_hit(expected_source: str, retrieval_hits: list[dict[str, Any]]) -> bool:
    if not retrieval_hits:
        return False
    return _normalize_source(expected_source) == _normalize_source(retrieval_hits[0].get("source", ""))


def _label_row(row: dict[str, Any]) -> str:
    if row["retrieval_hit_count"] == 0:
        return "retrieved_zero"
    if row["refusal"]:
        if row["reason"] == "citation_missing":
            return "citation_missing"
        return "insufficient_evidence"
    if not row["top1_hit"]:
        return "semantic_drift"
    if not row["citation_ok"]:
        return "citation_format_gap"
    return "success"


def _run_variant(name: str, strict_policy: bool) -> dict[str, Any]:
    os.environ["GENERATION_STRICT_POLICY"] = "1" if strict_policy else "0"
    # keep retrieval/rerank defaults; only test generation strictness
    cfg = AgentConfig.from_env()
    model = build_model_provider(cfg)
    service = RagAgentService(config=cfg, model=model)

    for doc_id, source, content in _load_raw_docs():
        service.ingest_document(doc_id=doc_id, source=source, content=content)

    rows: list[dict[str, Any]] = []
    for item in _load_eval_rows():
        q = item["question"]
        expected_source = str(item.get("source", ""))
        res = service.answer(q, append_to_eval_store=False)
        hits = list(res.get("retrieval_hits", []))
        answer = str(res.get("answer", ""))
        citations = list(res.get("citations", []))
        refusal = bool(res.get("refusal", False))
        row = {
            "question": q,
            "expected_source": expected_source,
            "trace_id": str(res.get("trace_id", "")),
            "latency_ms": int(res.get("latency_ms", 0)),
            "retrieval_hit_count": len(hits),
            "top1_hit": _top1_hit(expected_source, hits),
            "top3_sources": [str(h.get("source", "")) for h in hits[:3]],
            "refusal": refusal,
            "reason": str(res.get("reason", "")),
            "citation_ok": _citation_ok(answer=answer, citations=citations, refusal=refusal),
            "answer_preview": answer[:260],
        }
        row["label"] = _label_row(row)
        rows.append(row)

    n = len(rows)
    top1_hit_rate = round(sum(1 for r in rows if r["top1_hit"]) / n, 4) if n else 0.0
    citation_ok_rate = round(sum(1 for r in rows if r["citation_ok"]) / n, 4) if n else 0.0
    refusal_rate = round(sum(1 for r in rows if r["refusal"]) / n, 4) if n else 0.0
    avg_latency_ms = round(sum(r["latency_ms"] for r in rows) / n, 2) if n else 0.0
    semantic_drift_count = sum(1 for r in rows if r["label"] == "semantic_drift")
    metrics = {
        "sample_size": n,
        "top1_hit_rate": top1_hit_rate,
        "citation_ok": citation_ok_rate,
        "refusal_rate": refusal_rate,
        "avg_latency_ms": avg_latency_ms,
        "semantic_drift_count": semantic_drift_count,
    }
    return {
        "name": name,
        "strict_policy": strict_policy,
        "metrics": metrics,
        "rows": rows,
    }


def _pick_samples(current_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    success = [r for r in current_rows if r["label"] == "success"][:5]
    fail = [r for r in current_rows if r["label"] != "success"][:5]
    return success + fail


def main() -> None:
    load_dotenv(ROOT / ".env")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline = _run_variant("baseline_generation_policy_off", strict_policy=False)
    current = _run_variant("current_generation_policy_on", strict_policy=True)

    samples_10 = _pick_samples(current["rows"])
    OUT_SAMPLES.write_text(json.dumps(samples_10, ensure_ascii=False, indent=2), encoding="utf-8")

    failures = [r for r in current["rows"] if r["label"] != "success"]
    OUT_DIAG.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "failure_count": len(failures),
                "failures": failures,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(EVAL_SET.relative_to(ROOT)),
        "repro_commands": [
            "python scripts/run_ingest.py",
            "python scripts/eval_generation_closure.py",
        ],
        "config_version": {
            "git_commit": _git_rev(),
            "script": "scripts/eval_generation_closure.py",
            "generation_strict_policy_baseline": "0",
            "generation_strict_policy_current": "1",
        },
        "result_files": {
            "summary": str(OUT_SUMMARY.relative_to(ROOT)),
            "failure_diagnosis": str(OUT_DIAG.relative_to(ROOT)),
            "samples_10": str(OUT_SAMPLES.relative_to(ROOT)),
        },
        "metrics": {
            "baseline": baseline["metrics"],
            "current": current["metrics"],
        },
        "sample_10_count": len(samples_10),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(OUT_SUMMARY), "diag": str(OUT_DIAG), "samples": str(OUT_SAMPLES)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
