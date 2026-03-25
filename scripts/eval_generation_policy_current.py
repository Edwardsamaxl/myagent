from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
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
from agent.core.evidence_format import citations_are_valid, evaluate_anchor_coverage
from agent.core.schemas import EvalRecord
from agent.llm.providers import build_model_provider

EVAL_SET = ROOT / "data" / "eval" / "week1_eval_set.jsonl"
DATA_RAW = ROOT / "data" / "raw" / "finance"
OUT_DIR = ROOT / "runtime" / "day3"
OUT_FULL = OUT_DIR / "generation_policy_current.full.json"
OUT_SUMMARY = OUT_DIR / "generation_policy_current.summary.json"


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


def _run_variant(
    *,
    name: str,
    strict_policy: bool,
    refuse_th: float,
    cautious_th: float,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    os.environ["GENERATION_STRICT_POLICY"] = "1" if strict_policy else "0"
    os.environ["GENERATION_REFUSE_COVERAGE_THRESHOLD"] = str(refuse_th)
    os.environ["GENERATION_CAUTIOUS_COVERAGE_THRESHOLD"] = str(cautious_th)

    config = AgentConfig.from_env()
    model = build_model_provider(config)
    service = RagAgentService(config=config, model=model)
    for doc_id, source, content in _load_raw_docs():
        service.ingest_document(doc_id=doc_id, source=source, content=content)

    rows: list[dict[str, Any]] = []
    for item in items:
        q = str(item["question"])
        result = service.answer(q, append_to_eval_store=False)
        hits = list(result.get("retrieval_hits", []))
        answer = str(result.get("answer", ""))
        coverage, detail = evaluate_anchor_coverage(
            question=q,
            answer=answer,
            hits=[
                type("TmpHit", (), {"text": str(h.get("text_preview", ""))})()
                for h in hits
            ],
        )
        refs_ok = citations_are_valid(answer, len(hits)) if hits else False
        row = asdict(
            EvalRecord(
                trace_id=str(result.get("trace_id", "")),
                question=q,
                answer=answer,
                references=list(result.get("citations", [])),
                latency_ms=int(result.get("latency_ms", 0)),
                run_mode="offline",
            )
        )
        row.update(
            {
                "variant": name,
                "expected_answer": item.get("expected_answer", ""),
                "expected_source": item.get("source", ""),
                "retrieval_hit_count": len(hits),
                "refusal": bool(result.get("refusal", False)),
                "reason": str(result.get("reason", "")),
                "citation_valid_in_answer": refs_ok,
                "anchor_coverage": round(float(coverage), 4),
                "anchor_covered": int(detail["covered"]),
                "anchor_total": int(detail["anchors"]),
                "weighted_anchor_covered": int(detail.get("weighted_covered", 0)),
                "weighted_anchor_total": int(detail.get("weighted_total", 0)),
            }
        )
        rows.append(row)
    return {
        "name": name,
        "strict_policy": strict_policy,
        "refuse_threshold": refuse_th,
        "cautious_threshold": cautious_th,
        "rows": rows,
    }


def _summarize(rows: list[dict[str, Any]], *, false_refusal_anchor_gate: float = 0.15) -> dict[str, Any]:
    total = len(rows) or 1
    refusal_rate = sum(1 for r in rows if r["refusal"]) / total
    citation_covered_rate = (
        sum(1 for r in rows if (not r["refusal"]) and r["citation_valid_in_answer"]) / total
    )
    false_refusal_count = sum(
        1
        for r in rows
        if r["refusal"]
        and r["retrieval_hit_count"] > 0
        and r["anchor_coverage"] >= false_refusal_anchor_gate
    )
    hallucination_count = sum(
        1 for r in rows if (not r["refusal"]) and r["anchor_coverage"] < 0.35
    )
    anchor_avg = sum(float(r["anchor_coverage"]) for r in rows) / total
    return {
        "sample_size": len(rows),
        "refusal_rate": round(refusal_rate, 4),
        "citation_covered_rate": round(citation_covered_rate, 4),
        "false_refusal_count": false_refusal_count,
        "hallucination_case_count": hallucination_count,
        "anchor_coverage_avg": round(anchor_avg, 4),
        "false_refusal_definition": (
            f"refusal=true 且 retrieval_hit_count>0 且 anchor_coverage>={false_refusal_anchor_gate}"
        ),
    }


def _typical_cases(before_rows: list[dict[str, Any]], after_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_q_after = {r["question"]: r for r in after_rows}
    improved: list[dict[str, Any]] = []
    for b in before_rows:
        a = by_q_after.get(b["question"])
        if not a:
            continue
        if b["refusal"] and not a["refusal"]:
            improved.append(
                {
                    "question": b["question"],
                    "before": {
                        "trace_id": b["trace_id"],
                        "reason": b["reason"],
                        "refusal": b["refusal"],
                        "anchor_coverage": b["anchor_coverage"],
                        "answer": b["answer"][:220],
                    },
                    "after": {
                        "trace_id": a["trace_id"],
                        "reason": a["reason"],
                        "refusal": a["refusal"],
                        "anchor_coverage": a["anchor_coverage"],
                        "answer": a["answer"][:220],
                    },
                }
            )
    if len(improved) >= 5:
        return improved[:5]

    fallback: list[dict[str, Any]] = []
    for b in before_rows:
        a = by_q_after.get(b["question"])
        if not a:
            continue
        if (
            b["reason"] != a["reason"]
            or b["refusal"] != a["refusal"]
            or b["citation_valid_in_answer"] != a["citation_valid_in_answer"]
        ):
            fallback.append(
                {
                    "question": b["question"],
                    "before": {
                        "trace_id": b["trace_id"],
                        "reason": b["reason"],
                        "refusal": b["refusal"],
                        "anchor_coverage": b["anchor_coverage"],
                        "answer": b["answer"][:220],
                    },
                    "after": {
                        "trace_id": a["trace_id"],
                        "reason": a["reason"],
                        "refusal": a["refusal"],
                        "anchor_coverage": a["anchor_coverage"],
                        "answer": a["answer"][:220],
                    },
                }
            )
    return fallback[:5]


def main() -> None:
    load_dotenv(ROOT / ".env")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = _load_eval_rows()

    baseline = _run_variant(
        name="baseline",
        strict_policy=True,
        refuse_th=0.10,
        cautious_th=0.22,
        items=items,
    )
    tuned = _run_variant(
        name="tuned",
        strict_policy=True,
        refuse_th=0.07,
        cautious_th=0.16,
        items=items,
    )

    baseline_summary = _summarize(baseline["rows"])
    tuned_summary = _summarize(tuned["rows"])
    typical_cases = _typical_cases(baseline["rows"], tuned["rows"])

    full = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(EVAL_SET.relative_to(ROOT)),
        "variants": [baseline, tuned],
        "metrics": {"baseline": baseline_summary, "tuned": tuned_summary},
        "typical_cases": typical_cases,
    }
    OUT_FULL.write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "generated_at": full["generated_at"],
        "dataset": full["dataset"],
        "metrics": full["metrics"],
        "typical_cases": typical_cases,
        "files": {
            "full_report": str(OUT_FULL.relative_to(ROOT)),
            "summary": str(OUT_SUMMARY.relative_to(ROOT)),
        },
        "baseline_config": {
            "GENERATION_STRICT_POLICY": "1",
            "GENERATION_REFUSE_COVERAGE_THRESHOLD": "0.10",
            "GENERATION_CAUTIOUS_COVERAGE_THRESHOLD": "0.22",
        },
        "tuned_config": {
            "GENERATION_STRICT_POLICY": "1",
            "GENERATION_REFUSE_COVERAGE_THRESHOLD": "0.07",
            "GENERATION_CAUTIOUS_COVERAGE_THRESHOLD": "0.16",
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(full["metrics"], ensure_ascii=False, indent=2))
    print(f"full_report={OUT_FULL}")
    print(f"summary={OUT_SUMMARY}")


if __name__ == "__main__":
    main()
