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
OUT_FILE = OUT_DIR / "generation_policy_ab.json"
OUT_SUMMARY = OUT_DIR / "generation_policy_ab.summary.json"


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


def _run_variant(name: str, strict_policy: bool, items: list[dict[str, Any]]) -> dict[str, Any]:
    os.environ["GENERATION_STRICT_POLICY"] = "1" if strict_policy else "0"
    config = AgentConfig.from_env()
    model = build_model_provider(config)
    service = RagAgentService(config=config, model=model)
    for doc_id, source, content in _load_raw_docs():
        service.ingest_document(doc_id=doc_id, source=source, content=content)

    rows: list[dict[str, Any]] = []
    for item in items:
        q = str(item["question"])
        expected_source = str(item.get("source", ""))
        result = service.answer(q, append_to_eval_store=False)
        hits = result.get("retrieval_hits", [])
        answer = str(result.get("answer", ""))
        coverage, detail = evaluate_anchor_coverage(
            question=q,
            answer=answer,
            hits=[
                type("TmpHit", (), {"text": str(h.get("text_preview", ""))})()  # minimal object for helper
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
                "expected_source": expected_source,
                "retrieval_hit_count": len(hits),
                "refusal": bool(result.get("refusal", False)),
                "reason": str(result.get("reason", "")),
                "citation_valid_in_answer": refs_ok,
                "anchor_coverage": round(float(coverage), 4),
                "anchor_covered": int(detail["covered"]),
                "anchor_total": int(detail["anchors"]),
            }
        )
        rows.append(row)
    return {"name": name, "strict_policy": strict_policy, "rows": rows}


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows) or 1
    refusal_rate = sum(1 for r in rows if r["refusal"]) / total
    citation_covered_rate = (
        sum(1 for r in rows if (not r["refusal"]) and r["citation_valid_in_answer"]) / total
    )
    false_refusal_count = sum(
        1
        for r in rows
        if r["refusal"] and r["retrieval_hit_count"] > 0 and r["reason"] == "insufficient_evidence"
    )
    hallucination_count = sum(
        1 for r in rows if (not r["refusal"]) and r["anchor_coverage"] < 0.35
    )
    return {
        "sample_size": len(rows),
        "refusal_rate": round(refusal_rate, 4),
        "citation_covered_rate": round(citation_covered_rate, 4),
        "false_refusal_count": false_refusal_count,
        "hallucination_case_count": hallucination_count,
    }


def _typical_cases(before_rows: list[dict[str, Any]], after_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_q_after = {r["question"]: r for r in after_rows}
    changed: list[dict[str, Any]] = []
    for b in before_rows:
        a = by_q_after.get(b["question"])
        if not a:
            continue
        if (
            b["refusal"] != a["refusal"]
            or b["reason"] != a["reason"]
            or b["citation_valid_in_answer"] != a["citation_valid_in_answer"]
        ):
            changed.append(
                {
                    "question": b["question"],
                    "before": {
                        "trace_id": b["trace_id"],
                        "refusal": b["refusal"],
                        "reason": b["reason"],
                        "citation_valid_in_answer": b["citation_valid_in_answer"],
                        "anchor_coverage": b["anchor_coverage"],
                        "answer": b["answer"][:240],
                    },
                    "after": {
                        "trace_id": a["trace_id"],
                        "refusal": a["refusal"],
                        "reason": a["reason"],
                        "citation_valid_in_answer": a["citation_valid_in_answer"],
                        "anchor_coverage": a["anchor_coverage"],
                        "answer": a["answer"][:240],
                    },
                }
            )
    if len(changed) < 5:
        fallback = []
        for b, a in zip(before_rows, after_rows):
            fallback.append(
                {
                    "question": b["question"],
                    "before": {"trace_id": b["trace_id"], "reason": b["reason"], "refusal": b["refusal"]},
                    "after": {"trace_id": a["trace_id"], "reason": a["reason"], "refusal": a["refusal"]},
                }
            )
        changed.extend(fallback[: 5 - len(changed)])
    return changed[:5]


def main() -> None:
    load_dotenv(ROOT / ".env")
    items = _load_eval_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    before = _run_variant("before", strict_policy=False, items=items)
    after = _run_variant("after", strict_policy=True, items=items)

    before_summary = _summarize(before["rows"])
    after_summary = _summarize(after["rows"])
    cases = _typical_cases(before["rows"], after["rows"])

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(EVAL_SET.relative_to(ROOT)),
        "metrics": {"before": before_summary, "after": after_summary},
        "typical_cases": cases,
        "variants": [before, after],
    }
    OUT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_SUMMARY.write_text(
        json.dumps(
            {
                "generated_at": report["generated_at"],
                "dataset": report["dataset"],
                "metrics": report["metrics"],
                "typical_cases": cases,
                "files": {
                    "full_report": str(OUT_FILE.relative_to(ROOT)),
                    "summary": str(OUT_SUMMARY.relative_to(ROOT)),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"full_report={OUT_FILE}")
    print(f"summary={OUT_SUMMARY}")


if __name__ == "__main__":
    main()
