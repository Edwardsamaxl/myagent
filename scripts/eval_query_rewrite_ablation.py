from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent.application.agent_service import AgentService
from agent.config import AgentConfig

OUT_DIR = ROOT / "runtime" / "day3"
OUT_REPORT = OUT_DIR / "query_rewrite_ablation_report.json"
OUT_CASES = OUT_DIR / "query_rewrite_ablation_cases.json"

DATA_RAW = ROOT / "data" / "raw" / "finance"
MODES = ("rule", "llm", "hybrid")
QTYPE_ANCHORED = "明确问法"
QTYPE_MISSING_ANCHOR = "缺锚点问法"
QTYPE_FOLLOWUP = "指代追问"
QTYPE_TOOL = "工具倾向问法"


@dataclass
class TurnCase:
    case_id: str
    session_id: str
    turn_idx: int
    question: str
    qtype: str
    expected_clarify: bool
    expected_company: str | None = "贵州茅台"
    expected_year: str | None = None
    expected_metric_keywords: tuple[str, ...] = ()
    expected_answer_keywords: tuple[str, ...] = ()
    expected_outcome: str = "answer"  # answer | clarify | refusal


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _load_raw_docs() -> list[tuple[str, str, str]]:
    docs: list[tuple[str, str, str]] = []
    for company_dir in sorted(DATA_RAW.iterdir()):
        if not company_dir.is_dir():
            continue
        for ext in ("*.md", "*.txt"):
            for p in sorted(company_dir.glob(ext)):
                docs.append((p.stem, f"{company_dir.name}/{p.name}", p.read_text(encoding="utf-8")))
    return docs


def _build_cases() -> list[TurnCase]:
    # 10 个会话 * 3 轮 = 30 条，覆盖明确问法/缺锚点/指代追问/工具倾向
    return [
        TurnCase("s1_t1", "s1", 1, "贵州茅台2024年营业收入是多少？", QTYPE_ANCHORED, False, expected_year="2024", expected_metric_keywords=("营业收入",), expected_answer_keywords=("营业收入", "2024")),
        TurnCase("s1_t2", "s1", 2, "那净利润呢？", QTYPE_FOLLOWUP, False, expected_year="2024", expected_metric_keywords=("净利润",), expected_answer_keywords=("净利润", "2024")),
        TurnCase("s1_t3", "s1", 3, "还有同比呢？", QTYPE_FOLLOWUP, False, expected_year="2024", expected_metric_keywords=("同比", "增长"), expected_outcome="answer"),

        TurnCase("s2_t1", "s2", 1, "贵州茅台股票代码是多少？", QTYPE_ANCHORED, False, expected_metric_keywords=("股票代码",), expected_answer_keywords=("600519",)),
        TurnCase("s2_t2", "s2", 2, "在哪个交易所上市？", QTYPE_FOLLOWUP, False, expected_metric_keywords=("交易所", "上市"), expected_answer_keywords=("上海证券交易所", "上交所"), expected_outcome="answer"),
        TurnCase("s2_t3", "s2", 3, "法定代表人是谁？", QTYPE_FOLLOWUP, False, expected_metric_keywords=("法定代表人",), expected_outcome="answer"),

        TurnCase("s3_t1", "s3", 1, "营业收入是多少？", QTYPE_MISSING_ANCHOR, True, expected_company=None, expected_metric_keywords=("营业收入",), expected_outcome="clarify"),
        TurnCase("s3_t2", "s3", 2, "贵州茅台2024年。", "补充澄清", False, expected_year="2024", expected_metric_keywords=("营业收入",), expected_answer_keywords=("营业收入", "2024")),
        TurnCase("s3_t3", "s3", 3, "那归母净利润呢？", QTYPE_FOLLOWUP, False, expected_year="2024", expected_metric_keywords=("净利润",), expected_answer_keywords=("净利润",)),

        TurnCase("s4_t1", "s4", 1, "净利润同比多少？", QTYPE_MISSING_ANCHOR, True, expected_company=None, expected_metric_keywords=("净利润", "同比"), expected_outcome="clarify"),
        TurnCase("s4_t2", "s4", 2, "贵州茅台2024年报。", "补充澄清", False, expected_year="2024", expected_metric_keywords=("净利润", "同比"), expected_outcome="answer"),
        TurnCase("s4_t3", "s4", 3, "还有营业收入同比？", QTYPE_FOLLOWUP, False, expected_year="2024", expected_metric_keywords=("营业收入", "同比"), expected_outcome="answer"),

        TurnCase("s5_t1", "s5", 1, "贵州茅台2025年上半年营业收入是多少？", QTYPE_ANCHORED, False, expected_year="2025", expected_metric_keywords=("营业收入",), expected_outcome="answer"),
        TurnCase("s5_t2", "s5", 2, "那2024年同期是多少？", QTYPE_FOLLOWUP, False, expected_year="2025", expected_metric_keywords=("同期", "营业收入"), expected_outcome="answer"),
        TurnCase("s5_t3", "s5", 3, "同比变化如何？", QTYPE_FOLLOWUP, False, expected_year="2025", expected_metric_keywords=("同比", "变化"), expected_outcome="answer"),

        TurnCase("s6_t1", "s6", 1, "今天北京实时气温多少？", QTYPE_TOOL, False, expected_company=None, expected_outcome="refusal"),
        TurnCase("s6_t2", "s6", 2, "那现在几点？", QTYPE_TOOL, False, expected_company=None, expected_outcome="answer"),
        TurnCase("s6_t3", "s6", 3, "计算 170899152276.34 / 10000", QTYPE_TOOL, False, expected_company=None, expected_outcome="answer"),

        TurnCase("s7_t1", "s7", 1, "请问贵州茅台公司注册地址在哪里？", QTYPE_ANCHORED, False, expected_metric_keywords=("注册地址",), expected_outcome="answer"),
        TurnCase("s7_t2", "s7", 2, "联系电话呢？", QTYPE_FOLLOWUP, False, expected_metric_keywords=("联系电话",), expected_outcome="answer"),
        TurnCase("s7_t3", "s7", 3, "证券简称呢？", QTYPE_FOLLOWUP, False, expected_metric_keywords=("证券简称",), expected_outcome="answer"),

        TurnCase("s8_t1", "s8", 1, "2024年加权平均净资产收益率是多少？", QTYPE_MISSING_ANCHOR, True, expected_company=None, expected_year="2024", expected_metric_keywords=("净资产收益率",), expected_outcome="clarify"),
        TurnCase("s8_t2", "s8", 2, "贵州茅台。", "补充澄清", False, expected_year="2024", expected_metric_keywords=("净资产收益率",), expected_outcome="answer"),
        TurnCase("s8_t3", "s8", 3, "基本每股收益呢？", QTYPE_FOLLOWUP, False, expected_year="2024", expected_metric_keywords=("每股收益",), expected_outcome="answer"),

        TurnCase("s9_t1", "s9", 1, "贵州茅台2026年营业收入是多少？", QTYPE_ANCHORED, False, expected_year="2026", expected_metric_keywords=("营业收入",), expected_outcome="refusal"),
        TurnCase("s9_t2", "s9", 2, "那有2025年上半年的吗？", QTYPE_FOLLOWUP, False, expected_year="2025", expected_metric_keywords=("营业收入",), expected_outcome="answer"),
        TurnCase("s9_t3", "s9", 3, "再给个引用。", QTYPE_FOLLOWUP, False, expected_metric_keywords=("引用",), expected_outcome="answer"),

        TurnCase("s10_t1", "s10", 1, "贵州茅台2024年总资产是多少？", QTYPE_ANCHORED, False, expected_year="2024", expected_metric_keywords=("总资产",), expected_outcome="answer"),
        TurnCase("s10_t2", "s10", 2, "负债呢？", QTYPE_FOLLOWUP, False, expected_year="2024", expected_metric_keywords=("负债",), expected_outcome="answer"),
        TurnCase("s10_t3", "s10", 3, "能简要总结资产负债结构吗？", QTYPE_FOLLOWUP, False, expected_year="2024", expected_metric_keywords=("资产", "负债"), expected_outcome="answer"),
    ]


def _is_clarify(answer: str, rag: dict[str, Any] | None) -> bool:
    if rag is None:
        return True
    text = answer.strip()
    return text.startswith("请补充") or text.startswith("为更准确回答")


def _is_refusal(answer: str) -> bool:
    t = answer.strip()
    return t.startswith("拒答") or ("无法" in t and "结论" in t)


def _score_hit_quality(case: TurnCase, hits: list[dict[str, Any]]) -> int:
    if not hits:
        return 0
    for h in hits[:3]:
        source = str(h.get("source", ""))
        meta = h.get("metadata", {}) or {}
        preview = str(h.get("text_preview", "")) + " " + str(h.get("text", ""))
        company_ok = True if not case.expected_company else (case.expected_company in source or case.expected_company in preview)
        year_ok = True
        if case.expected_year:
            year_ok = (
                case.expected_year in preview
                or case.expected_year in json.dumps(meta, ensure_ascii=False)
                or case.expected_year in source
            )
        metric_ok = True
        if case.expected_metric_keywords:
            metric_ok = any(k in preview for k in case.expected_metric_keywords)
        if company_ok and year_ok and metric_ok:
            return 1
    return 0


def _score_answer_quality(case: TurnCase, answer: str, clarify: bool) -> int:
    if case.expected_outcome == "clarify":
        return 2 if clarify else 0
    if case.expected_outcome == "refusal":
        return 2 if _is_refusal(answer) else 0
    if _is_refusal(answer):
        return 0
    if case.expected_answer_keywords:
        hit = sum(1 for k in case.expected_answer_keywords if k in answer)
        if hit == len(case.expected_answer_keywords):
            return 2
        if hit > 0:
            return 1
        return 0
    # 无显式关键词时，给可用回答基础分
    return 1


def _score_clarify_reasonable(case: TurnCase, clarify: bool) -> int:
    if case.expected_clarify:
        return 1 if clarify else 0
    return 0 if clarify else 1


def _variance(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return round(statistics.pvariance(values), 6)


def _run_mode(mode: str, cases: list[TurnCase]) -> dict[str, Any]:
    os.environ["QUERY_REWRITE_MODE"] = mode
    os.environ["QUERY_REWRITE_TEMPERATURE"] = "0.0"
    os.environ["QUERY_REWRITE_MAX_TOKENS"] = "128"
    cfg = AgentConfig.from_env()
    service = AgentService(cfg)

    for doc_id, source, content in _load_raw_docs():
        service.ingest_document(doc_id=doc_id, source=source, content=content)

    rows: list[dict[str, Any]] = []
    prefix = f"qr-{mode}-{int(time.time())}"
    created_sessions: set[str] = set()
    try:
        for c in cases:
            session_id = f"{prefix}-{c.session_id}"
            created_sessions.add(session_id)
            t0 = time.perf_counter()
            result = service.chat(session_id=session_id, user_message=c.question, use_rag=True)
            t1 = time.perf_counter()
            rag = result.get("rag")
            hits = list((rag or {}).get("retrieval_hits", []))
            answer = str(result.get("answer", ""))
            clarify = _is_clarify(answer, rag if isinstance(rag, dict) else None)
            e2e_latency_ms = round((t1 - t0) * 1000, 2)
            rag_latency_ms = int((rag or {}).get("latency_ms", 0)) if isinstance(rag, dict) else 0
            hit_quality = _score_hit_quality(c, hits)
            answer_quality = _score_answer_quality(c, answer, clarify)
            clarify_reasonable = _score_clarify_reasonable(c, clarify)
            row = {
                "case_id": c.case_id,
                "session_id": session_id,
                "turn_idx": c.turn_idx,
                "question_type": c.qtype,
                "question": c.question,
                "answer": answer,
                "rag": rag,
                "retrieval_hits": hits,
                "clarify_triggered": clarify,
                "latency_ms": {
                    "e2e": e2e_latency_ms,
                    "rag": rag_latency_ms,
                },
                "scores": {
                    "hit_quality": hit_quality,         # 0/1
                    "answer_quality": answer_quality,   # 0/1/2
                    "clarify_reasonable": clarify_reasonable,  # 0/1
                },
            }
            rows.append(row)
    finally:
        # 清理本评测创建的临时会话，避免污染 Web 历史对话
        for sid in sorted(created_sessions):
            try:
                service.delete_chat_session(sid)
            except Exception:  # noqa: BLE001
                pass

    hit_values = [float(r["scores"]["hit_quality"]) for r in rows]
    ans_values = [float(r["scores"]["answer_quality"]) for r in rows]
    clr_values = [float(r["scores"]["clarify_reasonable"]) for r in rows]
    clar_rate_values = [1.0 if r["clarify_triggered"] else 0.0 for r in rows]
    lat_values = [float(r["latency_ms"]["e2e"]) for r in rows]

    metrics = {
        "sample_size": len(rows),
        "hit_quality_mean": round(sum(hit_values) / len(hit_values), 4),
        "hit_quality_var": _variance(hit_values),
        "answer_quality_mean": round(sum(ans_values) / len(ans_values), 4),
        "answer_quality_var": _variance(ans_values),
        "clarify_trigger_rate": round(sum(clar_rate_values) / len(clar_rate_values), 4),
        "clarify_reasonable_mean": round(sum(clr_values) / len(clr_values), 4),
        "clarify_reasonable_var": _variance(clr_values),
        "avg_e2e_latency_ms": round(sum(lat_values) / len(lat_values), 2),
        "p95_e2e_latency_ms": round(sorted(lat_values)[max(0, int(len(lat_values) * 0.95) - 1)], 2),
    }

    # 典型案例：分数最高 3 + 最低 2（共 5）
    ranked = sorted(
        rows,
        key=lambda r: (
            r["scores"]["answer_quality"],
            r["scores"]["hit_quality"],
            r["scores"]["clarify_reasonable"],
        ),
        reverse=True,
    )
    typical = ranked[:3] + list(reversed(ranked[-2:]))
    return {"mode": mode, "metrics": metrics, "rows": rows, "typical_cases": typical}


def main() -> None:
    load_dotenv(ROOT / ".env")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = _build_cases()

    runs: dict[str, Any] = {}
    for mode in MODES:
        runs[mode] = _run_mode(mode, cases)

    comparison = [
        {
            "mode": m,
            **runs[m]["metrics"],
        }
        for m in MODES
    ]

    # 简单推荐：回答质量优先，其次命中质量，再惩罚过度澄清与时延
    def _rank_score(item: dict[str, Any]) -> float:
        return (
            item["answer_quality_mean"] * 1.2
            + item["hit_quality_mean"] * 1.0
            + item["clarify_reasonable_mean"] * 0.8
            - item["clarify_trigger_rate"] * 0.4
            - (item["avg_e2e_latency_ms"] / 100000.0)
        )

    best = sorted(comparison, key=_rank_score, reverse=True)[0]

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_desc": "30-turn mixed multi-turn set for query-rewrite ablation",
        "repro_commands": [
            "python scripts/run_ingest.py",
            "python scripts/eval_query_rewrite_ablation.py",
        ],
        "config_version": {
            "git_commit": _git_rev(),
            "script": "scripts/eval_query_rewrite_ablation.py",
            "rewrite_modes": list(MODES),
            "query_rewrite_temperature": "0.0",
            "query_rewrite_max_tokens": "128",
        },
        "comparison_table": comparison,
        "recommended_mode": best["mode"],
        "recommendation_reason": "按回答质量优先、命中质量次之，并约束过度澄清与时延的综合评分最高。",
        "result_files": {
            "report": str(OUT_REPORT.relative_to(ROOT)),
            "cases": str(OUT_CASES.relative_to(ROOT)),
        },
    }
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    cases_out = {m: runs[m]["typical_cases"] for m in MODES}
    OUT_CASES.write_text(json.dumps(cases_out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "report": str(OUT_REPORT),
                "cases": str(OUT_CASES),
                "recommended_mode": best["mode"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
