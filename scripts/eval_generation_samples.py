from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent.application.rag_agent_service import RagAgentService
from agent.config import AgentConfig
from agent.core.evidence_format import citations_are_valid, contains_citation_marker
from agent.llm.providers import build_model_provider


def _load_raw_docs() -> list[tuple[str, str, str]]:
    data_raw = ROOT / "data" / "raw" / "finance"
    docs: list[tuple[str, str, str]] = []
    for company_dir in sorted(data_raw.iterdir()):
        if not company_dir.is_dir():
            continue
        for ext in ("*.md", "*.txt"):
            for p in sorted(company_dir.glob(ext)):
                docs.append((p.stem, f"{company_dir.name}/{p.name}", p.read_text(encoding="utf-8")))
    return docs


def main() -> None:
    load_dotenv(ROOT / ".env")
    cfg = AgentConfig.from_env()
    service = RagAgentService(config=cfg, model=build_model_provider(cfg))

    for doc_id, source, content in _load_raw_docs():
        service.ingest_document(doc_id=doc_id, source=source, content=content)

    questions = [
        "贵州茅台2024年营业收入是多少？",
        "贵州茅台2024年归属于上市公司股东的净利润是多少？",
        "贵州茅台法定代表人是谁？",
        "贵州茅台公司股票在哪个交易所上市？",
        "贵州茅台2024年加权平均净资产收益率是多少？",
        "今天北京的实时气温是多少？",  # 空检索倾向
        "贵州茅台2026年营业收入是多少？",  # 有命中但证据不足倾向
        "贵州茅台2025年上半年利润分配预案是什么？",
    ]

    rows: list[dict[str, object]] = []
    for q in questions:
        res = service.answer(q, append_to_eval_store=False)
        answer = str(res.get("answer", ""))
        hits = list(res.get("retrieval_hits", []))
        refusal = bool(res.get("refusal", False))
        reason = str(res.get("reason", ""))
        citation_ok = False
        if not refusal:
            citation_ok = contains_citation_marker(answer) and citations_are_valid(answer, len(hits))
        rows.append(
            {
                "question": q,
                "trace_id": str(res.get("trace_id", "")),
                "refusal": refusal,
                "reason": reason,
                "retrieval_hit_count": len(hits),
                "citation_ok": citation_ok,
                "answer_preview": answer[:260],
            }
        )

    out = ROOT / "runtime" / "day3" / "generation_samples.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written={out}")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
