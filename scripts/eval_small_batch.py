"""小批量 RAG 评测：固定 5 题，输出每题 retrieval_hits 前 3 条。

用途：
- 快速判断是“问法问题”还是“检索侧召回问题”。
- 输出保存到 runtime/day2_small_batch_eval.json，便于复盘与对比。
"""

from __future__ import annotations

import json
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
EVAL_SET = ROOT / "data" / "eval" / "week1_eval_set.jsonl"
OUT_FILE = ROOT / "runtime" / "day2_small_batch_eval.json"
RAG_URL = "http://127.0.0.1:7860/api/rag"
ALLOWED_SOURCES = {"贵州茅台/年报_2024.md", "贵州茅台/半年报_2025.md"}


def load_eval_items() -> list[dict]:
    return [
        json.loads(line)
        for line in EVAL_SET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_small_batch() -> list[dict]:
    items = load_eval_items()
    # 2题已知能答 + 3题常失败（数字口径/长问法）
    selected = [items[0], items[6], items[1], items[11], items[4]]
    rows: list[dict] = []

    for idx, item in enumerate(selected, 1):
        question = item["question"]
        row: dict = {
            "case": idx,
            "question": question,
            "expected": item.get("expected_answer", ""),
            "source_expected": item.get("source", ""),
            "type": item.get("type", ""),
        }
        try:
            resp = requests.post(
                RAG_URL,
                json={"question": question, "top_k": 5},
                timeout=180,
            )
            row["http_status"] = resp.status_code
            if resp.status_code != 200:
                row["error_body"] = resp.text[:1200]
                rows.append(row)
                continue

            data = resp.json()
            hits_top3 = []
            for hit in data.get("retrieval_hits", [])[:3]:
                source = hit.get("source")
                hits_top3.append(
                    {
                        "chunk_id": hit.get("chunk_id"),
                        "score": hit.get("score"),
                        "source": source,
                        "source_ok": source in ALLOWED_SOURCES,
                        "metadata": hit.get("metadata", {}),
                        "text_preview": hit.get("text_preview", ""),
                    }
                )
            row.update(
                {
                    "trace_id": data.get("trace_id"),
                    "latency_ms": data.get("latency_ms"),
                    "refusal": data.get("refusal"),
                    "reason": data.get("reason"),
                    "retrieval_hit_count": len(data.get("retrieval_hits", [])),
                    "retrieval_hits_top3": hits_top3,
                }
            )
        except Exception as e:  # noqa: BLE001
            row["http_status"] = "EXCEPTION"
            row["error_body"] = str(e)

        rows.append(row)

    return rows


def main() -> None:
    rows = run_small_batch()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入: {OUT_FILE}")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
