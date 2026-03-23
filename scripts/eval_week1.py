"""离线跑 `data/eval/week1_eval_set.jsonl`，结果写入 `DATA_DIR` 下 JSONL，指标与在线 `aggregate_eval_rows` 口径一致。

用法（需已按 `data/README.md` 入库，且模型可用）：

  python scripts/eval_week1.py

环境变量：

- `OFFLINE_EVAL_OUT`：输出文件路径，默认 `<DATA_DIR>/offline_eval_week1.jsonl`（`DATA_DIR` 默认 `./runtime`）
- `MODEL_PROVIDER` 等：与 `main.py` 相同

每条结果含 `trace_id`，可与同目录 `traces.jsonl` 对照；不写入在线 `eval_records.jsonl`。
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = str(CURRENT_DIR / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dotenv import load_dotenv

from agent.application.rag_agent_service import RagAgentService
from agent.config import AgentConfig
from agent.core.evaluation import aggregate_eval_rows, substring_match
from agent.core.schemas import EvalRecord
from agent.llm.providers import build_model_provider


def main() -> None:
    load_dotenv(CURRENT_DIR / ".env")
    config = AgentConfig.from_env()
    eval_path = CURRENT_DIR / "data" / "eval" / "week1_eval_set.jsonl"
    out_path = Path(
        os.getenv("OFFLINE_EVAL_OUT", str(config.data_dir / "offline_eval_week1.jsonl"))
    ).resolve()

    lines = [ln for ln in eval_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    items = [json.loads(ln) for ln in lines]

    model = build_model_provider(config)
    rag = RagAgentService(config=config, model=model)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")

    rows: list[dict] = []
    for item in items:
        question = item["question"]
        expected = item.get("expected_answer", "")
        result = rag.answer(question, append_to_eval_store=False)
        latency_ms = int(result["latency_ms"])
        trace_id = str(result["trace_id"])
        answer = str(result["answer"])
        sm = substring_match(expected, answer)
        record = EvalRecord(
            trace_id=trace_id,
            question=question,
            answer=answer,
            references=list(result.get("citations", [])),
            latency_ms=latency_ms,
            run_mode="offline",
        )
        row = asdict(record)
        row["expected_answer"] = expected
        row["source"] = item.get("source", "")
        row["eval_type"] = item.get("type", "")
        row["substring_match"] = sm
        row["eval_set_id"] = "week1"
        rows.append(row)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = aggregate_eval_rows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"已写入: {out_path}")


if __name__ == "__main__":
    main()
