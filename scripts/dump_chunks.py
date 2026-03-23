"""本地运行 ingestion，把切块结果写到文件，便于检查。

用法（在项目根目录）:
  python scripts/dump_chunks.py
  python scripts/dump_chunks.py data/raw/finance/贵州茅台/年报_2024.md

会读取 .env 中的 CHUNK_SIZE / CHUNK_OVERLAP（或默认值），输出到 data/debug/。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from agent.config import AgentConfig
from agent.core.ingestion import DocumentIngestionPipeline


def main() -> None:
    cfg = AgentConfig.from_env()
    pipe = DocumentIngestionPipeline(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )

    if len(sys.argv) > 1:
        paths = [Path(sys.argv[1])]
    else:
        paths = sorted((ROOT / "data" / "raw" / "finance").rglob("*.md"))
        if not paths:
            print("未找到 md，请指定文件: python scripts/dump_chunks.py <path/to/file.md>")
            sys.exit(1)

    out_dir = ROOT / "data" / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        if not path.exists():
            print(f"跳过（不存在）: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        doc_id = path.stem
        try:
            source = str(path.relative_to(ROOT))
        except ValueError:
            source = str(path)
        chunks, result = pipe.ingest_text(doc_id=doc_id, source=source, content=text)

        # 人类可读：带分隔符
        lines: list[str] = [
            f"# chunk dump: {source}",
            f"# CHUNK_SIZE={cfg.chunk_size} CHUNK_OVERLAP={cfg.chunk_overlap}",
            f"# total_chunks={result.total_chunks}",
            "",
        ]
        for c in chunks:
            lines.append("=" * 72)
            lines.append(
                f"[{c.chunk_id}] len={len(c.text)} source={c.source} metadata={c.metadata!r}"
            )
            lines.append("-" * 72)
            lines.append(c.text)
            lines.append("")

        txt_path = out_dir / f"chunks_{doc_id}.txt"
        txt_path.write_text("\n".join(lines), encoding="utf-8")

        # 机器可读：jsonl
        jsonl_path = out_dir / f"chunks_{doc_id}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(
                    json.dumps(
                        {
                            "chunk_id": c.chunk_id,
                            "len": len(c.text),
                            "metadata": c.metadata,
                            "text": c.text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        print(f"已写入: {txt_path}")
        print(f"已写入: {jsonl_path} ({result.total_chunks} 块)")


if __name__ == "__main__":
    main()
