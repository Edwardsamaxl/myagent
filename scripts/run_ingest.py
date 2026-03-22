"""批量将 data/raw/finance/ 下的 txt/md 文档入库。

使用前请先启动 Web 服务：python main.py
本脚本会向 http://127.0.0.1:7860/api/ingest 发送请求。

注意：当前检索索引为内存存储，重启 main.py 后需重新入库。
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "finance"
INGEST_URL = "http://127.0.0.1:7860/api/ingest"


def main() -> None:
    if not DATA_RAW.exists():
        print(f"目录不存在: {DATA_RAW}")
        sys.exit(1)

    files: list[tuple[Path, str, str, str]] = []
    for company_dir in sorted(DATA_RAW.iterdir()):
        if not company_dir.is_dir():
            continue
        company = company_dir.name
        for ext in ("*.txt", "*.md"):
            for f in sorted(company_dir.glob(ext)):
                content = f.read_text(encoding="utf-8")
                doc_id = f.stem
                source = f"{company}/{f.name}"
                files.append((f, doc_id, source, content))

    if not files:
        print("未找到任何 txt/md 文件。")
        sys.exit(0)

    print(f"找到 {len(files)} 个文档，开始入库...")
    for f, doc_id, source, content in files:
        try:
            r = requests.post(INGEST_URL, json={"doc_id": doc_id, "source": source, "content": content}, timeout=60)
            r.raise_for_status()
            data = r.json()
            print(f"  ✓ {source}: chunks={data.get('deduplicated_chunks', '?')}")
        except requests.exceptions.ConnectionError:
            print("连接失败，请确保 main.py 已启动（python main.py）")
            sys.exit(1)
        except Exception as e:
            print(f"  ✗ {source}: {e}")

    print("入库完成。")


if __name__ == "__main__":
    main()
