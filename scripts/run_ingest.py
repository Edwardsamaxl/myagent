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

    n_files = len(files)
    print(f"找到 {n_files} 个文档，开始入库...")
    total_raw_chunks = 0
    total_ingested_chunks = 0
    total_dropped_duplicates = 0
    ok = 0
    failed: list[tuple[str, str]] = []

    for f, doc_id, source, content in files:
        try:
            r = requests.post(INGEST_URL, json={"doc_id": doc_id, "source": source, "content": content}, timeout=60)
            r.raise_for_status()
            data = r.json()
            raw_n = int(data.get("total_chunks", 0))
            ingested_n = int(data.get("deduplicated_chunks", 0))
            dropped_n = int(data.get("dropped_duplicates", max(0, raw_n - ingested_n)))
            total_raw_chunks += raw_n
            total_ingested_chunks += ingested_n
            total_dropped_duplicates += dropped_n
            ok += 1
            print(f"  OK  {source}: raw={raw_n} ingested={ingested_n} dropped={dropped_n}")
        except requests.exceptions.ConnectionError:
            print("连接失败，请确保 main.py 已启动（python main.py）")
            sys.exit(1)
        except Exception as e:
            err = str(e)
            failed.append((source, err))
            print(f"  ERR {source}: {err}")

    print("")
    print("=== 入库汇总（可复现口径）===")
    print(f"  扫描文档数: {n_files}")
    print(f"  成功: {ok}，失败: {len(failed)}")
    print(f"  总原始块数（total_chunks 之和）: {total_raw_chunks}")
    print(f"  总入库块数（deduplicated_chunks 之和）: {total_ingested_chunks}")
    print(f"  总去重块数（dropped_duplicates 之和）: {total_dropped_duplicates}")
    if failed:
        print("  失败列表:")
        for src, err in failed:
            print(f"    - {src}: {err}")
        sys.exit(1)
    print("入库完成。")


if __name__ == "__main__":
    main()
