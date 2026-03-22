"""从 PDF 提取文本并保存到 data/raw/finance/<公司>/ 目录。

使用 pdfplumber：
- 正文用 extract_text()
- 表格用 extract_tables() 单独提取并格式化为易读文本，避免数字粘成一团
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def format_table(rows: list[list]) -> str:
    """将表格行格式化为 | 列1 | 列2 | 列3 | 形式。"""
    if not rows:
        return ""
    lines = []
    for row in rows:
        cells = [str(c).strip() if c is not None and str(c).strip() else "" for c in row]
        if any(cells):  # 跳过全空行
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) if lines else ""


def extract_pdf_with_tables(pdf_path: Path) -> str:
    """使用 pdfplumber 提取：正文 + 结构化表格。"""
    import pdfplumber

    page_texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # 正文
            text = page.extract_text()
            parts = []

            if text and text.strip():
                parts.append(text.strip())

            # 表格（单独提取，保持结构）
            tables = page.extract_tables()
            if tables:
                for t in tables:
                    if t:
                        formatted = format_table(t)
                        if formatted:
                            parts.append(f"[表格]\n{formatted}")

            if parts:
                page_texts.append("\n\n".join(parts))

    return "\n\n".join(page_texts)


def clean_text(text: str) -> str:
    """简单清洗：换行统一、空格压缩、页码移除。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\d{1,3}\s*/\s*\d{1,3}", "", text)
    return text.strip()


def main() -> None:
    data_dir = ROOT / "data" / "raw" / "finance" / "贵州茅台"
    data_dir.mkdir(parents=True, exist_ok=True)

    pdfs = [
        (
            Path(r"C:\Users\Ed\Downloads\贵州茅台：贵州茅台2024年年度报告.pdf"),
            "年报_2024.txt",
        ),
        (
            Path(r"C:\Users\Ed\Downloads\贵州茅台：贵州茅台2025年半年度报告.pdf"),
            "半年报_2025.txt",
        ),
    ]

    for pdf_path, out_name in pdfs:
        if not pdf_path.exists():
            print(f"跳过（文件不存在）: {pdf_path}")
            continue
        print(f"提取: {pdf_path.name}")
        try:
            raw = extract_pdf_with_tables(pdf_path)
        except ImportError:
            print("  需要安装 pdfplumber: pip install pdfplumber")
            sys.exit(1)
        cleaned = clean_text(raw)
        out_path = data_dir / out_name
        out_path.write_text(cleaned, encoding="utf-8")
        print(f"  已保存: {out_path} ({len(cleaned)} 字符)")


if __name__ == "__main__":
    main()
