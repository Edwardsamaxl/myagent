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
    finance_dir = ROOT / "data" / "raw" / "finance"
    if not finance_dir.exists():
        print(f"目录不存在: {finance_dir}")
        sys.exit(1)

    pdf_paths = sorted(finance_dir.rglob("*.pdf"))
    if not pdf_paths:
        print(f"未找到 PDF: {finance_dir}")
        sys.exit(0)

    print(f"找到 {len(pdf_paths)} 个 PDF，开始转换 TXT...")
    converted = 0
    failed = 0
    for pdf_path in pdf_paths:
        print(f"提取: {pdf_path}")
        try:
            raw = extract_pdf_with_tables(pdf_path)
        except ImportError:
            print("  需要安装 pdfplumber: pip install pdfplumber")
            sys.exit(1)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  失败: {exc}")
            continue
        cleaned = clean_text(raw)
        out_path = pdf_path.with_suffix(".txt")
        out_path.write_text(cleaned, encoding="utf-8")
        converted += 1
        print(f"  已保存: {out_path} ({len(cleaned)} 字符)")

    print("")
    print("=== 转换汇总（PDF -> TXT）===")
    print(f"  扫描 PDF 数: {len(pdf_paths)}")
    print(f"  成功: {converted}")
    print(f"  失败: {failed}")


if __name__ == "__main__":
    main()
