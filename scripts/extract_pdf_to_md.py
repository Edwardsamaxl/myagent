"""将 PDF 转为 Markdown，保留层级结构，便于后续分块与检索。

- 第X节 -> ## 标题
- 一、二、三、 -> ### 子标题
- 表格保持 | 列 | 列 | 格式
- 不提取图片
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def format_table_md(rows: list[list]) -> str:
    """表格转为 Markdown 表格。"""
    if not rows:
        return ""
    lines = []
    for row in rows:
        cells = [str(c).strip() if c is not None and str(c).strip() else "" for c in row]
        cells = [c.replace("|", "｜").replace("\n", " ") for c in cells]
        if any(cells):
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) if lines else ""


def extract_pdf_to_md(pdf_path: Path) -> str:
    """PDF -> 结构化 Markdown。"""
    import pdfplumber

    page_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            parts = []
            if text and text.strip():
                parts.append(text.strip())

            tables = page.extract_tables()
            for t in tables:
                if t:
                    md_table = format_table_md(t)
                    if md_table:
                        parts.append(md_table)

            if parts:
                page_parts.append("\n\n".join(parts))

    raw = "\n\n".join(page_parts)
    return _to_structured_md(raw)


def _to_structured_md(text: str) -> str:
    """对提取文本做结构标记：第X节 -> ##，一、二、 -> ###。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\d{1,3}\s*/\s*\d{1,3}", "", text)

    lines = text.split("\n")
    out: list[str] = []
    i = 0

    # 章节标题模式：行首「第X节」或「一、」「二、」等（排除长数字行）
    section_pat = re.compile(r"^第([一二三四五六七八九十百]+)节\s*(.*)$")
    subsection_pat = re.compile(r"^([一二三四五六七八九十百]+)、\s*(.*)$")

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 第X节 -> ## 第二节 公司简介...
        m1 = section_pat.match(stripped)
        if m1:
            out.append("")
            out.append(f"## {stripped}")
            i += 1
            continue

        # 一、二、三、 -> ###（排除：行过长且含大量数字，可能是表格行）
        m2 = subsection_pat.match(stripped)
        if m2 and len(stripped) < 80:
            digit_count = sum(1 for c in stripped if c.isdigit())
            if digit_count < 8:  # 标题一般数字少
                out.append("")
                out.append(f"### {stripped}")
                i += 1
                continue

        # 普通行
        if stripped:
            out.append(line)
        elif out and out[-1] != "":
            out.append("")
        i += 1

    return "\n".join(out).strip()


def main() -> None:
    finance_dir = ROOT / "data" / "raw" / "finance"
    if not finance_dir.exists():
        print(f"目录不存在: {finance_dir}")
        sys.exit(1)

    pdf_paths = sorted(finance_dir.rglob("*.pdf"))
    if not pdf_paths:
        print(f"未找到 PDF: {finance_dir}")
        sys.exit(0)

    print(f"找到 {len(pdf_paths)} 个 PDF，开始转换 Markdown...")
    converted = 0
    failed = 0
    for pdf_path in pdf_paths:
        print(f"提取: {pdf_path}")
        try:
            md_content = extract_pdf_to_md(pdf_path)
        except ImportError:
            print("  需要: pip install pdfplumber")
            sys.exit(1)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  失败: {exc}")
            continue
        out_path = pdf_path.with_suffix(".md")
        out_path.write_text(md_content, encoding="utf-8")
        converted += 1
        print(f"  已保存: {out_path} ({len(md_content)} 字符)")

    print("")
    print("=== 转换汇总（PDF -> MD）===")
    print(f"  扫描 PDF 数: {len(pdf_paths)}")
    print(f"  成功: {converted}")
    print(f"  失败: {failed}")


if __name__ == "__main__":
    main()
