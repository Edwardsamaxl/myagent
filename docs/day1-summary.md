# Day 1 工作总结

## 1. 需求边界

- **场景**：金融 A 股文档问答，辅助获取上市公司信息（财报、市盈率等）
- **问题类型**：财务指标、估值指标、公司基本面、重大事项、研报观点、时间线/事件
- **不做**：实时股价、买卖建议、跨多公司复杂推理
- **一句话**：本 Agent 回答「已入库文档」中的公司基本面、财报、估值、公告、研报等问题，不提供实时行情，也不给出买卖建议。

## 2. 数据与目录

| 目录/文件 | 说明 |
|-----------|------|
| `data/raw/finance/贵州茅台/` | 贵州茅台源文档 |
| `年报_2024.md` | 2024 年年报（PDF 转 Markdown） |
| `半年报_2025.md` | 2025 年半年报（PDF 转 Markdown） |
| `data/eval/week1_eval_set.jsonl` | 25 条评估集（问题 + 参考答案 + 来源） |

## 3. PDF 转 Markdown

- **脚本**：`scripts/extract_pdf_to_md.py`
- **能力**：PDF → 结构化 MD（`##` / `###` 标题、表格 `| 列 | 列 |`），不提取图片
- **工具**：pdfplumber（正文 + `extract_tables()`）

## 4. Ingestion 改造

- **分块策略**：
  1. 以 `chunk_size` 为目标切点，到长度后再判断；
  2. 若切点落在句中，向前延伸到句末（`。！？；` 或 `\n`）；
  3. 单块最大长度 `chunk_size * 2`；
  4. 有 `##`/`###` 时先按小节切，再对每节用上述策略；
  5. **灵活重叠**：下一块起点尽量对齐句首，使重叠区至少包含整句。
- **去重**：移除 hash 去重，按顺序编号，便于排查异常。

## 5. 切块预览

- **脚本**：`scripts/dump_chunks.py`
- **用途**：本地运行 ingestion，将切块结果写入 `data/debug/`，便于检查
- **输出**：`chunks_<文档名>.txt`（人读）、`chunks_<文档名>.jsonl`（机器读）

## 6. 新增/修改文件

| 文件 | 说明 |
|------|------|
| `scripts/extract_pdf_to_md.py` | PDF → MD |
| `scripts/extract_pdf.py` | PDF → txt（保留） |
| `scripts/dump_chunks.py` | 切块预览 |
| `scripts/run_ingest.py` | 批量入库（支持 .md） |
| `src/agent/core/ingestion.py` | 结构感知 + 定长+句末 + 灵活重叠 |
| `data/README.md` | 数据目录与流程说明 |
| `src/agent/core/ingestion.py` | Ingestion 实现与模块内注释 |
| `docs/day2-daily-plan.md` | Day 2 起元数据与入库契约（协调见 `docs/coordination.md`） |

## 7. 验收

- [x] 能解释：本 Agent 解决什么、不解决什么
- [x] 数据目录与评估集就绪
- [x] PDF 可转 MD，表格结构保留
- [x] 切块可预览，效果符合预期
