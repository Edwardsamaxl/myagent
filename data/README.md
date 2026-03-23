# 数据目录说明

## 结构

```
data/
├── raw/finance/          # 原始文档（供 RAG 摄入）
│   └── 贵州茅台/
│       ├── 年报_2024.txt
│       └── 半年报_2025.txt
└── eval/                 # 评估集（问题+参考答案+证据来源）
    └── week1_eval_set.jsonl
```

## 查看切块长什么样（不入库）

在项目根目录执行：

```powershell
python scripts/dump_chunks.py
```

会处理 `data/raw/finance/` 下所有 `.md`，并把结果写到：

- `data/debug/chunks_<文档名>.txt` — 带分隔线，人类阅读
- `data/debug/chunks_<文档名>.jsonl` — 每行一个 JSON，方便程序处理

只切某一个文件：

```powershell
python scripts/dump_chunks.py data/raw/finance/贵州茅台/年报_2024.md
```

切块参数与线上一致：在项目根 `.env` 里设 `CHUNK_SIZE`、`CHUNK_OVERLAP`（默认 500 / 80）。

## 入库流程

1. **启动 Web 服务**：`python main.py`
2. **批量入库**：`python scripts/run_ingest.py`（会读取 `data/raw/finance/` 下所有 `.txt`/`.md` 并 POST 到 `/api/ingest`；结束时打印**文档数、成功/失败数、总块数**）

或通过 Web 页面「RAG 文档摄入」区域手动粘贴单篇内容入库。

**注意**：当前检索索引为内存存储，重启 main.py 后需重新入库。

## 文档级元数据（入库后写入每个 chunk）

每条 `DocumentChunk` 的 `metadata` 为字符串键值对，默认由 **`source` 推导**（与 `data/raw/finance/<公司名>/<文件名>` 或 `run_ingest` 使用的 `公司名/文件名` 一致）：

| 键 | 含义 | 推导规则（摘要） |
|----|------|------------------|
| `source` | 来源标识 | 与请求中的 `source` 相同 |
| `company` | 公司名 | 路径中 `finance/` 下一级目录，或 `公司/文件` 时取第一段 |
| `doc_type` | 文档类型 | 文件名含「半年度报告/半年报/半年度」→ 半年报；含「年度报告」或含「年报」且不含「半年」→ 年报；否则按扩展名（如 `markdown`/`text`/`pdf`） |
| `date` | 年份 | 文件名中 `_YYYY` 或 `YYYY年` |

可选：在 `POST /api/ingest` 的 JSON 中增加 **`metadata` 对象**，键值均为字符串，会**覆盖**上述推导中的同名字段。

## 评估集

`data/eval/week1_eval_set.jsonl` 每行一条，格式：

```json
{"question": "...", "expected_answer": "...", "source": "贵州茅台/年报_2024.txt", "type": "财务指标"}
```

## PDF 转 Markdown（推荐）

1. 将 PDF 放入 `data/raw/finance/<公司名>/` 下
2. 修改 `scripts/extract_pdf_to_md.py` 中的 `pdfs` 列表
3. 运行：`python scripts/extract_pdf_to_md.py`（输出 .md）
4. 运行 `python scripts/run_ingest.py` 入库

MD 格式保留 ##/### 结构，ingestion 会按结构分块，效果优于纯 txt。

## 依赖

- `pypdf`：PDF 文本提取
- `requests`：批量入库脚本需要
