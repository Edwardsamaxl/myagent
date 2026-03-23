# Day 2 Ingestion 说明（可交接版）

本文档用于补齐 Day 2 的实现解释，供后续 Agent（尤其是 Day 3 检索侧）快速理解当前入库行为与统计口径。

---

## 1. 当前目标与边界

- 目标：让文档入库链路具备可追溯 metadata、可观测统计，以及可控去重行为。
- 边界：本阶段不改 `retrieval.py` / `rerank.py` / `generation.py` 算法，仅处理 ingestion 相关路径。

---

## 2. 入库链路（请求到 chunk）

主链路如下：

1. `POST /api/ingest` 接收 `doc_id/source/content` 与可选参数（`metadata`、`chunk_size`、`chunk_overlap`、`dedup_across_docs`）。
2. `AgentService.ingest_document` 透传到 `RagAgentService.ingest_document`。
3. `DocumentIngestionPipeline.ingest_text` 执行：
   - 文本清洗
   - 分块（固定长度 + 句末对齐 + 重叠）
   - 内容哈希去重
   - 组装 `DocumentChunk`
4. `InMemoryHybridRetriever.upsert_chunks` 写入内存检索索引。

---

## 3. 分块策略（当前默认）

- 默认参数：`chunk_size=500`，`chunk_overlap=80`（来自环境变量，支持单次请求覆盖）。
- 切分原则：
  - 先按固定长度确定候选切点。
  - 切点落在句中则向后延伸到句末。
  - 遇到超长句有安全上限，必要时硬切。
  - 文本存在 `##/###` 时先按小节切开，再在节内分块。
  - 下一块起点至少保留 `chunk_overlap`，并尽量对齐句首。

---

## 4. metadata 规则

每个 `DocumentChunk` 都会写入 `metadata`（字符串键值对）。

默认由 `source` 推导：

- `source`：原始来源标识（原样保留）
- `company`：从 `data/raw/finance/<公司>/...` 或 `公司/文件` 推导
- `date`：从文件名中的 `_YYYY` 或 `YYYY年` 提取
- `doc_type`：
  - 含 `半年度报告/半年报/半年度` -> `半年报`
  - 含 `年度报告` 或含 `年报`（且非半年） -> `年报`
  - 兜底按扩展名（如 `markdown`/`text`/`pdf`）

请求体可传 `metadata` 对象覆盖同名字段。

---

## 5. 去重策略与默认值

### 5.1 哈希与归一化

- 哈希算法：`sha256(normalized_text)`
- `normalized_text`：将连续空白压缩为单空格后再哈希

### 5.2 默认行为

- **同一次 ingest 内去重：开启**
- **跨文档去重：关闭（默认）**
  - 需显式传 `dedup_across_docs=true` 才开启

### 5.3 跨文档去重的当前实现特点

- 跨文档哈希集合是**进程内内存态**。
- 重启 `main.py` 后会重置（与当前内存检索索引一致）。

---

## 6. 统计字段口径（必须统一）

`POST /api/ingest` 成功响应中的字段含义：

- `total_chunks`：原始分块数（去重前）
- `deduplicated_chunks`：去重后实际写入数
- `dropped_duplicates`：被去重丢弃数

关系：`total_chunks = deduplicated_chunks + dropped_duplicates`

---

## 7. 批量入库脚本口径

`scripts/run_ingest.py` 会输出：

- 扫描文档数
- 成功/失败文档数
- 总原始块数（各文档 `total_chunks` 之和）
- 总入库块数（各文档 `deduplicated_chunks` 之和）
- 总去重块数（各文档 `dropped_duplicates` 之和）
- 失败列表（非空则退出码 1）

---

## 8. 验收与排查命令

### 8.1 标准入库

```powershell
python scripts/run_ingest.py
```

### 8.2 单次请求覆盖参数（仅本次生效）

```powershell
python -c "import requests, pathlib, json; p=pathlib.Path(r'e:\CursorProject\myagent\data\raw\finance\贵州茅台\年报_2024.md'); r=requests.post('http://127.0.0.1:7860/api/ingest', json={'doc_id':'demo','source':'贵州茅台/年报_2024.md','content':p.read_text(encoding='utf-8'),'chunk_size':300,'chunk_overlap':60}, timeout=120); print(r.status_code); print(json.dumps(r.json(), ensure_ascii=False, indent=2))"
```

### 8.3 验证跨文档去重开关差异

```powershell
python -c "import requests, json; u='http://127.0.0.1:7860/api/ingest'; c='重复测试内容。'*80; a=requests.post(u,json={'doc_id':'a','source':'x/a.md','content':c,'dedup_across_docs':False}).json(); b=requests.post(u,json={'doc_id':'b','source':'x/b.md','content':c,'dedup_across_docs':True}).json(); print(json.dumps({'off_then_on':[a,b]}, ensure_ascii=False, indent=2))"
```

### 8.4 常见现象说明

- `deduplicated_chunks` 与 `total_chunks` 相同：说明该次无重复块（正常）。
- `retrieval_hit_count=0`：优先排查是否重启后未重新入库（当前检索索引是内存态）。
- `/api/rag` 报 `rag_upstream_http_error`：通常是模型服务问题，不等于入库失败。

---

## 9. 交接给 Day 3 的最小前提

- ingestion 字段契约已稳定：`chunk_id/doc_id/text/source/metadata`。
- `/api/ingest` 参数契约已稳定：`metadata`、`chunk_size`、`chunk_overlap`、`dedup_across_docs`。
- 去重统计口径已稳定：`total_chunks/deduplicated_chunks/dropped_duplicates`。

检索侧可在不改这些字段名的前提下继续优化召回与重排质量。
