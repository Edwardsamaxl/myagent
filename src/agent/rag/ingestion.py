"""文档摄入：清洗 -> 分块。

分块策略（与用户约定一致）：
1. 仍以「固定长度 chunk_size」为**目标切点**（先切到这里再判断）。
2. 若切点落在句中，则**向前延伸**，直到本句结束（句末标点或换行），再落刀。
3. 若整句过长超过安全上限，则退回硬切，避免单块无限长。
4. 若文本含 ##/###，先按小节切开，再对每个小节用上述策略。
5. 重叠区：**至少** chunk_overlap 字符；且下一块起点尽量对齐到**句首**，使重叠里至少包含整句语义（重叠长度可大于 chunk_overlap）。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .schemas import DocumentChunk


def derive_doc_metadata_from_source(source: str) -> dict[str, str]:
    """从 ``source`` 推导文档级元数据（与 ``data/raw/finance/<公司>/<文件>`` 及 ``公司/文件名`` 约定一致）。

    无法解析的键省略；``source`` 始终写入以便追溯。
    """
    if not source or not source.strip():
        return {"source": source}
    s = source.replace("\\", "/").strip()
    parts = [p for p in s.split("/") if p]
    company = ""
    filename = ""
    if "finance" in parts:
        idx = parts.index("finance")
        if idx + 1 < len(parts):
            company = parts[idx + 1]
        if idx + 2 < len(parts):
            filename = parts[idx + 2]
    elif len(parts) >= 2:
        company = parts[0]
        filename = parts[-1]
    elif len(parts) == 1:
        filename = parts[0]

    meta: dict[str, str] = {"source": source}
    if company:
        meta["company"] = company
    if filename:
        stem = Path(filename).stem
        suf = Path(filename).suffix.lower()
        m = re.search(r"_(\d{4})\b", filename) or re.search(r"(\d{4})年", filename)
        if m:
            meta["date"] = m.group(1)
        # 「年度报告」不含连续子串「年报」，需单独匹配；半年度报告优先于年度。
        if "半年度报告" in stem or "半年报" in stem or "半年度" in stem:
            meta["doc_type"] = "半年报"
        elif "年度报告" in stem or ("年报" in stem and "半年" not in stem):
            meta["doc_type"] = "年报"
        elif suf == ".md":
            meta["doc_type"] = "markdown"
        elif suf == ".txt":
            meta["doc_type"] = "text"
        elif suf:
            meta["doc_type"] = suf.lstrip(".")
    return meta


@dataclass
class IngestionResult:
    doc_id: str
    source: str
    total_chunks: int  # 原始分块数（去重前）
    deduplicated_chunks: int  # 去重后实际写入数
    dropped_duplicates: int  # 去重丢弃数


class DocumentIngestionPipeline:
    """分块管道：固定长度为先，再对齐到句末。"""

    # 句末：中文标点 + 换行（财报一行一条时常用）
    _SENTENCE_END = re.compile(r"[。！？；\n]")

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap 不能为负数")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 单块最大长度：避免一句极长时无限延伸（例如无标点长段落）
        self._max_chunk_len = chunk_size * 2
        # 进程内已写入 chunk 的内容哈希（用于可选跨文档去重）
        self._global_chunk_hashes: set[str] = set()

    def ingest_text(
        self,
        doc_id: str,
        source: str,
        content: str,
        doc_metadata: dict[str, str] | None = None,
        *,
        dedup_across_docs: bool = False,
    ) -> tuple[list[DocumentChunk], IngestionResult]:
        cleaned = self._clean_text(content)
        raw_chunks = self._split_text(cleaned)
        deduped_texts, dropped_duplicates = self._deduplicate_chunks(
            raw_chunks, dedup_across_docs=dedup_across_docs
        )
        merged_meta = derive_doc_metadata_from_source(source)
        if doc_metadata:
            for k, v in doc_metadata.items():
                merged_meta[str(k)] = str(v)
        chunks = self._to_document_chunks(
            doc_id=doc_id, source=source, texts=deduped_texts, chunk_metadata=merged_meta
        )
        result = IngestionResult(
            doc_id=doc_id,
            source=source,
            total_chunks=len(raw_chunks),
            deduplicated_chunks=len(chunks),
            dropped_duplicates=dropped_duplicates,
        )
        return chunks, result

    @staticmethod
    def _normalize_for_hash(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _deduplicate_chunks(
        self, texts: list[str], *, dedup_across_docs: bool
    ) -> tuple[list[str], int]:
        local_hashes: set[str] = set()
        kept: list[str] = []
        dropped = 0
        for text in texts:
            normalized = self._normalize_for_hash(text)
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if digest in local_hashes:
                dropped += 1
                continue
            if dedup_across_docs and digest in self._global_chunk_hashes:
                dropped += 1
                continue
            local_hashes.add(digest)
            self._global_chunk_hashes.add(digest)
            kept.append(text)
        return kept, dropped

    @staticmethod
    def _clean_text(text: str) -> str:
        """静态方法：不访问 self，只做纯文本清洗。

        @staticmethod 表示「属于类的工具函数」，第一个参数不是 self。
        这样调用时写成 DocumentIngestionPipeline._clean_text(x) 或 self._clean_text(x) 均可，
        语义上表明：不依赖实例状态。
        """
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _split_text(self, text: str) -> list[str]:
        """总入口：有 Markdown 结构则先分节，每节内再「定长 + 句末」。"""
        if not text:
            return []
        if self._has_structure(text):
            sections = self._split_into_sections(text)
            out: list[str] = []
            for sec in sections:
                out.extend(self._split_fixed_then_sentence_end(sec))
            return [c for c in out if c.strip()]
        return self._split_fixed_then_sentence_end(text)

    def _has_structure(self, text: str) -> bool:
        """是否含 ## / ### 标题行。"""
        return bool(re.search(r"^#{2,3}\s", text, re.MULTILINE))

    def _split_into_sections(self, text: str) -> list[str]:
        """按 ## 或 ### 行切出小节（含标题）；## 前有正文则单独成块。"""
        pattern = re.compile(r"^(#{2,3}\s.+)$", re.MULTILINE)
        matches = list(pattern.finditer(text))
        if not matches:
            return [text] if text.strip() else []

        sections: list[str] = []
        if matches[0].start() > 0:
            prefix = text[: matches[0].start()].strip()
            if prefix:
                sections.append(prefix)
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sec = text[start:end].strip()
            if sec:
                sections.append(sec)
        return sections

    def _split_fixed_then_sentence_end(self, text: str) -> list[str]:
        """核心：从 start 起先取「约 chunk_size」字，再在切点附近对齐到句末，然后按 overlap 滑动。"""
        text = text.strip()
        if not text:
            return []

        chunks: list[str] = []
        start = 0
        n = len(text)

        while start < n:
            # 1) 固定长度切点（目标）
            raw_end = min(start + self.chunk_size, n)

            # 2) 已到文末：整段收尾
            if raw_end >= n:
                piece = text[start:n].strip()
                if piece:
                    chunks.append(piece)
                break

            # 3) 在 raw_end 附近对齐到「句末」：从 raw_end 起向前找第一个句末标点/换行（含在 raw_end 上）
            end = self._extend_cut_to_sentence_end(text, start, raw_end)

            # 4) 防止单块过长：若延伸后仍超上限，退回 raw_end 或再硬切
            if end - start > self._max_chunk_len:
                end = raw_end

            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)

            # 5) 下一块起点：灵活重叠 —— 至少 chunk_overlap，且尽量从「句首」开始（重叠可大于 chunk_overlap）
            next_start = self._overlap_next_start(text, chunk_start=start, end=end)
            if next_start <= start:
                next_start = start + max(1, self.chunk_size // 4)
            start = next_start

        return chunks

    def _overlap_next_start(self, text: str, chunk_start: int, end: int) -> int:
        """计算下一块的起始下标。

        - 重叠区间 [next_start, end) 长度尽量 >= chunk_overlap；
        - 尽量让 next_start 落在「句首」，使与上一块重叠的部分至少包含完整一句；
        - next_start 不得早于当前块起点 chunk_start（避免下一块出现当前块没收到的更早正文）；
        - 若整句过长，无法同时满足「句首 + 最小重叠」，退回 max(min_start, chunk_start)。
        """
        min_start = max(0, end - self.chunk_overlap)
        sent_start = self._sentence_start_before_pos(text, min_start)
        aligned = max(sent_start, chunk_start)
        if end - aligned >= self.chunk_overlap:
            return aligned
        return max(min_start, chunk_start)

    def _sentence_start_before_pos(self, text: str, pos: int) -> int:
        """位置 pos 所在句子的句首下标（句界：。！？；或换行）。"""
        if pos <= 0:
            return 0
        for i in range(min(pos, len(text)) - 1, -1, -1):
            if text[i] in "。！？；\n":
                return i + 1
        return 0

    def _extend_cut_to_sentence_end(self, text: str, chunk_start: int, raw_end: int) -> int:
        """已知候选切点为 raw_end（不含），若落在句中则延伸到句末再切。

        做法：在区间 [raw_end, min(len, raw_end + look_ahead)] 内找第一个句末字符之后的位置；
        若找不到（极少），用 raw_end。
        """
        n = len(text)
        # 从 raw_end-1 开始看：若当前已在句末后，可直接用 raw_end
        if raw_end > 0 and raw_end <= n:
            prev = text[raw_end - 1]
            if prev in "。！？；\n":
                return raw_end

        look = min(n, raw_end + self.chunk_size)
        segment = text[raw_end:look]
        m = self._SENTENCE_END.search(segment)
        if m:
            return raw_end + m.end()

        # 窗口内没有句末：再往后搜一小段，避免一句刚好跨在边界
        look2 = min(n, raw_end + self._max_chunk_len - (raw_end - chunk_start))
        segment2 = text[raw_end:look2]
        m2 = self._SENTENCE_END.search(segment2)
        if m2:
            pos = raw_end + m2.end()
            if pos - chunk_start <= self._max_chunk_len:
                return pos

        return raw_end

    def _to_document_chunks(
        self,
        doc_id: str,
        source: str,
        texts: list[str],
        chunk_metadata: dict[str, str],
    ) -> list[DocumentChunk]:
        """顺序编号为 DocumentChunk（不做内容去重：正常分块几乎不会重复，去重会掩盖异常）。"""
        out: list[DocumentChunk] = []
        for idx, chunk_text in enumerate(texts):
            chunk_id = f"{doc_id}-{idx:04d}"
            out.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=chunk_text,
                    source=source,
                    metadata=dict(chunk_metadata),
                )
            )
        return out
