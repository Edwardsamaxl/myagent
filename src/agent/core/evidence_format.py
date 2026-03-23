"""证据块与引用行格式：与 `RetrievalHit` 及 `rag_agent_service.answer` 返回的 retrieval_hits 字段一致。

若 `src/agent/core/retrieval.py` 或检索管线返回结构变化，须先与检索侧对齐，再同步本模块与 GroundedGenerator 的 prompt。
"""

from __future__ import annotations

from typing import Any

from .schemas import RetrievalHit


def format_citation_lines(hits: list[RetrievalHit]) -> list[str]:
    """与证据序号 [1]..[n] 一一对应的引用行，供 API citations / EvalRecord.references。"""
    lines: list[str] = []
    for idx, hit in enumerate(hits, start=1):
        lines.append(f"[{idx}] chunk_id={hit.chunk_id} score={hit.score} source={hit.source}")
    return lines


def format_evidence_block_from_hits(hits: list[RetrievalHit]) -> str:
    """供生成模型使用的完整块文本（每条含全文 text）。"""
    parts: list[str] = []
    for idx, hit in enumerate(hits, start=1):
        parts.append(
            f"[{idx}] chunk_id={hit.chunk_id} score={hit.score} source={hit.source}\n"
            f"内容:\n{hit.text}"
        )
    return "\n\n".join(parts)


def format_evidence_block_from_api_dicts(hits: list[dict[str, Any]]) -> str:
    """供对话层注入用户消息：结构与 answer() 的 retrieval_hits 一致，正文为 text_preview。"""
    parts: list[str] = []
    for idx, h in enumerate(hits, start=1):
        chunk_id = h.get("chunk_id", "")
        score = h.get("score", "")
        source = h.get("source", "")
        metadata = h.get("metadata", {})
        preview = h.get("text_preview", "")
        if isinstance(metadata, dict) and metadata:
            meta_text = " ".join(f"{k}={v}" for k, v in metadata.items())
        else:
            meta_text = "无"
        parts.append(
            f"[{idx}] chunk_id={chunk_id} score={score} source={source}\n"
            f"metadata: {meta_text}\n"
            f"片段（预览）:\n{preview}"
        )
    return "\n\n".join(parts)
