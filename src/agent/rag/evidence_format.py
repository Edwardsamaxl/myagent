"""证据块与引用行格式：与 `RetrievalHit` 及 `rag_agent_service.answer` 返回的 retrieval_hits 字段一致。

若检索管线返回结构变化，须先与检索侧对齐，再同步本模块与 GroundedGenerator 的 prompt。
"""

from __future__ import annotations

import re
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


_CITATION_MARK_RE = re.compile(r"\[(\d+)\]")
_ANCHOR_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{2,}|\d+(?:\.\d+)?%?")
_PUNCT_RE = re.compile(r"[，。；：、,.!?（）()【】\[\]\s]+")
_STOPWORDS = {
    "公司",
    "多少",
    "什么",
    "如何",
    "是否",
    "以及",
    "这个",
    "那个",
    "根据",
    "报告",
    "数据",
    "其中",
}
_LOW_INFO_PHRASES = (
    "年度报告全文",
    "半年度报告全文",
    "目录",
    "释义",
    "公司简介",
)


def citation_ids_for_hits(hits: list[RetrievalHit]) -> list[str]:
    """按证据顺序返回稳定引用编号，如 ['[1]', '[2]']。"""
    return [f"[{idx}]" for idx, _ in enumerate(hits, start=1)]


def select_evidence_hits(
    question: str,
    hits: list[RetrievalHit],
    max_evidence: int = 3,
) -> list[RetrievalHit]:
    """从 rerank 命中中选择用于生成的证据集合（稳定顺序）。"""
    if not hits:
        return []
    max_evidence = max(1, max_evidence)
    q_anchors = set(_extract_anchor_tokens(question))
    if not q_anchors:
        return hits[:max_evidence]

    scored: list[tuple[int, int, RetrievalHit]] = []
    for idx, hit in enumerate(hits):
        text = hit.text.lower()
        overlap = sum(1 for token in q_anchors if token.lower() in text)
        scored.append((overlap, -idx, hit))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

    selected = [item[2] for item in scored[:max_evidence] if item[0] > 0]
    if not selected:
        # 若锚点都不匹配，至少保留 top1，交由 generation 走 insufficient_evidence 分流。
        return hits[:1]
    return selected


def contains_citation_marker(text: str) -> bool:
    """判断回答正文是否含至少一个引用编号标记。"""
    return bool(_CITATION_MARK_RE.search(text))


def extract_citation_indices(text: str) -> list[int]:
    """提取回答中的引用编号，如 [1][3] -> [1, 3]。"""
    return [int(m.group(1)) for m in _CITATION_MARK_RE.finditer(text)]


def citations_are_valid(text: str, max_index: int) -> bool:
    """引用合法性：存在且全部落在 [1, max_index]。"""
    indices = extract_citation_indices(text)
    if not indices:
        return False
    return all(1 <= idx <= max_index for idx in indices)


def pick_key_evidence_snippet(
    question: str,
    hits: list[RetrievalHit],
    max_chars: int = 120,
) -> str:
    """抽取更可读的关键依据片段：优先包含问题锚点与数字信息的句子。"""
    if not hits:
        return ""
    q_tokens = _extract_anchor_tokens(question)
    candidates: list[str] = []
    for hit in hits[:3]:
        text = hit.text.replace("\r", "\n")
        for seg in re.split(r"[。！？\n]", text):
            seg = seg.strip()
            if len(seg) < 8:
                continue
            if "|" in seg and seg.count("|") >= 3:
                # 降低表格噪声权重
                continue
            candidates.append(seg)

    if not candidates:
        text = _PUNCT_RE.sub(" ", hits[0].text).strip()
        return text[:max_chars]

    def _score(seg: str) -> tuple[int, int, int]:
        lower = seg.lower()
        overlap = sum(1 for t in q_tokens if t.lower() in lower)
        has_digit = 1 if re.search(r"\d", seg) else 0
        length_penalty = abs(len(seg) - 40)
        return overlap, has_digit, -length_penalty

    best = max(candidates, key=_score)
    clean = _PUNCT_RE.sub(" ", best).strip()
    return clean[:max_chars]


def is_low_information_snippet(snippet: str) -> bool:
    """判断片段是否仅为目录/标题类低信息文本。"""
    text = snippet.strip()
    if len(text) < 10:
        return True
    if ("年度报告" in text or "半年度报告" in text) and sum(ch.isdigit() for ch in text) < 2:
        return True
    if sum(ch.isdigit() for ch in text) == 0 and all(p in text for p in ("公司", "报告")):
        return True
    return any(p in text for p in _LOW_INFO_PHRASES)


def evaluate_anchor_coverage(question: str, answer: str, hits: list[RetrievalHit]) -> tuple[float, dict[str, int]]:
    """轻量锚点覆盖率：数字/年份/百分比/关键实体在证据文本中的覆盖程度。"""
    evidence_text = "\n".join(hit.text for hit in hits).lower()
    q_anchors = _extract_anchor_tokens(question)
    a_anchors = _extract_anchor_tokens(answer)
    # 关键断言以回答为主，问题锚点作补充
    anchors = list(dict.fromkeys(a_anchors + q_anchors))
    if not anchors:
        return 1.0, {"anchors": 0, "covered": 0}
    covered = sum(1 for token in anchors if token.lower() in evidence_text)
    return covered / len(anchors), {"anchors": len(anchors), "covered": covered}


def _extract_anchor_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for m in _ANCHOR_TOKEN_RE.finditer(text):
        token = m.group(0).strip()
        if len(token) <= 1:
            continue
        if token in _STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def normalize_structured_answer(raw: str, citation_ids: list[str]) -> str:
    """规范化为三段结构：结论 / 关键依据 / 引用编号。"""
    text = raw.strip()
    if not text:
        text = "无法基于当前证据形成可靠结论。"

    # 若模型已给出结构，仅补齐“引用编号”段。
    if "结论" in text and "关键依据" in text:
        if "引用编号" not in text:
            refs = "".join(citation_ids) if citation_ids else "无"
            return f"{text}\n\n引用编号：{refs}"
        return text

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    conclusion = lines[0] if lines else text
    key_evidence = "；".join(lines[1:3]) if len(lines) > 1 else "证据见检索命中片段。"
    refs = "".join(citation_ids) if citation_ids else "无"
    return (
        f"结论：{conclusion}\n"
        f"关键依据：{key_evidence}\n"
        f"引用编号：{refs}"
    )
