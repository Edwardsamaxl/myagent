from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .schemas import EvalRecord

logger = logging.getLogger(__name__)

# 在线 / 离线聚合共用同一套键名（口径统一）
METRIC_TOTAL_REQUESTS = "total_requests"
METRIC_AVG_LATENCY_MS = "avg_latency_ms"
METRIC_AVG_ESTIMATED_COST_USD = "avg_estimated_cost_usd"
METRIC_SUBSTRING_MATCH_RATE = "substring_match_rate"


def _row_has_substring_match(row: dict[str, Any]) -> bool | None:
    v = row.get("substring_match")
    if v is None:
        return None
    return bool(v)


def aggregate_eval_rows(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """从 JSON 行字典列表聚合指标；在线 `eval_records.jsonl` 与离线批评估输出共用此函数。"""
    if not rows:
        return {
            METRIC_TOTAL_REQUESTS: 0,
            METRIC_AVG_LATENCY_MS: 0.0,
            METRIC_AVG_ESTIMATED_COST_USD: 0.0,
            METRIC_SUBSTRING_MATCH_RATE: None,
        }
    total = len(rows)
    avg_latency = sum(float(r.get("latency_ms", 0)) for r in rows) / total
    avg_cost = sum(float(r.get("estimated_cost_usd", 0.0)) for r in rows) / total
    match_flags = [_row_has_substring_match(r) for r in rows]
    explicit = [m for m in match_flags if m is not None]
    match_rate: float | None = None
    if explicit:
        match_rate = round(sum(1 for m in explicit if m) / len(explicit), 4)
    return {
        METRIC_TOTAL_REQUESTS: total,
        METRIC_AVG_LATENCY_MS: round(avg_latency, 2),
        METRIC_AVG_ESTIMATED_COST_USD: round(avg_cost, 6),
        METRIC_SUBSTRING_MATCH_RATE: match_rate,
    }


def load_eval_rows_from_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 评估记录（每行一个对象）。"""
    if not path.exists():
        return []
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


def substring_match(expected: str, actual: str) -> bool:
    """离线指标：参考答案是否以子串形式出现在模型答案中（去空白后比较）。"""
    e = expected.replace(" ", "").replace("\n", "").strip()
    a = actual.replace(" ", "").replace("\n", "").strip()
    if not e:
        return False
    return e in a


class EvaluationStore:
    """在线评估记录持久化；聚合指标与 `aggregate_eval_rows` 口径一致。"""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.file_path.write_text("", encoding="utf-8")

    def append(self, record: EvalRecord) -> None:
        line = json.dumps(asdict(record), ensure_ascii=False)
        with self.file_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def summary(self) -> dict[str, float | int | None]:
        rows = load_eval_rows_from_jsonl(self.file_path)
        return aggregate_eval_rows(rows)


# ---------------------------------------------------------------------------
# RAG 评估指标体系
# ---------------------------------------------------------------------------

@dataclass
class RetrievalEvalRecord:
    """单条检索评估记录（离线批评估使用）。"""
    query: str
    expected_answers: list[str]
    retrieval_hits: list[str]
    hit_ids: list[str] = field(default_factory=list)


@dataclass
class GenerationEvalRecord:
    """单条生成评估记录（离线批评估使用）。"""
    query: str
    evidence: list[str]
    generated_answer: str
    expected_answer: str | None = None


# ---------------------------------------------------------------------------
# 检索指标计算
# ---------------------------------------------------------------------------

def _normalize_numbers(text: str) -> str:
    """Remove comma thousand-separators from numbers for format-agnostic matching."""
    return re.sub(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", lambda m: m.group().replace(",", ""), text)


def recall_at_k(
    hits: list[str],
    expected_answers: list[str],
    k: int,
) -> float:
    """Recall@K：Top-K 检索结果中包含正确答案的比例。

    判断"包含"：expected_answer 子串是否出现在 hit 文本中。
    数字格式归一化：源文本中的千分位逗号会被移除后再匹配，
    例如 "1,234,567" 与 "1234567" 可匹配。
    """
    top_k_hits = hits[:k]
    if not expected_answers:
        return 0.0
    matched = 0
    for expected in expected_answers:
        expected_clean = expected.strip()
        if not expected_clean:
            continue
        expected_norm = _normalize_numbers(expected_clean)
        for hit in top_k_hits:
            hit_norm = _normalize_numbers(hit)
            if expected_norm in hit_norm or expected_clean in hit:
                matched += 1
                break
    return matched / len(expected_answers)


def hit_rate_at_k(
    hits: list[str],
    expected_answers: list[str],
    k: int,
) -> float:
    """HitRate@K：Top-K 中是否存在任意一个命中文档（0 或 1）。"""
    top_k_hits = hits[:k]
    for expected in expected_answers:
        expected_clean = expected.strip()
        if not expected_clean:
            continue
        expected_norm = _normalize_numbers(expected_clean)
        for hit in top_k_hits:
            hit_norm = _normalize_numbers(hit)
            if expected_norm in hit_norm or expected_clean in hit:
                return 1.0
    return 0.0


def mean_reciprocal_rank(
    hits: list[str],
    expected_answers: list[str],
) -> float:
    """MRR：第一个命中的位置权重，1/rank；无命中则 0。"""
    for rank, hit in enumerate(hits, start=1):
        for expected in expected_answers:
            expected_clean = expected.strip()
            if expected_clean:
                hit_norm = _normalize_numbers(hit)
                expected_norm = _normalize_numbers(expected_clean)
                if expected_norm in hit_norm or expected_clean in hit:
                    return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# 生成指标评估（LLM-based）
# ---------------------------------------------------------------------------

class GroundednessEvaluator:
    """基于 LLM 的 Groundedness 评估。

    判断生成回答中有多少比例的内容能被证据支持。
    实现：prompt 让 LLM 判断每个关键陈述是否在 evidence 中有依据，
    返回 0~1 分。
    """

    SYSTEM_PROMPT = (
        "你是一个评估助手。请判断【回答】中的每个关键陈述是否能在【证据】中找到对应依据。\n"
        "如果回答中大部分关键内容都能在证据中找到依据，返回分数接近 1.0；"
        "如果大量内容是幻觉或无法从证据得出，分数接近 0.0。\n"
        "最终返回一个 0~1 的分数，只输出数字，不要解释。"
    )

    USER_PROMPT = "【证据】\n{evidence}\n\n【回答】\n{answer}\n\n评分（0~1）："

    def __init__(self, model: Any) -> None:
        self.model = model

    def evaluate(self, answer: str, evidence: list[str]) -> float:
        if not answer.strip() or not evidence:
            return 0.0
        evidence_text = "\n".join(evidence)
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.USER_PROMPT.format(evidence=evidence_text, answer=answer)},
        ]
        try:
            response = self.model.generate(
                messages=messages,
                temperature=0.0,
                max_tokens=16,
            ).strip()
            # 提取第一个数字
            import re
            nums = re.findall(r"0?\.\d+|\d+", response)
            if nums:
                score = float(nums[0])
                return min(max(score, 0.0), 1.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[GroundednessEvaluator] 评估失败: {exc}")
        return 0.0


class RelevanceEvaluator:
    """基于 LLM 的 Answer Relevance 评估。

    判断生成回答是否切题（与问题相关）。
    实现：prompt 让 LLM 评估回答相对于问题的相关程度。
    """

    SYSTEM_PROMPT = (
        "你是一个评估助手。请判断【回答】是否充分回答了【问题】。\n"
        "如果回答精准切题、覆盖问题核心，返回分数接近 1.0；"
        "如果回答偏离问题、答非所问，分数接近 0.0。\n"
        "最终返回一个 0~1 的分数，只输出数字，不要解释。"
    )

    USER_PROMPT = "【问题】\n{question}\n\n【回答】\n{answer}\n\n评分（0~1）："

    def __init__(self, model: Any) -> None:
        self.model = model

    def evaluate(self, question: str, answer: str) -> float:
        if not answer.strip():
            return 0.0
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.USER_PROMPT.format(question=question, answer=answer)},
        ]
        try:
            response = self.model.generate(
                messages=messages,
                temperature=0.0,
                max_tokens=16,
            ).strip()
            import re
            nums = re.findall(r"0?\.\d+|\d+", response)
            if nums:
                score = float(nums[0])
                return min(max(score, 0.0), 1.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[RelevanceEvaluator] 评估失败: {exc}")
        return 0.0


# ---------------------------------------------------------------------------
# 评估数据集加载
# ---------------------------------------------------------------------------

def load_retrieval_test_set(path: Path) -> list[RetrievalEvalRecord]:
    """加载检索评估集（支持 jsonl 和 json 格式）。"""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        records = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    else:
        records = json.loads(text)
    result = []
    for r in records:
        expected = r.get("expected_answers", [])
        if isinstance(expected, str):
            expected = [expected]
        result.append(RetrievalEvalRecord(
            query=r.get("question", ""),
            expected_answers=expected,
            retrieval_hits=[],  # 填充自 retrieval
            hit_ids=[],
        ))
    return result

