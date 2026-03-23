from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    text: str
    source: str
    # 文档级键示例：source、company、doc_type、date（由入库路径或请求体 metadata 写入）
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class RetrievalHit:
    chunk_id: str
    score: float
    text: str
    source: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class GenerationResult:
    answer: str
    citations: list[str]
    refusal: bool
    reason: str = ""


@dataclass
class EvalRecord:
    """在线请求一条一行写入 `eval_records.jsonl`；与离线批评估行字段对齐（见 evaluation 模块）。"""

    trace_id: str
    question: str
    answer: str
    references: list[str]
    latency_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    run_mode: str = "online"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

