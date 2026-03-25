from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SessionTaskState:
    """会话级任务态（旁路持久化，不参与 40 条消息截断）。"""

    phase: str = "idle"  # idle | awaiting_clarification | active
    pending_context: str = ""  # 澄清前用户原问，供下一轮拼接
    last_intent: str | None = None
    last_plan_summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SessionTaskState:
        return cls(
            phase=str(d.get("phase", "idle")),
            pending_context=str(d.get("pending_context", "")),
            last_intent=d.get("last_intent"),
            last_plan_summary=d.get("last_plan_summary"),
        )


class SessionMetaStore:
    """sessions_meta.json：session_id -> SessionTaskState。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}", encoding="utf-8")

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _save(self, data: dict[str, dict[str, Any]]) -> None:
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, session_id: str) -> SessionTaskState:
        raw = self._load().get(session_id)
        if not raw:
            return SessionTaskState()
        return SessionTaskState.from_dict(raw)

    def put(self, session_id: str, state: SessionTaskState) -> None:
        data = self._load()
        data[session_id] = state.to_dict()
        self._save(data)

    def delete(self, session_id: str) -> None:
        data = self._load()
        if session_id in data:
            del data[session_id]
            self._save(data)
