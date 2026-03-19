from __future__ import annotations

import json
from pathlib import Path

from ..llm.providers import Message


class SessionStore:
    def __init__(self, sessions_file: Path) -> None:
        self.sessions_file = sessions_file
        self.sessions_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.sessions_file.exists():
            self.sessions_file.write_text("{}", encoding="utf-8")

    def _load(self) -> dict[str, list[Message]]:
        try:
            return json.loads(self.sessions_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _save(self, data: dict[str, list[Message]]) -> None:
        self.sessions_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def list_session_ids(self) -> list[str]:
        return sorted(self._load().keys())

    def get_history(self, session_id: str) -> list[Message]:
        data = self._load()
        return data.get(session_id, [])

    def set_history(self, session_id: str, messages: list[Message]) -> None:
        filtered = [m for m in messages if m.get("role") in {"user", "assistant"}]
        data = self._load()
        data[session_id] = filtered[-40:]
        self._save(data)
