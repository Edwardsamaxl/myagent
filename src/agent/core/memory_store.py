from __future__ import annotations

from datetime import datetime
from pathlib import Path


DEFAULT_MEMORY_HEADER = """# MEMORY

这里记录 Agent 的长期记忆。建议只写：
- 用户偏好
- 重要决策
- 长期目标
"""


class MemoryStore:
    def __init__(self, memory_file: Path) -> None:
        self.memory_file = memory_file
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.memory_file.exists():
            self.memory_file.write_text(DEFAULT_MEMORY_HEADER, encoding="utf-8")

    def read(self) -> str:
        return self.memory_file.read_text(encoding="utf-8").strip()

    def append(self, note: str) -> str:
        note = note.strip()
        if not note:
            return "记忆内容为空，未写入。"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.memory_file.open("a", encoding="utf-8") as f:
            f.write(f"\n\n## {timestamp}\n- {note}\n")
        return "已写入 MEMORY.md。"
