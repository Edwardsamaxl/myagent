from __future__ import annotations

from pathlib import Path


DEFAULT_SKILL = """# SKILL: Example

## Purpose
演示技能文件格式。

## Steps
1. 读取需求
2. 给出可执行步骤
3. 生成结果
"""


class SkillStore:
    def __init__(self, skills_dir: Path) -> None:
        self.skills_dir = skills_dir
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        sample = self.skills_dir / "example.md"
        if not sample.exists():
            sample.write_text(DEFAULT_SKILL, encoding="utf-8")

    def list_skills(self) -> list[str]:
        return sorted(p.stem for p in self.skills_dir.glob("*.md"))

    def read_skill(self, name: str) -> str:
        path = self.skills_dir / f"{name}.md"
        if not path.exists():
            return f"技能不存在: {name}"
        return path.read_text(encoding="utf-8")

    def upsert_skill(self, name: str, content: str) -> str:
        name = name.strip().replace(" ", "_")
        if not name:
            return "技能名为空。"
        path = self.skills_dir / f"{name}.md"
        path.write_text(content.strip() + "\n", encoding="utf-8")
        return f"技能已保存: {path.name}"

    def render_for_prompt(self, max_chars: int = 4000) -> str:
        chunks: list[str] = []
        for skill in self.list_skills():
            text = self.read_skill(skill)
            chunks.append(f"## {skill}\n{text}")
        merged = "\n\n".join(chunks).strip()
        if len(merged) > max_chars:
            return merged[:max_chars] + "\n...[技能内容已截断]"
        return merged
