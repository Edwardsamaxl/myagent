from __future__ import annotations

"""工具注册：供 SimpleAgent 多步调用。不负责检索/证据；知识型回答由 RAG（GroundedGenerator）或上层注入的「[检索证据]」提供。"""

import ast
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from ..core.memory_store import MemoryStore
from ..core.skill_store import SkillStore


ToolFunc = Callable[[str], str]


@dataclass
class Tool:
    name: str
    description: str
    func: ToolFunc


def get_time(_: str) -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _safe_eval_math(expr: str) -> float:
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Constant,
    )

    node = ast.parse(expr, mode="eval")
    for sub_node in ast.walk(node):
        if not isinstance(sub_node, allowed_nodes):
            raise ValueError("只允许基础算术表达式。")

    return float(eval(compile(node, "<calc>", "eval"), {"__builtins__": {}}))


def calculate(expr: str) -> str:
    expr = expr.strip()
    if not expr:
        return "表达式为空。"
    try:
        result = _safe_eval_math(expr)
    except Exception as exc:  # noqa: BLE001
        return f"计算失败: {exc}"
    return str(result)


def _safe_path(path_value: str, workspace_dir: Path) -> Path:
    candidate = (workspace_dir / path_value.strip()).resolve()
    if workspace_dir.resolve() not in candidate.parents and candidate != workspace_dir.resolve():
        raise ValueError("仅允许访问 workspace 目录。")
    return candidate


def default_tools(
    memory_store: MemoryStore,
    skill_store: SkillStore,
    workspace_dir: Path,
) -> dict[str, Tool]:
    def read_memory(_: str) -> str:
        return memory_store.read()

    def remember(note: str) -> str:
        return memory_store.append(note)

    def list_skills(_: str) -> str:
        skills = skill_store.list_skills()
        if not skills:
            return "暂无技能。"
        return "\n".join(f"- {item}" for item in skills)

    def read_skill(name: str) -> str:
        return skill_store.read_skill(name.strip())

    def save_skill(raw_input: str) -> str:
        if "::" not in raw_input:
            return "格式错误。请使用: 技能名::技能内容"
        name, content = raw_input.split("::", 1)
        return skill_store.upsert_skill(name, content)

    def read_workspace_file(path_value: str) -> str:
        try:
            file_path = _safe_path(path_value, workspace_dir)
        except ValueError as exc:
            return f"读取失败: {exc}"
        if not file_path.exists():
            return f"文件不存在: {file_path.name}"
        if file_path.is_dir():
            return "目标是目录，不是文件。"
        return file_path.read_text(encoding="utf-8")

    return {
        "get_time": Tool(
            name="get_time",
            description="获取当前本地时间，输入可为空字符串。",
            func=get_time,
        ),
        "calculate": Tool(
            name="calculate",
            description="计算基础算术表达式，例如: (2+3)*4。",
            func=calculate,
        ),
        "read_memory": Tool(
            name="read_memory",
            description="读取长期记忆 MEMORY.md，输入为空。",
            func=read_memory,
        ),
        "remember": Tool(
            name="remember",
            description="追加一条长期记忆到 MEMORY.md，输入是记忆内容。",
            func=remember,
        ),
        "list_skills": Tool(
            name="list_skills",
            description="列出所有技能名，输入为空。",
            func=list_skills,
        ),
        "read_skill": Tool(
            name="read_skill",
            description="读取指定技能，输入技能名（不含 .md）。",
            func=read_skill,
        ),
        "save_skill": Tool(
            name="save_skill",
            description="保存或更新技能，输入格式: 技能名::技能内容。",
            func=save_skill,
        ),
        "read_workspace_file": Tool(
            name="read_workspace_file",
            description="读取 workspace 内文件，输入相对路径。",
            func=read_workspace_file,
        ),
    }
