from __future__ import annotations

"""工具注册：供 SimpleAgent 多步调用。不负责检索/证据；知识型回答由 RAG（GroundedGenerator）或上层注入的「[检索证据]」提供。"""

import ast
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import requests

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


def create_rag_tool(rag_service: Any) -> Tool:
    """创建知识库检索工具，让 Agent 自主决定何时检索。"""

    def search_knowledge_base(query: str) -> str:
        if not query.strip():
            return "查询字符串为空。"
        try:
            result = rag_service.answer(query, append_to_eval_store=False)
            hits = result.get("retrieval_hits", []) or []
            if not hits:
                return "检索结果为空，无法找到相关信息。"

            lines = []
            for i, hit in enumerate(hits[:5], 1):
                preview = (hit.get("text_preview") or hit.get("text", ""))[:300]
                source = hit.get("source", "unknown")
                score = hit.get("score", 0)
                lines.append(f"[{i}] 来源: {source} (相关度: {score:.2f})")
                lines.append(f"内容: {preview}")
                lines.append("")
            return "\n".join(lines)
        except Exception as exc:  # noqa: BLE001
            return f"检索失败: {exc}"

    return Tool(
        name="search_knowledge_base",
        description="在知识库中检索相关信息。输入查询字符串，返回相关文档片段。用于回答需要事实依据的问题，如数据、指标、事件等。",
        func=search_knowledge_base,
    )


def _safe_web_search(query: str) -> str:
    """执行网络搜索，使用 DuckDuckGo Instant Answer API（免费无需 key）。"""
    import urllib.parse

    encoded_query = urllib.parse.quote(query)
    # DuckDuckGo Instant Answer API
    url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_redirect=1"

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        # 强制使用 UTF-8 编码处理响应
        resp.encoding = "utf-8"
        data = resp.json()
    except requests.RequestException as exc:
        return f"搜索请求失败: {exc}"

    results = []
    # 优先取 RelatedTopics（新闻/实体）
    for topic in data.get("RelatedTopics", [])[:5]:
        if "Text" in topic and "FirstURL" in topic:
            results.append({
                "title": topic.get("Text", "")[:100],
                "url": topic.get("FirstURL", ""),
                "snippet": "",
            })

    # 如果没有 RelatedTopics，尝试取 Abstract
    if not results and data.get("AbstractText"):
        abstract = data.get("AbstractText", "")
        source = data.get("AbstractURL", "")
        return f"{abstract}\n\n来源: {source}"

    if not results:
        return "未找到相关搜索结果。"

    lines = []
    for i, r in enumerate(results[:5], 1):
        lines.append(f"[{i}] {r['title']}")
        if r["snippet"]:
            lines.append(f"    {r['snippet']}")
        lines.append(f"    链接: {r['url']}")
        lines.append("")
    return "\n".join(lines)


def default_tools(
    memory_store: MemoryStore,
    skill_store: SkillStore,
    workspace_dir: Path,
    rag_service: Any = None,
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

    tools = {
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
        "web_search": Tool(
            name="web_search",
            description="搜索互联网，返回相关网页摘要。用于查询实时信息、新闻、公开数据等无法从本地知识库获取的内容。",
            func=_safe_web_search,
        ),
    }
    # 如果传入了 rag_service，添加知识库检索工具
    if rag_service is not None:
        tools["search_knowledge_base"] = create_rag_tool(rag_service)
    return tools
