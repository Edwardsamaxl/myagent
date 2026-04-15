from __future__ import annotations

"""工具注册：供 SimpleAgent 多步调用。不负责检索/证据；知识型回答由 RAG（GroundedGenerator）或上层注入的「[检索证据]」提供。"""

import ast
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Semaphore
from typing import Any, Callable

import requests

# 模块级并发控制：最多 3 个并发搜索请求
_search_semaphore = Semaphore(3)

from ..core.memory_store import MemoryStore
from ..core.skill_store import SkillStore
from .schemas import Tool as UnifiedTool, ToolSchema, ToolSource

try:
    from .builders import build_tool
except ImportError:
    build_tool = None  # type: ignore[assignment]

# RAG tool uses the coverage threshold from rag_tool module
try:
    from .rag_tool import create_search_knowledge_base_tool
except ImportError:
    create_search_knowledge_base_tool = None  # type: ignore[assignment]


ToolFunc = Callable[[str], str]


# Backward-compatible Tool dataclass for SimpleAgent (name/description/func interface)
@dataclass
class Tool:
    name: str
    description: str
    func: ToolFunc


@dataclass
class ToolRegistry:
    """Registry for managing tools with built-in protection.

    Tools sourced from ToolSource.BUILTIN cannot be overridden.
    """
    _tools: dict[str, UnifiedTool] = field(default_factory=dict)

    def register_tool(self, tool: UnifiedTool) -> None:
        if tool.name in self._tools:
            existing = self._tools[tool.name]
            if hasattr(existing, "source") and existing.source == ToolSource.BUILTIN:
                raise ValueError(f"Cannot override built-in tool: {tool.name}")
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> UnifiedTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[UnifiedTool]:
        return list(self._tools.values())


def get_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class _MathEvaluator(ast.NodeVisitor):
    """纯 AST 数学表达式求值器，无 eval() 调用。"""

    def visit_Constant(self, node: ast.Constant) -> float:
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"不支持的常量类型: {type(node.value)}")

    def visit_BinOp(self, node: ast.BinOp) -> float:
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        if op_type is ast.Add:
            return left + right
        if op_type is ast.Sub:
            return left - right
        if op_type is ast.Mult:
            return left * right
        if op_type is ast.Div:
            if right == 0:
                raise ValueError("除零错误。")
            return left / right
        if op_type is ast.Mod:
            return left % right
        if op_type is ast.Pow:
            return left ** right
        raise ValueError(f"不支持的二元操作符: {op_type.__name__}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        operand = self.visit(node.operand)
        op_type = type(node.op)
        if op_type is ast.UAdd:
            return +operand
        if op_type is ast.USub:
            return -operand
        raise ValueError(f"不支持的一元操作符: {op_type.__name__}")


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
            raise ValueError(f"只允许基础算术表达式，发现: {type(sub_node).__name__}")

    evaluator = _MathEvaluator()
    return evaluator.visit(node.body)


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


def _create_rag_tool_from_rag_service(rag_service: Any) -> Tool:
    """Create knowledge base search tool using rag_tool module.

    This tool wraps the retrieval pipeline (intent classify, query rewrite,
    retrieval, rerank) and returns formatted evidence blocks without LLM generation.
    """
    if create_search_knowledge_base_tool is None:
        # Fallback if rag_tool module unavailable
        def search_knowledge_base(query: str) -> str:
            return "RAG工具不可用：rag_tool模块加载失败。"

        return Tool(
            name="search_knowledge_base",
            description="在知识库中检索相关信息。输入查询字符串，返回相关文档片段。",
            func=search_knowledge_base,
        )

    name, description, func = create_search_knowledge_base_tool(rag_service)
    return Tool(name=name, description=description, func=func)


def _safe_web_search_with_retry(query: str, max_retries: int = 3) -> str:
    """执行网络搜索，使用 DuckDuckGo Instant Answer API（免费无需 key）。

    包含并发控制（Semaphore 3）、重试（最多 3 次）和指数退避。
    """
    import urllib.parse

    # 并发控制
    acquired = _search_semaphore.acquire(timeout=15)
    if not acquired:
        return "搜索请求超时：系统繁忙，请稍后重试。"

    try:
        encoded_query = urllib.parse.quote(query)
        # DuckDuckGo Instant Answer API
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_redirect=1"

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                resp.encoding = "utf-8"
                data = resp.json()
                break  # 成功，跳出重试循环
            except requests.RequestException as exc:
                last_error = exc
                if attempt < max_retries:
                    # 指数退避: 0.5s, 1s, 2s...
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                return f"搜索请求失败: {last_error}"

        # 处理结果（与原逻辑一致）
        results = []
        for topic in data.get("RelatedTopics", [])[:5]:
            if "Text" in topic and "FirstURL" in topic:
                results.append({
                    "title": topic.get("Text", "")[:100],
                    "url": topic.get("FirstURL", ""),
                    "snippet": "",
                })

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
    finally:
        _search_semaphore.release()


def default_tools(
    memory_store: MemoryStore,
    skill_store: SkillStore,
    workspace_dir: Path,
    rag_service: Any = None,
) -> dict[str, Tool]:
    """Build the default tool set for SimpleAgent.

    Returns dict[str, Tool] for backward compatibility with SimpleAgent's
    name/description/func interface.
    """
    def read_memory() -> str:
        return memory_store.read()

    def remember(note: str) -> str:
        return memory_store.append(note)

    def list_skills() -> str:
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
            func=_safe_web_search_with_retry,
        ),
    }
    # 如果传入了 rag_service，添加知识库检索工具
    if rag_service is not None:
        tools["search_knowledge_base"] = _create_rag_tool_from_rag_service(rag_service)
    return tools
