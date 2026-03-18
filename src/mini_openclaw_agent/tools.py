from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime
from typing import Callable


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


def default_tools() -> dict[str, Tool]:
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
    }

