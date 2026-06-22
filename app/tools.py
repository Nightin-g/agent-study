"""Tools that the Agent can call.

在 Agent 应用里，tool 的本质通常就是一个普通函数。
关键区别是：我们会把函数的名字、说明、参数结构告诉模型。
模型看完说明后，可以决定是否调用这个函数，以及传什么参数。
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import json
import operator
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTES_FILE = PROJECT_ROOT / "notes" / "sample_notes.txt"


# 这段 TOOL_DEFINITIONS 会被发送给 LLM。
# 它不是工具的实现，而是工具的“说明书”：
# - name: 模型调用工具时使用的名字
# - description: 告诉模型什么时候应该调用它
# - parameters: 告诉模型需要传哪些参数，格式类似 JSON Schema
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time for a timezone. Use this when the user asks about time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone_name": {
                        "type": "string",
                        "description": "IANA timezone name, for example Asia/Shanghai or UTC.",
                        "default": "Asia/Shanghai",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calculate a basic arithmetic expression. Use this for exact math.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Expression with numbers, parentheses, and + - * / // % **.",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "Search the local sample notes file by keyword. Use this when the user asks about local notes or the study plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Keyword to search in notes/sample_notes.txt.",
                    }
                },
                "required": ["keyword"],
                "additionalProperties": False,
            },
        },
    },
]


def get_current_time(timezone_name: str = "Asia/Shanghai") -> str:
    """Return current time in the requested timezone.

    这是一个很适合入门的工具示例：
    模型自己不知道“此刻”的精确本地时间，所以它需要调用外部工具。
    """

    try:
        tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        # 如果用户传了无效时区，尽量回退到 UTC 或系统本地时区，避免整个请求失败。
        tz = timezone.utc if timezone_name.upper() == "UTC" else datetime.now().astimezone().tzinfo

    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


# ast 解析后，数学表达式会变成语法树。
# 我们只允许下面这些运算，避免执行任意 Python 代码。
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def calculate(expression: str) -> str:
    """Safely calculate a basic arithmetic expression.

    注意这里没有用 eval(expression)。
    eval 可以执行任意 Python 代码，不适合处理来自用户或模型的输入。
    这里用 ast 白名单，只允许数字和基础数学运算。
    """

    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Expression is required.")
    if len(expression) > 200:
        raise ValueError("Expression is too long.")

    tree = ast.parse(expression, mode="eval")
    result = _eval_math_node(tree)
    return f"{expression} = {result}"


def _eval_math_node(node: ast.AST) -> float | int:
    """Recursively evaluate a safe math AST node."""

    if isinstance(node, ast.Expression):
        return _eval_math_node(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_math_node(node.operand))

    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_math_node(node.left)
        right = _eval_math_node(node.right)

        # 加一点保护，避免 9 ** 999999 这类表达式拖垮本地机器。
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("Exponent is too large.")

        result = _BIN_OPS[type(node.op)](left, right)
        if abs(result) > 1_000_000_000_000:
            raise ValueError("Result is too large.")
        return result

    raise ValueError("Only basic arithmetic expressions are supported.")


def search_notes(keyword: str) -> str:
    """Search the local sample notes file by keyword.

    这里固定只读 notes/sample_notes.txt，避免把工具变成任意文件读取器。
    返回 JSON 字符串，是为了让模型更容易理解结构化结果。
    """

    if not isinstance(keyword, str) or not keyword.strip():
        return _json_error("keyword is required")
    if not NOTES_FILE.exists():
        return _json_error("notes file not found")

    keyword = keyword.strip()
    keyword_lower = keyword.lower()
    matches: list[dict[str, Any]] = []

    with NOTES_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            text = line.rstrip("\n")
            if keyword_lower in text.lower():
                matches.append({"line_number": line_number, "text": text})

    return _json_ok({"keyword": keyword, "count": len(matches), "matches": matches})


def _json_ok(data: dict[str, Any]) -> str:
    return json.dumps({"ok": True, **data}, ensure_ascii=False)


def _json_error(error: str) -> str:
    return json.dumps({"ok": False, "error": error}, ensure_ascii=False)


ToolFunction = Callable[..., str]


TOOL_REGISTRY: dict[str, ToolFunction] = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "search_notes": search_notes,
}


def run_tool(name: str, arguments: dict[str, Any]) -> str:
    """Dispatch a model-requested tool call to the matching Python function.

    LLM 返回的 tool call 只有字符串形式的函数名和 JSON 参数。
    真实项目里通常会维护一个 tool registry，这里也采用同样的结构。
    """

    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        return f"Unknown tool: {name}"
    if not isinstance(arguments, dict):
        return "Tool error: arguments must be a JSON object."

    try:
        return tool(**arguments)
    except Exception as exc:
        # 工具失败时，把错误也作为工具结果返回给模型。
        # 这样模型可以根据错误信息解释原因，而不是整个后端直接崩掉。
        return f"Tool error: {exc}"
