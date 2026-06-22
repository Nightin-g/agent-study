"""Request and response schemas.

FastAPI 使用 Pydantic 模型来做两件事：
1. 校验请求：比如 message 不能为空，temperature 必须在 0 到 2 之间。
2. 生成文档：打开 /docs 时，FastAPI 会根据这些模型生成接口说明。
"""

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Data sent by the browser to POST /api/chat."""

    # 用户输入的问题。限制长度可以避免误传超大文本。
    message: str = Field(..., min_length=1, max_length=8000)

    # 是否允许模型调用工具。关闭后就退化成普通聊天。
    use_tools: bool = True

    # temperature 控制回答的随机性。
    # 0 更稳定，更适合数学和代码；越高越发散，适合创意写作。
    temperature: float = Field(0.7, ge=0, le=2)


class ToolStep(BaseModel):
    """One tool call made during an Agent run."""

    name: str
    arguments: dict[str, Any]
    result: str


class ChatResponse(BaseModel):
    """Data returned by POST /api/chat."""

    answer: str
    model: str

    # tool_steps 是给学习用的轨迹。
    # 它能让你看到模型调用了什么工具、传了什么参数、拿到了什么结果。
    tool_steps: list[ToolStep] = Field(default_factory=list)

    # usage 通常包含 token 用量。不同模型或 SDK 版本可能返回略有不同。
    usage: dict[str, Any] | None = None

    # 本次请求写入的本地 JSONL 日志路径。
    log_path: str | None = None
