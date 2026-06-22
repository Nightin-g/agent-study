"""LLM client and the minimal Agent loop.

这个文件是整个项目最值得重点阅读的地方。

普通聊天调用通常是：
用户消息 -> 模型 -> 最终回答

Agent 调用多了一个循环：
用户消息 -> 模型 -> 模型请求工具 -> 后端执行工具 -> 工具结果发回模型 -> 最终回答
"""

from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from .config import settings
from .logger import JsonlLogger
from .tools import TOOL_DEFINITIONS, run_tool


# system prompt 是“系统级指令”。
# 它通常用来定义助手的身份、回答风格、工具使用规则。
# 真实产品里，system prompt 是非常重要的产品设计资产。
SYSTEM_PROMPT = """
You are a helpful LLM agent demo for a beginner learning agent development.
Answer in Chinese unless the user asks for another language.
When exact arithmetic or current time is needed, use the provided tools.
When the user asks about local notes or the study plan, use search_notes.
Keep the final answer concise and explain any tool result naturally.
""".strip()


def _build_client() -> AsyncOpenAI:
    """Create an OpenAI-compatible async client for DeepSeek.

    DeepSeek 支持 OpenAI 兼容接口，所以这里可以直接使用 OpenAI SDK。
    关键是把 base_url 改成 DeepSeek 的地址。
    """

    return AsyncOpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    )


async def run_agent(
    user_message: str,
    *,
    use_tools: bool = True,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Run one Agent conversation turn.

    参数说明：
    - user_message: 浏览器传来的用户问题。
    - use_tools: 是否把工具列表发给模型。
    - temperature: 控制回答随机性。

    返回值会被 FastAPI 转成 JSON 返回给浏览器。
    """

    client = _build_client()
    logger = JsonlLogger()

    # messages 是 Chat Completions API 的核心数据结构。
    # role=system: 系统指令
    # role=user: 用户输入
    # role=assistant: 模型回复
    # role=tool: 工具执行结果
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # tool_steps 专门给前端展示学习轨迹，不是调用模型必须的数据。
    tool_steps: list[dict[str, Any]] = []

    # 记录最后一次模型响应信息，方便返回给前端。
    last_model = settings.deepseek_model
    last_usage: dict[str, Any] | None = None

    logger.log(
        "user_input",
        0,
        {
            "content": user_message,
            "use_tools": use_tools,
            "temperature": temperature,
        },
    )

    # Agent 循环：模型可能一次就回答，也可能先请求一个或多个工具。
    for step in range(1, settings.agent_max_steps + 1):
        request: dict[str, Any] = {
            "model": settings.deepseek_model,
            "messages": messages,
            "temperature": temperature,
        }

        if use_tools:
            # 把工具说明书交给模型，模型才知道有哪些工具可以用。
            request["tools"] = TOOL_DEFINITIONS
            request["tool_choice"] = "auto"

        logger.log(
            "model_request",
            step,
            {
                "model": settings.deepseek_model,
                "message_count": len(messages),
                "messages": _messages_for_log(messages),
                "tool_names": _tool_names() if use_tools else [],
                "tool_choice": request.get("tool_choice"),
                "temperature": temperature,
            },
        )

        # 真正调用 DeepSeek API 的地方。
        try:
            response = await client.chat.completions.create(**request)
        except Exception as exc:
            logger.log("error", step, {"where": "model_request", "error": str(exc)})
            raise

        last_model = response.model
        last_usage = response.usage.model_dump(exclude_none=True) if response.usage else None

        assistant_message = response.choices[0].message
        logger.log(
            "model_response",
            step,
            {
                "model": last_model,
                "content": assistant_message.content,
                "tool_calls": _tool_calls_for_log(assistant_message.tool_calls),
                "usage": last_usage,
            },
        )

        # 把模型这一步的输出追加进 messages。
        # 如果模型请求了工具，tool_calls 也必须保留下来，后面 tool 结果要和它对应。
        assistant_payload: dict[str, Any] = {"role": "assistant"}
        if assistant_message.content is not None:
            assistant_payload["content"] = assistant_message.content
        if assistant_message.tool_calls:
            assistant_payload["tool_calls"] = [
                tool_call.model_dump(exclude_none=True) for tool_call in assistant_message.tool_calls
            ]
        messages.append(assistant_payload)

        if use_tools and assistant_message.tool_calls:
            # 模型没有直接给最终答案，而是要求后端调用工具。
            for tool_call in assistant_message.tool_calls:
                name = tool_call.function.name
                arguments, parse_error = _parse_tool_arguments(tool_call.function.arguments)

                logger.log(
                    "tool_call_detected",
                    step,
                    {
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "raw_arguments": tool_call.function.arguments,
                        "arguments": arguments,
                        "parse_error": parse_error,
                    },
                )

                # 执行本地 Python 函数。
                result = f"Tool error: {parse_error}" if parse_error else run_tool(name, arguments)
                logger.log(
                    "tool_result",
                    step,
                    {
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "result": result,
                    },
                )

                # 记录给前端看，帮助你理解 Agent 发生了什么。
                tool_steps.append(
                    {
                        "name": name,
                        "arguments": arguments,
                        "result": result,
                    }
                )

                # 把工具结果放回 messages，再进入下一轮模型调用。
                # tool_call_id 用来告诉模型：这个结果对应刚才哪一次工具请求。
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": result,
                    }
                )

            # 工具结果已经准备好，继续下一轮，让模型基于工具结果生成回答。
            continue

        # 没有工具调用，说明模型已经给出了最终回答。
        answer = (assistant_message.content or "").strip()
        logger.log(
            "final_answer",
            step,
            {
                "answer": answer,
                "tool_step_count": len(tool_steps),
                "log_path": logger.display_path,
            },
        )
        return {
            "answer": answer,
            "model": last_model,
            "tool_steps": tool_steps,
            "usage": last_usage,
            "log_path": logger.display_path,
        }

    # 如果达到最大轮数还没有结束，就返回保护性提示。
    answer = "Agent 已达到最大执行步数，还没有生成最终回答。可以调大 AGENT_MAX_STEPS，或简化你的问题。"
    logger.log(
        "final_answer",
        settings.agent_max_steps,
        {
            "answer": answer,
            "reason": "max_steps",
            "tool_step_count": len(tool_steps),
            "log_path": logger.display_path,
        },
    )
    return {
        "answer": answer,
        "model": last_model,
        "tool_steps": tool_steps,
        "usage": last_usage,
        "log_path": logger.display_path,
    }


def _parse_tool_arguments(raw_arguments: str) -> tuple[dict[str, Any], str | None]:
    """Parse tool arguments returned by the model.

    OpenAI 兼容接口里，tool_call.function.arguments 通常是 JSON 字符串。
    后端要先把它解析成 dict，才能传给 Python 函数。
    """

    try:
        data = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        return {}, "tool arguments must be valid JSON"

    if not isinstance(data, dict):
        return {}, "tool arguments must be a JSON object"

    return data, None


def _tool_names() -> list[str]:
    """Return tool names from the OpenAI-style tool definitions."""

    return [tool["function"]["name"] for tool in TOOL_DEFINITIONS]


def _tool_calls_for_log(tool_calls: Any) -> list[dict[str, Any]]:
    return [
        {
            "id": tool_call.id,
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        }
        for tool_call in (tool_calls or [])
    ]


def _messages_for_log(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_message_for_log(message) for message in messages]


def _message_for_log(message: dict[str, Any]) -> dict[str, Any]:
    payload = dict(message)
    content = payload.get("content")
    if isinstance(content, str):
        payload["content"] = _truncate(content)
    return payload


def _truncate(text: str, limit: int = 3000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"
