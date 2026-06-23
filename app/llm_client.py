"""Compatibility entry point for running the Agent.

真正的 LangGraph 编排在 app/agent_graph.py。
这里保留 run_agent()，让 FastAPI 入口和前端接口不用跟着大改。
"""

from __future__ import annotations

from typing import Any

from .agent_graph import run_langgraph_agent


async def run_agent(
    user_message: str,
    *,
    use_tools: bool = True,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Run one Agent conversation turn."""

    return await run_langgraph_agent(
        user_message,
        use_tools=use_tools,
        temperature=temperature,
    )
