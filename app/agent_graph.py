"""LangGraph-based Agent workflow.

这个文件把原来的手写 Agent 循环改成显式图结构：

prepare -> agent -> tools -> agent -> final
                  -> max_steps -> final

学习重点：
- State 保存运行过程中的 messages、工具轨迹和提示信息。
- Node 是一个个小函数，只负责一步。
- Edge 决定下一步走向，工具调用不再靠手写 while/for 控制整体流程。
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph, add_messages
from pydantic import BaseModel, Field

from .config import settings
from .logger import JsonlLogger
from .tools import calculate, get_current_time, run_tool, search_notes


SYSTEM_PROMPT = """
You are a helpful LLM agent demo for a beginner learning agent development.
Answer in Chinese unless the user asks for another language.
When exact arithmetic or current time is needed, use the provided tools.
When the user asks about local notes or the study plan, use search_notes.
Keep the final answer concise and explain any tool result naturally.
""".strip()


MAX_STEPS_ANSWER = "Agent 已达到最大执行步数，还没有生成最终回答。可以调大 AGENT_MAX_STEPS，或简化你的问题。"


class AgentState(TypedDict, total=False):
    """State shared by all LangGraph nodes."""

    messages: Annotated[list[BaseMessage], add_messages]
    intent_hint: str
    tool_steps: list[dict[str, Any]]
    llm_calls: int
    model: str
    usage: dict[str, Any] | None


class TimeArgs(BaseModel):
    timezone_name: str = Field(
        default="Asia/Shanghai",
        description="IANA timezone name, for example Asia/Shanghai or UTC.",
    )


class CalculateArgs(BaseModel):
    expression: str = Field(description="Expression with numbers, parentheses, and + - * / // % **.")


class SearchNotesArgs(BaseModel):
    keyword: str = Field(description="Keyword to search in notes/sample_notes.txt.")


async def run_langgraph_agent(
    user_message: str,
    *,
    use_tools: bool = True,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Run one Agent turn through a compiled LangGraph graph."""

    logger = JsonlLogger()
    logger.log("user_input", 0, _input_log(user_message, use_tools, temperature))

    graph = _compile_graph(logger, use_tools=use_tools, temperature=temperature)
    final_state = await graph.ainvoke(_initial_state(user_message))

    return _build_response(final_state, logger)


def _compile_graph(logger: JsonlLogger, *, use_tools: bool, temperature: float):
    tools = _build_tools() if use_tools else []
    model = _build_model(temperature, tools)

    graph = StateGraph(AgentState)
    graph.add_node("prepare", _prepare_node(logger))
    graph.add_node("agent", _agent_node(logger, model, tools, temperature))
    graph.add_node("tools", _tools_node(logger))
    graph.add_node("max_steps", _max_steps_node(logger))
    graph.add_node("final", _final_node(logger))

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "agent")
    graph.add_conditional_edges("agent", _route_after_agent, ["tools", "max_steps", "final"])
    graph.add_edge("tools", "agent")
    graph.add_edge("max_steps", "final")
    graph.add_edge("final", END)
    return graph.compile()


def _initial_state(user_message: str) -> AgentState:
    return {
        "messages": [HumanMessage(content=user_message)],
        "tool_steps": [],
        "llm_calls": 0,
        "model": settings.deepseek_model,
        "usage": None,
    }


def _build_model(temperature: float, tools: list[BaseTool]):
    model = ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=temperature,
    )
    return model.bind_tools(tools) if tools else model


def _build_tools() -> list[BaseTool]:
    return [
        StructuredTool.from_function(
            name="get_current_time",
            func=get_current_time,
            description="Get the current time for a timezone. Use this when the user asks about time.",
            args_schema=TimeArgs,
        ),
        StructuredTool.from_function(
            name="calculate",
            func=calculate,
            description="Calculate a basic arithmetic expression. Use this for exact math.",
            args_schema=CalculateArgs,
        ),
        StructuredTool.from_function(
            name="search_notes",
            func=search_notes,
            description="Search the local sample notes file by keyword.",
            args_schema=SearchNotesArgs,
        ),
    ]


def _prepare_node(logger: JsonlLogger):
    def prepare(state: AgentState) -> dict[str, Any]:
        hint = _guess_intent(_latest_human_text(state["messages"]))
        logger.log("graph_prepare", 0, {"intent_hint": hint})
        return {"intent_hint": hint}

    return prepare


def _agent_node(logger: JsonlLogger, model, tools: list[BaseTool], temperature: float):
    async def call_agent(state: AgentState) -> dict[str, Any]:
        step = state.get("llm_calls", 0) + 1
        messages = _model_messages(state)
        logger.log("model_request", step, _model_request_log(state, messages, tools, temperature))

        try:
            response = await model.ainvoke(messages)
        except Exception as exc:
            logger.log("error", step, {"where": "model_request", "error": str(exc)})
            raise

        logger.log("model_response", step, _model_response_log(response))
        return {
            "messages": [response],
            "llm_calls": step,
            "model": _response_model(response),
            "usage": _response_usage(response),
        }

    return call_agent


def _tools_node(logger: JsonlLogger):
    def call_tools(state: AgentState) -> dict[str, Any]:
        step = state.get("llm_calls", 0)
        tool_messages: list[ToolMessage] = []
        tool_steps = list(state.get("tool_steps", []))

        for tool_call in _last_ai_message(state).tool_calls:
            result = _run_tool_call(tool_call, logger, step)
            tool_steps.append(_tool_step(tool_call, result))
            tool_messages.append(_tool_message(tool_call, result))

        return {"messages": tool_messages, "tool_steps": tool_steps}

    return call_tools


def _max_steps_node(logger: JsonlLogger):
    def max_steps(state: AgentState) -> dict[str, Any]:
        logger.log("graph_max_steps", state.get("llm_calls", 0), {"reason": "max_steps"})
        return {"messages": [AIMessage(content=MAX_STEPS_ANSWER)]}

    return max_steps


def _final_node(logger: JsonlLogger):
    def final(state: AgentState) -> dict[str, Any]:
        logger.log("final_answer", state.get("llm_calls", 0), _final_log(state, logger))
        return {}

    return final


def _route_after_agent(state: AgentState) -> Literal["tools", "max_steps", "final"]:
    if not _last_ai_message(state).tool_calls:
        return "final"
    if state.get("llm_calls", 0) >= settings.agent_max_steps:
        return "max_steps"
    return "tools"


def _run_tool_call(tool_call: dict[str, Any], logger: JsonlLogger, step: int) -> str:
    name = tool_call["name"]
    arguments = tool_call.get("args") or {}
    logger.log("tool_call_detected", step, _tool_call_log(tool_call, arguments))

    result = run_tool(name, arguments)
    logger.log("tool_result", step, {"tool_call_id": tool_call.get("id"), "name": name, "result": result})
    return result


def _model_messages(state: AgentState) -> list[BaseMessage]:
    return [
        SystemMessage(content=_system_prompt(state.get("intent_hint", "general"))),
        *state["messages"],
    ]


def _system_prompt(intent_hint: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nRuntime graph hint: current request looks like {intent_hint}."


def _guess_intent(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("calculate", "计算", "+", "-", "*", "/", "数学")):
        return "math"
    if any(token in lowered for token in ("time", "时间", "几点", "日期")):
        return "time"
    if any(token in lowered for token in ("note", "notes", "笔记", "学习计划", "搜索")):
        return "notes"
    return "general"


def _build_response(state: AgentState, logger: JsonlLogger) -> dict[str, Any]:
    return {
        "answer": _answer_text(state),
        "model": state.get("model", settings.deepseek_model),
        "tool_steps": state.get("tool_steps", []),
        "usage": state.get("usage"),
        "log_path": logger.display_path,
    }


def _answer_text(state: AgentState) -> str:
    return _message_text(state["messages"][-1]).strip()


def _latest_human_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return _message_text(message)
    return ""


def _last_ai_message(state: AgentState) -> AIMessage:
    message = state["messages"][-1]
    if not isinstance(message, AIMessage):
        raise RuntimeError("Expected the last graph message to be an AIMessage.")
    return message


def _tool_step(tool_call: dict[str, Any], result: str) -> dict[str, Any]:
    return {
        "name": tool_call["name"],
        "arguments": tool_call.get("args") or {},
        "result": result,
    }


def _tool_message(tool_call: dict[str, Any], result: str) -> ToolMessage:
    return ToolMessage(
        content=result,
        name=tool_call["name"],
        tool_call_id=tool_call["id"],
    )


def _input_log(user_message: str, use_tools: bool, temperature: float) -> dict[str, Any]:
    return {"content": user_message, "use_tools": use_tools, "temperature": temperature, "runtime": "langgraph"}


def _model_request_log(
    state: AgentState,
    messages: list[BaseMessage],
    tools: list[BaseTool],
    temperature: float,
) -> dict[str, Any]:
    return {
        "model": settings.deepseek_model,
        "message_count": len(messages),
        "messages": [_message_for_log(message) for message in messages],
        "tool_names": [tool.name for tool in tools],
        "intent_hint": state.get("intent_hint"),
        "temperature": temperature,
    }


def _model_response_log(message: AIMessage) -> dict[str, Any]:
    return {
        "model": _response_model(message),
        "content": _message_text(message),
        "tool_calls": message.tool_calls,
        "usage": _response_usage(message),
    }


def _tool_call_log(tool_call: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool_call_id": tool_call.get("id"),
        "name": tool_call["name"],
        "arguments": arguments,
    }


def _final_log(state: AgentState, logger: JsonlLogger) -> dict[str, Any]:
    return {
        "answer": _answer_text(state),
        "tool_step_count": len(state.get("tool_steps", [])),
        "intent_hint": state.get("intent_hint"),
        "log_path": logger.display_path,
    }


def _message_for_log(message: BaseMessage) -> dict[str, Any]:
    return {
        "type": message.type,
        "content": _truncate(_message_text(message)),
        "tool_calls": getattr(message, "tool_calls", None),
    }


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return str(content)


def _response_model(message: AIMessage) -> str:
    metadata = message.response_metadata or {}
    return metadata.get("model_name") or metadata.get("model") or settings.deepseek_model


def _response_usage(message: AIMessage) -> dict[str, Any] | None:
    if message.usage_metadata:
        return dict(message.usage_metadata)
    metadata = message.response_metadata or {}
    token_usage = metadata.get("token_usage")
    return dict(token_usage) if isinstance(token_usage, dict) else None


def _truncate(text: str, limit: int = 3000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"
