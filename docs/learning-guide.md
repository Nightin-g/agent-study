# 学习说明

这份文档帮你按“从简单到核心”的顺序理解项目。读代码时不用一次看完，可以边运行、边改、边观察页面变化。

## 1. 这个项目到底在学什么

普通 LLM API 调用的重点是：把用户问题发给模型，拿到回答。

Agent 应用多了一层重点：模型可以决定调用外部工具。引入 LangGraph 后，还会多一层“流程图”：

- State：保存消息、工具轨迹、模型调用次数等运行状态。
- Node：执行一步动作，例如准备上下文、调用模型、执行工具。
- Edge：决定下一个节点，例如有工具调用就去 tools，没有工具调用就去 final。

工具可以是：

- 获取当前时间。
- 查询数据库。
- 调用搜索 API。
- 读取文件。
- 执行业务系统里的某个操作。

这个 demo 里先放了三个最小工具：

- `get_current_time`：让模型获得当前时间。
- `calculate`：让模型做精确数学计算。
- `search_notes`：让模型搜索本地学习笔记。

## 2. messages 是什么

`messages` 是 Chat Completions API 最核心的数据结构。它像一段对话历史：

```python
messages = [
    {"role": "system", "content": "你是一个有帮助的助手"},
    {"role": "user", "content": "请计算 23 * 19 + 7"},
]
```

常见 role：

- `system`：给模型的最高优先级指令。
- `user`：用户输入。
- `assistant`：模型的回复。
- `tool`：工具执行结果。

Agent 循环的关键就是不断维护这份 `messages`。

## 3. tool definition 是什么

模型不能直接看到你的 Python 函数。你需要给模型一份工具说明书，告诉它：

- 工具叫什么。
- 工具用来做什么。
- 工具需要哪些参数。

这就是 `app/tools.py` 里的 `TOOL_DEFINITIONS`。

注意：`TOOL_DEFINITIONS` 只是说明书，不是真正执行工具的代码。真正执行工具的是 `run_tool()`。

这个项目还引入了 `TOOL_REGISTRY`。它像一个工具目录，把工具名映射到真正的 Python 函数。这样新增工具时，不需要在 `run_tool()` 里堆很多 `if/elif`。

## 4. LangGraph 状态图怎么看

核心代码在 `app/agent_graph.py`。当前图结构是：

```text
START
  ↓
prepare
  ↓
agent
  ├─ 有 tool_calls 且未超步数 → tools → agent
  ├─ 有 tool_calls 但达到步数 → max_steps → final
  └─ 没有 tool_calls → final
  ↓
END
```

`prepare` 是一个适合学习的小扩展：它不用模型，只用简单规则给问题打一个 `intent_hint`，例如 `math`、`time`、`notes`、`general`。这说明 LangGraph 里不只有 LLM 节点，也可以放普通 Python 逻辑。

## 5. 一次工具调用的完整流程

以“请计算 23 * 19 + 7”为例：

1. 浏览器把问题发给 `/api/chat`。
2. `run_agent()` 调用 LangGraph 编译好的状态图。
3. `prepare` 节点给问题打上 `math` 提示。
4. `agent` 节点把用户问题、系统提示和工具说明发给 DeepSeek。
5. 模型判断需要精确计算，于是返回一个 tool call。
6. `tools` 节点调用本地 `calculate()`。
7. `tools` 节点把计算结果作为 `ToolMessage` 追加到 State。
8. 图回到 `agent` 节点。
9. 模型基于工具结果生成自然语言回答。

这就是最小 Agent。

## 6. JSONL 日志怎么看

项目会把 Agent 运行过程写到：

```text
logs/agent-events.jsonl
```

JSONL 是“一行一个 JSON 对象”。你可以把它当成 Agent 的运行记录。常见事件包括：

- `user_input`：用户输入。
- `graph_prepare`：LangGraph 准备节点给出的 `intent_hint`。
- `model_request`：后端发给模型的消息和工具列表。
- `model_response`：模型返回的内容，可能是最终文本，也可能是工具调用。
- `tool_call_detected`：模型决定调用哪个工具。
- `tool_result`：本地工具返回了什么。
- `graph_max_steps`：图达到最大模型调用次数。
- `final_answer`：最终答案。

学习时可以先在浏览器看 `Agent Trace`，再打开日志文件看更完整的上下文。

## 7. 为什么不用 eval 做计算

Python 的 `eval()` 可以执行任意代码，不适合处理用户或模型传来的字符串。

这个项目用 `ast` 把数学表达式解析成语法树，然后只允许基础数学节点。这样既能完成计算，又能避免执行危险代码。

## 8. 后续可以怎么扩展

你可以尝试新增工具，例如：

- `search_web(query: str)`：搜索网页。
- `read_note(filename: str)`：读取本地笔记。
- `write_todo(item: str)`：写入待办事项。
- `get_weather(city: str)`：查询天气。

新增工具通常需要改三处：

1. 在 `app/tools.py` 写一个 Python 函数。
2. 在 `TOOL_REGISTRY` 里把工具名映射到对应函数。
3. 在 `app/agent_graph.py` 的 `_build_tools()` 里增加一个 `StructuredTool`。

如果你想练 LangGraph，可以继续新增节点，例如：

- `guardrail`：检查输入是否过长或为空。
- `rewrite_query`：把用户问题改写成更适合工具检索的关键词。
- `summarize_trace`：把工具调用过程总结成一句调试说明。

## 9. 建议你动手改的小练习

1. 把 `temperature` 调成 0，观察回答是否更稳定。
2. 关闭“启用工具”，再问数学题，观察结果有什么不同。
3. 在 `SYSTEM_PROMPT` 里加一句“回答要非常简短”，观察模型风格变化。
4. 新增一个 `echo` 工具，让模型可以原样返回某段文本。
5. 把 `AGENT_MAX_STEPS` 改成 1，再问需要工具的问题，观察保护机制。
6. 修改 `_guess_intent()`，给更多问题打上不同 `intent_hint`。
7. 让 `search_notes` 支持多个关键词，并观察日志里参数如何变化。
