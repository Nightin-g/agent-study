# LLM Agent Demo

这是一个用于学习 LLM Agent 应用开发的小项目。它用 Python + FastAPI 调用 DeepSeek 的 OpenAI 兼容接口，并用 LangGraph 演示一个最小 Agent 状态图：模型先判断是否需要调用本地工具，工具返回结果后，图再回到模型节点生成最终回答。

项目刻意保持简单：不引入数据库、不引入复杂前端框架，只把 Agent 编排交给 LangGraph。这样你可以看清楚 LLM API、messages、tools、tool_calls、tool registry、StateGraph、JSONL 日志这些核心概念。

## 技术选型

| 技术 | 作用 | 为什么用它 |
| --- | --- | --- |
| Python 3.11+ | 主语言 | AI/Agent/RAG 生态最成熟 |
| FastAPI | 本地 Web API | 写法清晰，自动支持接口文档 |
| Uvicorn | 启动 FastAPI 服务 | FastAPI 常用运行器 |
| LangGraph | 编排 Agent 状态图 | 用 node 和 edge 表达 Agent 流程 |
| LangChain OpenAI | 调用 DeepSeek | DeepSeek 支持 OpenAI 兼容格式 |
| python-dotenv | 读取 `.env` | 把 API Key 放在配置里，不写死进代码 |
| Pydantic | 校验请求和响应 | 让接口数据结构更清楚 |
| 原生 HTML/CSS/JS | 简单网页 | 避免一开始被 React/Vue 分散注意力 |
| JSONL | 记录 Agent 事件 | 方便逐行观察模型请求、工具调用和最终答案 |

## 项目结构

```text
llm-agent-demo/
├─ README.md
├─ requirements.txt
├─ run_dev.py              # 本地启动入口，方便新手直接运行
├─ .env.example            # 配置模板
├─ .env                    # 你的本地配置，填 API Key，不提交
├─ .gitignore
├─ app/
│  ├─ __init__.py
│  ├─ main.py              # FastAPI 入口，接收浏览器请求
│  ├─ config.py            # 读取环境变量
│  ├─ agent_graph.py       # LangGraph 状态图，编排模型节点和工具节点
│  ├─ llm_client.py        # 保留 run_agent 入口，转调用 agent_graph
│  ├─ schemas.py           # 请求和响应的数据结构
│  ├─ logger.py            # JSONL 事件日志
│  └─ tools.py             # Agent 可以调用的本地工具
├─ notes/
│  └─ sample_notes.txt     # search_notes 工具读取的本地笔记样例
├─ static/
│  ├─ index.html           # 浏览器页面
│  ├─ style.css            # 页面样式
│  └─ app.js               # 前端请求后端接口
└─ docs/
   └─ learning-guide.md    # 更详细的学习说明
```

## 先理解一次请求发生了什么

```text
浏览器输入问题
  ↓
static/app.js 调用 POST /api/chat
  ↓
app/main.py 接收请求并校验参数
  ↓
app/llm_client.py 调用 LangGraph Agent
  ↓
app/agent_graph.py 进入 prepare 节点，写入 intent_hint
  ↓
agent 节点调用 DeepSeek，模型决定是否需要调用工具
  ↓
tools 节点执行 app/tools.py 里的本地工具
  ↓
图回到 agent 节点，把工具结果再次发给模型
  ↓
模型生成最终回答
  ↓
浏览器显示 answer 和 tool trace
  ↓
logs/agent-events.jsonl 记录完整事件
```

这就是一个最小 LangGraph Agent 的核心：Graph 负责流程，模型负责决策，工具负责执行。

## 快速开始

进入项目目录：

```powershell
cd D:\Code\agents\llm-agent-demo
```

如果你已经看到 `.venv`，说明我已经帮你创建过虚拟环境。以后你通常只需要安装依赖和启动服务：

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python run_dev.py
```

浏览器打开：

```text
http://127.0.0.1:8000
```

## 配置 DeepSeek

打开 `.env`，把 `DEEPSEEK_API_KEY` 改成你自己的 Key：

```env
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_API_KEY=sk-your-deepseek-key-here
DEEPSEEK_MODEL=deepseek-v4-flash
AGENT_LOG_PATH=logs/agent-events.jsonl
```

说明：

- `DEEPSEEK_BASE_URL` 是 DeepSeek 的 OpenAI 兼容接口地址。
- `DEEPSEEK_API_KEY` 是你的私密凭证，不要发给别人。
- `DEEPSEEK_MODEL` 是模型名，可以换成 DeepSeek 文档里支持的模型。
- `AGENT_LOG_PATH` 是 Agent 事件日志路径，默认写到 `logs/agent-events.jsonl`。
- `.env` 已经被 `.gitignore` 忽略，避免误提交 Key。

DeepSeek 官方 API 文档：https://api-docs.deepseek.com/

## 可以试的提示词

```text
请计算 23 * 19 + 7，并告诉我现在的本地时间。
```

这个问题会触发两个本地工具：

- `calculate`：做精确数学计算。
- `get_current_time`：读取本地时间。

再试一个会读取本地笔记的问题：

```text
从本地学习笔记里搜索 tool_call_id，并解释它有什么用。
```

这个问题会触发：

- `search_notes`：搜索 `notes/sample_notes.txt`。

再试一个不一定需要工具的问题：

```text
用初学者能理解的方式解释：Agent 和普通聊天机器人有什么区别？
```

## 建议阅读顺序

1. 先看 `app/main.py`：理解 HTTP 请求怎么进入后端。
2. 再看 `app/schemas.py`：理解请求体和响应体长什么样。
3. 再看 `app/config.py`：理解 `.env` 如何变成 Python 配置。
4. 重点看 `app/agent_graph.py`：理解 LangGraph 的 State、Node、Edge 和条件路由。
5. 再看 `app/llm_client.py`：理解为什么保留兼容入口。
6. 再看 `app/tools.py`：理解如何把本地函数暴露给模型。
7. 再看 `app/logger.py`：理解 JSONL 日志如何记录 Agent 每一步。
8. 最后看 `static/app.js`：理解网页如何调用后端。

## Agent 事件日志

每次调用 `/api/chat` 都会向 `logs/agent-events.jsonl` 追加日志。常见事件包括：

- `user_input`：用户输入和运行参数。
- `model_request`：发给模型的消息摘要和工具列表。
- `model_response`：模型返回的文本或工具调用。
- `graph_prepare`：LangGraph 的准备节点给请求打的简单意图标签。
- `tool_call_detected`：模型请求调用哪个工具、传了什么参数。
- `tool_result`：本地工具执行结果。
- `graph_max_steps`：状态图达到最大模型调用步数。
- `final_answer`：最终返回给浏览器的答案。

`logs/` 已经在 `.gitignore` 中，不会被提交到 Git。

更多解释见：[docs/learning-guide.md](docs/learning-guide.md)。
