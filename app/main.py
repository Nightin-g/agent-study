"""FastAPI application entry point.

这个文件负责把“浏览器请求”接到“Agent 逻辑”上。

你可以把它理解成项目的大门：
- GET / 返回网页
- GET /api/health 返回配置状态
- POST /api/chat 接收用户问题并调用 Agent
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import APIConnectionError, APIError, AuthenticationError, RateLimitError

from .config import settings
from .llm_client import run_agent
from .schemas import ChatRequest, ChatResponse


# BASE_DIR 指向项目根目录 llm-agent-demo。
# __file__ 是当前文件 app/main.py，parent.parent 就是上一级的上一级。
BASE_DIR = Path(__file__).resolve().parent.parent


# 创建 FastAPI 应用。
# title 和 version 会显示在自动接口文档 /docs 里。
app = FastAPI(
    title="LLM Agent Demo",
    version="0.1.0",
)


# 把 static/ 目录挂载成静态资源目录。
# 这样浏览器才能访问 /static/app.js 和 /static/style.css。
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index() -> FileResponse:
    """Return the demo web page."""

    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/health")
def health() -> dict[str, object]:
    """Return basic runtime status for the browser page.

    前端会用这个接口显示当前模型、base_url，以及 API Key 是否已配置。
    注意：这里不会返回 API Key 的真实值，避免泄露。
    """

    return {
        "ok": True,
        "base_url": settings.deepseek_base_url,
        "model": settings.deepseek_model,
        "has_api_key": settings.has_api_key,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> dict[str, object]:
    """Handle a chat request from the browser.

    request 已经被 FastAPI 和 Pydantic 校验过：
    - message 一定是非空字符串
    - temperature 一定在 0 到 2 之间
    """

    if not settings.has_api_key:
        raise HTTPException(status_code=400, detail="请先在 .env 中填写 DEEPSEEK_API_KEY。")

    try:
        return await run_agent(
            request.message,
            use_tools=request.use_tools,
            temperature=request.temperature,
        )
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail="DeepSeek API Key 无效或没有权限。") from exc
    except RateLimitError as exc:
        raise HTTPException(status_code=429, detail="DeepSeek API 请求过于频繁，请稍后再试。") from exc
    except APIConnectionError as exc:
        raise HTTPException(status_code=502, detail="无法连接 DeepSeek API，请检查网络或 base_url。") from exc
    except APIError as exc:
        raise HTTPException(status_code=502, detail=f"DeepSeek API 返回错误：{exc.message}") from exc
