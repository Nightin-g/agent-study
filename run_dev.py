r"""Local development launcher.

运行方式：
    .\.venv\Scripts\python run_dev.py

为什么单独放一个 run_dev.py？
对初学者来说，比记住一长串 uvicorn 命令更直观。
"""

import uvicorn

from app.config import settings


if __name__ == "__main__":
    # 这里启动 FastAPI 应用。
    # "app.main:app" 的意思是：找到 app/main.py 里的 app 对象。
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
    )
