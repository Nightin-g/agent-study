"""Application configuration.

这个文件专门负责读取配置。

为什么不直接在代码里写 API Key？
1. API Key 是敏感信息，不应该写死在源码里。
2. 本地、测试、线上可能使用不同的 Key 和模型。
3. `.env` 改起来更方便，不需要改 Python 代码。
"""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# 读取项目根目录下的 .env 文件。
# load_dotenv 会把 .env 里的键值对放进 os.environ，后面就能用 os.getenv 读取。
load_dotenv(PROJECT_ROOT / ".env")


def _int_env(name: str, default: int) -> int:
    """Read an integer environment variable with a safe fallback.

    环境变量本质上都是字符串，比如 APP_PORT=8000 读出来是 "8000"。
    这里统一把字符串转成 int。如果用户填错了，就回退到默认值。
    """

    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError:
        return default


def _path_env(name: str, default: Path) -> Path:
    """Read a path environment variable relative to the project root."""

    raw_value = os.getenv(name)
    if not raw_value:
        return default

    path = Path(raw_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    """Typed settings used by the rest of the application.

    dataclass 的好处是字段集中、可读性强。
    frozen=True 表示创建后不希望被随意修改，配置应该稳定。
    """

    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    # FastAPI 服务监听地址。127.0.0.1 只允许本机访问，适合本地学习。
    app_host: str = os.getenv("APP_HOST", "127.0.0.1")
    app_port: int = _int_env("APP_PORT", 8000)

    # Agent 循环上限。模型可以多次请求工具，但不能无限循环。
    agent_max_steps: int = _int_env("AGENT_MAX_STEPS", 5)

    # Agent 事件日志。默认放在 logs/ 里，方便学习时观察每一步。
    agent_log_path: Path = _path_env("AGENT_LOG_PATH", PROJECT_ROOT / "logs" / "agent-events.jsonl")

    @property
    def has_api_key(self) -> bool:
        """Return True only when the API key looks like a real user value."""

        return bool(self.deepseek_api_key and not self.deepseek_api_key.startswith("sk-your-"))


# 其他模块统一 import settings，避免每个文件都重复读环境变量。
settings = Settings()
