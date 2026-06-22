"""Small JSONL logger for observing Agent runs.

JSONL 是“一行一个 JSON 对象”的日志格式。
它适合学习 Agent：你可以按时间顺序看到用户输入、模型响应、工具调用和最终答案。
"""

from __future__ import annotations

from datetime import datetime
import json
import os
import re
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, settings


SECRET_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")


class JsonlLogger:
    """Write Agent events to a local JSONL file."""

    def __init__(self, log_path: Path | None = None) -> None:
        self.log_path = log_path or settings.agent_log_path

    @property
    def display_path(self) -> str:
        """Return a project-relative path for UI and API responses."""

        try:
            return str(self.log_path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(self.log_path)

    def log(self, event_type: str, step: int, data: Any) -> None:
        """Append one JSON event to the log file."""

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "event_type": event_type,
            "step": step,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "data": self._redact(data),
        }
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _redact(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        if isinstance(value, str):
            return self._redact_string(value)
        return value

    def _redact_string(self, value: str) -> str:
        redacted = value
        for secret in self._known_secret_values():
            redacted = redacted.replace(secret, "[REDACTED]")
        return SECRET_PATTERN.sub("[REDACTED_API_KEY]", redacted)

    def _known_secret_values(self) -> list[str]:
        values = []
        for name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
            secret = os.getenv(name)
            if secret:
                values.append(secret)
        if settings.deepseek_api_key:
            values.append(settings.deepseek_api_key)
        return values
