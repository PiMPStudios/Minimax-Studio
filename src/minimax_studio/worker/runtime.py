from __future__ import annotations

import threading
from typing import Any

from minimax_studio.config import AppConfig, load_config


class Runtime:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.job_changed = threading.Condition()
        self.config = load_config()
        self.downloads: dict[str, dict[str, Any]] = {}
        self.download_stops: dict[str, threading.Event] = {}
        self.jobs: dict[str, dict[str, Any]] = {}
        self.music_pipe: Any = None
        self.music_pipe_path: str | None = None
        self.h3_pipe: Any = None
        self.h3_pipe_path: str | None = None
        self.comfy_proc: Any = None
        self.comfy_log: Any = None

    def reload_config(self) -> AppConfig:
        with self.lock:
            self.config = load_config()
            return self.config


runtime = Runtime()
