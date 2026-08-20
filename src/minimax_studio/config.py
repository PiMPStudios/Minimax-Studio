from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel


def default_config_path() -> Path:
    override = os.environ.get("MINIMAX_STUDIO_CONFIG")
    if override:
        return Path(override)
    home = Path.home()
    system = os.name
    if system == "nt":
        root = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
        return root / "MiniMaxStudio" / "config.json"
    if Path("/Applications").exists() and (home / "Library").exists():
        return home / "Library" / "Application Support" / "MiniMaxStudio" / "config.json"
    xdg = Path(os.environ.get("XDG_CONFIG_HOME", home / ".config"))
    return xdg / "minimax-studio" / "config.json"


class AppConfig(BaseModel):
    output_dir: str | None = None
    models_dir: str | None = None
    hf_token: str | None = None
    minimax_api_key: str | None = None
    minimax_api_base: str = "https://api.minimax.io"
    llm_base_url: str = "http://127.0.0.1:8080/v1"
    llm_model: str = "qwen3.8-27b-q4kxl"
    llm_api_key: str | None = None

    def resolved_llm_key(self) -> str | None:
        if self.llm_api_key:
            return self.llm_api_key
        env = os.environ.get("MINIMAX_STUDIO_LLM_KEY")
        if env:
            return env
        path = Path.home() / ".config" / "llama-api.key"
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            return text or None
        return None

    def has_output_dir(self) -> bool:
        return bool(self.output_dir) and Path(self.output_dir).is_dir()

    def models_root(self) -> Path:
        if self.models_dir:
            return Path(self.models_dir)
        if self.output_dir:
            return Path(self.output_dir) / "models"
        raise RuntimeError("Set an output directory first.")

    def history_root(self) -> Path:
        if not self.output_dir:
            raise RuntimeError("Set an output directory first.")
        return Path(self.output_dir) / "history"

    def ensure_dirs(self) -> None:
        if self.output_dir:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.models_root().mkdir(parents=True, exist_ok=True)
        self.history_root().mkdir(parents=True, exist_ok=True)


def load_config(path: Path | None = None) -> AppConfig:
    path = path or default_config_path()
    if not path.is_file():
        return AppConfig()
    data = json.loads(path.read_text(encoding="utf-8"))
    return AppConfig.model_validate(data)


def save_config(config: AppConfig, path: Path | None = None) -> None:
    path = path or default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
