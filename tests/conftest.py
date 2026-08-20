from __future__ import annotations

from pathlib import Path

import pytest

from minimax_studio.config import AppConfig, save_config
from minimax_studio.worker.runtime import runtime


@pytest.fixture
def studio_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config_path = tmp_path / "config.json"
    output = tmp_path / "out"
    output.mkdir()
    monkeypatch.setenv("MINIMAX_STUDIO_CONFIG", str(config_path))
    save_config(
        AppConfig(
            output_dir=str(output),
            models_dir=str(output / "models"),
            comfy_models_dir=str(output / "models"),
        ),
        config_path,
    )
    runtime.reload_config()
    runtime.downloads.clear()
    runtime.download_stops.clear()
    runtime.jobs.clear()
    runtime.music_pipe = None
    runtime.music_pipe_path = None
    runtime.comfy_proc = None
    runtime.comfy_log = None
    runtime.config.ensure_dirs()
    return output
