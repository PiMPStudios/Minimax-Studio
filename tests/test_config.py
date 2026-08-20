from pathlib import Path

import pytest

from minimax_studio.config import AppConfig, load_config, save_config


def test_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "cfg.json"
    monkeypatch.setenv("MINIMAX_STUDIO_CONFIG", str(path))
    config = AppConfig(output_dir=str(tmp_path), hf_token="secret")
    save_config(config, path)
    loaded = load_config(path)
    assert loaded.output_dir == str(tmp_path)
    assert loaded.hf_token == "secret"
    assert loaded.models_root() == tmp_path / "models"
    assert loaded.comfy_url == "http://127.0.0.1:8188"
    assert loaded.comfy_root is None
    assert loaded.comfy_extra_args is None
    assert loaded.welcome_seen is False
