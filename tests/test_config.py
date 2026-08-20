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
    assert loaded.use_os_keyring is False
    raw = path.read_text(encoding="utf-8")
    assert "secret" in raw


class _FakeKeyring:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, key: str) -> str | None:
        return self.store.get((service, key))

    def set_password(self, service: str, key: str, value: str) -> None:
        self.store[(service, key)] = value

    def delete_password(self, service: str, key: str) -> None:
        self.store.pop((service, key), None)


def test_os_keyring_keeps_tokens_out_of_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "cfg.json"
    monkeypatch.setenv("MINIMAX_STUDIO_CONFIG", str(path))
    ring = _FakeKeyring()
    monkeypatch.setattr("minimax_studio.secrets._keyring", lambda: ring)
    monkeypatch.setattr("minimax_studio.secrets.keyring_available", lambda: True)
    config = AppConfig(
        output_dir=str(tmp_path),
        hf_token="hf-secret",
        minimax_api_key="mm-secret",
        use_os_keyring=True,
    )
    save_config(config, path)
    raw = path.read_text(encoding="utf-8")
    assert "hf-secret" not in raw
    assert "mm-secret" not in raw
    loaded = load_config(path)
    assert loaded.hf_token == "hf-secret"
    assert loaded.minimax_api_key == "mm-secret"
    assert loaded.use_os_keyring is True
