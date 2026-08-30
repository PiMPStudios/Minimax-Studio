from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from minimax_studio.worker.backends.h3 import guard_resolution
from minimax_studio.worker.jobs import JobRequest

HW_OK = {
    "torch_available": True,
    "cuda": True,
    "gpus": [{"name": "RTX", "vram_gb": 24}],
    "ffmpeg": True,
}


def _fake_h3_resolve(monkeypatch, resolved: str) -> None:
    from minimax_studio.worker.preflight import probe  # noqa: F401

    monkeypatch.setattr("minimax_studio.worker.preflight.probe", lambda: dict(HW_OK))
    monkeypatch.setattr(
        "minimax_studio.worker.backends.h3.resolve_h3_backend",
        lambda _backend, _mode="fl2va": resolved,
    )


def test_preflight_2k_blocked_on_local_backend(studio_home: Path, monkeypatch) -> None:
    from minimax_studio.worker.preflight import preflight

    _fake_h3_resolve(monkeypatch, "cuda")
    result = preflight("h3", "local", resolution="2K")
    assert result["ok"] is False
    assert "2K" in result["detail"]
    assert "API" in result["detail"]
    # the same job at 768P is fine
    assert preflight("h3", "local", resolution="768P")["ok"] is True


def test_preflight_2k_allowed_on_api(studio_home: Path, monkeypatch) -> None:
    from minimax_studio.worker.preflight import preflight

    _fake_h3_resolve(monkeypatch, "api")
    result = preflight("h3", "api", resolution="2K")
    assert result["ok"] is True
    assert result["backend"] == "api"


def test_preflight_music_api_names_dropped_params(
    studio_home: Path, monkeypatch
) -> None:
    from minimax_studio.worker.preflight import preflight

    monkeypatch.setattr("minimax_studio.worker.preflight.probe", lambda: dict(HW_OK))
    monkeypatch.setattr(
        "minimax_studio.worker.backends.music.resolve_music_backend",
        lambda _backend: "api",
    )
    result = preflight("music", "api")
    assert result["ok"] is True
    for param in ("Duration", "Seed", "Steps", "CFG"):
        assert param in result["detail"]
    assert "lyrics_optimizer" in result["detail"]
    assert "LoRAs do not apply" in result["detail"]


def test_guard_resolution_fails_loudly_off_api() -> None:
    request = JobRequest(kind="h3", prompt="x", resolution="2K")
    for backend in ("cuda", "comfy", "stub"):
        with pytest.raises(RuntimeError, match="2K"):
            guard_resolution(request, backend)
    guard_resolution(request, "api")  # allowed
    guard_resolution(JobRequest(kind="h3", prompt="x", resolution="768P"), "cuda")


def test_music_tags_include_post_chorus_and_solo() -> None:
    from minimax_studio.ui.pages.music_page import TAGS

    assert "[Post-Chorus]" in TAGS
    assert "[Solo]" in TAGS


def test_2k_option_greyed_for_local_backends() -> None:
    from PySide6.QtWidgets import QApplication

    from minimax_studio.ui.pages.video_page import VideoPage
    from minimax_studio.ui.state import StudioState

    QApplication.instance()
    state = StudioState()
    page = VideoPage(SimpleNamespace(), state)
    two_k = next(
        i for i in range(page.resolution.count())
        if page.resolution.itemText(i) == "2K"
    )
    assert page.resolution.model().item(two_k).isEnabled()  # auto: selectable

    state.set_backend("api")
    page.resolution.setCurrentText("2K")
    assert page.resolution.currentText() == "2K"

    state.set_backend("comfy")  # 2K must grey out and fall back
    assert not page.resolution.model().item(two_k).isEnabled()
    assert page.resolution.currentText() == "768P"

    state.set_backend("api")  # and come back when the API is selected again
    assert page.resolution.model().item(two_k).isEnabled()
