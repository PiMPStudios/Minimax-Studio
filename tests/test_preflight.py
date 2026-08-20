from pathlib import Path

from minimax_studio.worker.preflight import preflight


def test_preflight_h3_without_packs(studio_home: Path) -> None:
    result = preflight("h3", "auto")
    assert result["ok"] is False
    assert result["backend"] is None
    assert result["detail"]


def test_preflight_music_stub_ok(studio_home: Path, monkeypatch) -> None:
    monkeypatch.setenv("MINIMAX_STUDIO_STUB", "1")
    result = preflight("music", "stub")
    assert result["ok"] is True
    assert result["backend"] == "stub"
