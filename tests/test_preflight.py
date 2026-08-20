from pathlib import Path

from minimax_studio.worker.preflight import preflight


def test_preflight_h3_without_packs(studio_home: Path) -> None:
    result = preflight("h3", "auto")
    assert result["ok"] is False
    assert result["backend"] is None
    assert result["detail"]
    assert "ffmpeg" in result
    assert "warnings" in result


def test_preflight_music_stub_ok(studio_home: Path, monkeypatch) -> None:
    monkeypatch.setenv("MINIMAX_STUDIO_STUB", "1")
    result = preflight("music", "stub")
    assert result["ok"] is True
    assert result["backend"] == "stub"
    assert isinstance(result["ffmpeg"], bool)


def test_preflight_h3_cuda_warns_without_ffmpeg(studio_home: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "minimax_studio.worker.preflight.probe",
        lambda: {
            "torch_available": True,
            "cuda": True,
            "gpus": [{"name": "RTX", "vram_gb": 32}],
            "ffmpeg": False,
        },
    )
    monkeypatch.setattr(
        "minimax_studio.worker.backends.h3.resolve_h3_backend",
        lambda _backend: "cuda",
    )
    result = preflight("h3", "cuda")
    assert result["ok"] is True
    assert result["ffmpeg"] is False
    assert result["warnings"]
    assert "ffmpeg" in result["detail"].lower()
