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
        lambda _backend, _mode="fl2va": "cuda",
    )
    result = preflight("h3", "cuda")
    assert result["ok"] is True
    assert result["ffmpeg"] is False
    assert result["warnings"]
    assert "ffmpeg" in result["detail"].lower()


def test_preflight_fast_blocks_without_turbo(studio_home: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "minimax_studio.worker.preflight.probe",
        lambda: {
            "torch_available": False,
            "cuda": True,
            "gpus": [{"name": "RTX", "vram_gb": 32}],
            "ffmpeg": True,
        },
    )
    monkeypatch.setattr(
        "minimax_studio.worker.backends.h3.resolve_h3_backend",
        lambda _backend, _mode="fl2va": "comfy",
    )
    monkeypatch.setattr(
        "minimax_studio.worker.backends.h3._find_turbo_lora",
        lambda _mode="t2va": None,
    )
    result = preflight("h3", "comfy", "t2va", "fast")
    assert result["ok"] is False
    assert "Turbo" in result["detail"]


def test_preflight_fast_ok_with_turbo(studio_home: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "minimax_studio.worker.preflight.probe",
        lambda: {
            "torch_available": False,
            "cuda": True,
            "gpus": [{"name": "RTX", "vram_gb": 32}],
            "ffmpeg": True,
        },
    )
    monkeypatch.setattr(
        "minimax_studio.worker.backends.h3.resolve_h3_backend",
        lambda _backend, _mode="fl2va": "comfy",
    )
    monkeypatch.setattr(
        "minimax_studio.worker.backends.h3._find_turbo_lora",
        lambda _mode="t2va": "/models/loras/turbo.safetensors",
    )
    result = preflight("h3", "comfy", "t2va", "fast")
    assert result["ok"] is True
    assert "Turbo" in result["detail"]
