from pathlib import Path

from minimax_studio.worker.catalog import PACKS
from minimax_studio.worker.downloads import pack_status
from minimax_studio.worker.ping import ping_services
from minimax_studio.worker.presets import save_preset


def test_partial_pack_flag(studio_home: Path) -> None:
    dest = studio_home / "models" / PACKS["music3-cuda"].local_dir
    dest.mkdir(parents=True)
    blob = dest / "chunk.bin"
    blob.write_bytes(b"x" * (1024 * 1024 + 10))
    status = pack_status(PACKS["music3-cuda"], studio_home / "models")
    assert status["ready"] is False
    assert status["partial"] is True


def test_preset_keeps_assets_and_lora(studio_home: Path) -> None:
    saved = save_preset(
        {
            "name": "Ref shot",
            "kind": "h3",
            "mode": "fl2va",
            "assets": [{"role": "first_frame", "path": "/tmp/a.png"}],
            "lora_id": "/tmp/x.safetensors",
            "lora_strength": 0.7,
            "speed": "fast",
        }
    )
    assert saved["assets"][0]["path"] == "/tmp/a.png"
    assert saved["lora_id"].endswith("x.safetensors")
    assert saved["speed"] == "fast"


def test_ping_without_minimax_key(studio_home, monkeypatch) -> None:
    class _Resp:
        status_code = 200

    monkeypatch.setattr("minimax_studio.worker.ping.httpx.get", lambda *a, **k: _Resp())
    from minimax_studio.worker.runtime import runtime

    runtime.config.minimax_api_key = None
    result = ping_services()
    assert result["minimax"]["ok"] is False
    assert result["minimax"]["detail"] == "no key"
    assert result["llm"]["ok"] is True
    assert result["comfy"]["ok"] is True
