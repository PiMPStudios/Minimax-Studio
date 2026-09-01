from pathlib import Path

import pytest

from minimax_studio.worker.history import delete_entry, list_history, record_entry
from minimax_studio.worker.jobs import JobRequest, cancel_job, start_job


def test_stub_names_that_it_skips_loras(studio_home: Path, monkeypatch) -> None:
    from minimax_studio.worker.backends.music import generate_music

    monkeypatch.setenv("MINIMAX_STUDIO_STUB", "1")
    messages: list[str] = []

    def capture(_job_id: str, **fields) -> None:
        if "message" in fields:
            messages.append(str(fields["message"]))

    monkeypatch.setattr(
        "minimax_studio.worker.backends.music.update_job", capture
    )
    generate_music(
        "stub-lora",
        JobRequest(
            kind="music",
            backend="stub",
            prompt="folk",
            loras=[{"id": "a.safetensors", "strength": 0.8}],
        ),
    )
    assert any("skips LoRAs" in item for item in messages)


def test_cancel_stub_job(studio_home: Path, monkeypatch) -> None:
    monkeypatch.setenv("MINIMAX_STUDIO_STUB", "1")
    job = start_job(JobRequest(kind="music", backend="stub", prompt="x"))
    rec = cancel_job(job["id"])
    assert rec["status"] in {"cancelling", "cancelled", "done"}


def test_history_delete(studio_home: Path) -> None:
    record_entry(
        {
            "id": "abc123abc123",
            "kind": "music",
            "prompt": "gone",
            "output_path": str(studio_home / "history" / "abc123abc123" / "audio.wav"),
        }
    )
    assert any(item["id"] == "abc123abc123" for item in list_history())
    delete_entry("abc123abc123")
    assert all(item["id"] != "abc123abc123" for item in list_history())
    assert not (studio_home / "history" / "abc123abc123").exists()


def test_mlx_blank_lyrics_are_not_an_instrumental(
    studio_home: Path, monkeypatch
) -> None:
    import sys
    import types

    import numpy as np

    captured: dict = {}

    class FakeResult:
        audio = np.zeros(16)
        sample_rate = 32000

    class FakeModel:
        def generate(self, **kwargs):
            captured.update(kwargs)
            yield FakeResult()

    mlx_audio = types.ModuleType("mlx_audio")
    mlx_music = types.ModuleType("mlx_audio.music")
    mlx_music.load = lambda path: FakeModel()
    monkeypatch.setitem(sys.modules, "mlx_audio", mlx_audio)
    monkeypatch.setitem(sys.modules, "mlx_audio.music", mlx_music)

    from minimax_studio.worker.backends import music
    from minimax_studio.worker.catalog import PACKS
    from minimax_studio.worker.runtime import runtime

    pack = runtime.config.models_root() / PACKS["music3-mlx"].local_dir
    pack.mkdir(parents=True, exist_ok=True)
    dest = runtime.config.history_root() / "mlxjob"
    dest.mkdir(parents=True, exist_ok=True)
    music._generate_mlx(
        "mlxjob",
        JobRequest(kind="music", prompt="folk", lyrics=""),
        dest / "audio.wav",
    )
    assert captured.get("lyrics") == ""


def test_mlx_random_seed_is_an_int_mlx_audio_will_accept(
    studio_home: Path, monkeypatch
) -> None:
    """mlx-audio.music.generate types seed as int (default 0). Studio's default
    seed is -1 (random). Passing None TypeErrors on Apple Silicon generate."""
    import sys
    import types

    import numpy as np

    captured: dict = {}

    class FakeResult:
        audio = np.zeros(16)
        sample_rate = 32000

    class FakeModel:
        def generate(self, **kwargs):
            captured.update(kwargs)
            yield FakeResult()

    mlx_audio = types.ModuleType("mlx_audio")
    mlx_music = types.ModuleType("mlx_audio.music")
    mlx_music.load = lambda path: FakeModel()
    monkeypatch.setitem(sys.modules, "mlx_audio", mlx_audio)
    monkeypatch.setitem(sys.modules, "mlx_audio.music", mlx_music)

    from minimax_studio.worker.backends import music
    from minimax_studio.worker.catalog import PACKS
    from minimax_studio.worker.runtime import runtime

    pack = runtime.config.models_root() / PACKS["music3-mlx"].local_dir
    pack.mkdir(parents=True, exist_ok=True)
    dest = runtime.config.history_root() / "mlxseed"
    dest.mkdir(parents=True, exist_ok=True)
    music._generate_mlx(
        "mlxseed",
        JobRequest(kind="music", prompt="folk", seed=-1),
        dest / "audio.wav",
    )
    seed = captured.get("seed")
    assert isinstance(seed, int)
    assert seed >= 0
    music._generate_mlx(
        "mlxseed",
        JobRequest(kind="music", prompt="folk", seed=7),
        dest / "audio.wav",
    )
    assert captured.get("seed") == 7


def test_music_api_and_mlx_refuse_loras(studio_home: Path, monkeypatch) -> None:
    from minimax_studio.worker.backends.music import generate_music
    from minimax_studio.worker.jobs import JobRequest

    monkeypatch.setattr(
        "minimax_studio.worker.backends.music.resolve_music_backend",
        lambda requested: "api",
    )
    with pytest.raises(RuntimeError, match="LoRAs only load"):
        generate_music(
            "x",
            JobRequest(
                kind="music",
                prompt="folk",
                loras=[{"id": "/models/loras/a.safetensors", "strength": 0.8}],
            ),
        )
    monkeypatch.setattr(
        "minimax_studio.worker.backends.music.resolve_music_backend",
        lambda requested: "mlx",
    )
    with pytest.raises(RuntimeError, match="this job resolved to mlx"):
        generate_music(
            "y",
            JobRequest(
                kind="music",
                prompt="folk",
                loras=[{"id": "/models/loras/a.safetensors", "strength": 0.8}],
            ),
        )


def test_music_api_requires_key(studio_home: Path) -> None:
    from minimax_studio.worker.backends.music_api import generate_music_api
    from minimax_studio.worker.runtime import runtime

    runtime.config.minimax_api_key = None
    try:
        generate_music_api("x", JobRequest(kind="music", prompt="folk"))
        raise AssertionError("expected missing key")
    except RuntimeError as exc:
        assert "API key" in str(exc)


class _MusicApiResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "base_resp": {"status_code": 0},
            "data": {"audio": b"RIFF".hex()},
        }


def _patch_music_api_client(monkeypatch, calls: dict) -> None:
    class FakeClient:
        def __init__(self, timeout=None):
            pass

        def post(self, url, headers=None, json=None):
            calls["url"] = url
            calls["json"] = json
            return _MusicApiResponse()

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "minimax_studio.worker.backends.music_api.httpx.Client", FakeClient
    )
    monkeypatch.setattr(
        "minimax_studio.worker.backends.music_api.is_cancelled",
        lambda _job: False,
    )


def test_music_api_blank_lyrics_uses_the_optimizer(
    studio_home: Path, monkeypatch
) -> None:
    from minimax_studio.worker.backends.music_api import generate_music_api
    from minimax_studio.worker.runtime import runtime

    runtime.config.minimax_api_key = "k"
    calls: dict = {}
    _patch_music_api_client(monkeypatch, calls)
    generate_music_api("blank-lyrics", JobRequest(kind="music", prompt="folk", lyrics=""))
    payload = calls["json"]
    assert payload["lyrics_optimizer"] is True
    assert "lyrics" not in payload
    assert payload["is_instrumental"] is False


def test_music_api_supplied_lyrics_skip_the_optimizer(
    studio_home: Path, monkeypatch
) -> None:
    from minimax_studio.worker.backends.music_api import generate_music_api
    from minimax_studio.worker.runtime import runtime

    runtime.config.minimax_api_key = "k"
    calls: dict = {}
    _patch_music_api_client(monkeypatch, calls)
    generate_music_api(
        "with-lyrics",
        JobRequest(kind="music", prompt="folk", lyrics="[Verse]\nhello"),
    )
    payload = calls["json"]
    assert "lyrics_optimizer" not in payload
    assert payload["lyrics"] == "[Verse]\nhello"
    assert payload["is_instrumental"] is False


def test_music_api_cancel_skips_the_post(studio_home: Path, monkeypatch) -> None:
    from minimax_studio.worker.backends.music_api import generate_music_api
    from minimax_studio.worker.jobs import CancelledError
    from minimax_studio.worker.runtime import runtime

    runtime.config.minimax_api_key = "k"
    calls: list = []

    class FakeClient:
        def __init__(self, timeout=None):
            pass

        def post(self, *args, **kwargs):
            calls.append(1)
            raise AssertionError("cancelled jobs must not POST")

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "minimax_studio.worker.backends.music_api.httpx.Client", FakeClient
    )
    monkeypatch.setattr(
        "minimax_studio.worker.backends.music_api.is_cancelled",
        lambda _job: True,
    )
    with pytest.raises(CancelledError):
        generate_music_api("x", JobRequest(kind="music", prompt="folk"))
    assert calls == []
