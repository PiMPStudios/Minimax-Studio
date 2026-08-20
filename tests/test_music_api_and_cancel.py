from pathlib import Path

from minimax_studio.worker.history import delete_entry, list_history, record_entry
from minimax_studio.worker.jobs import JobRequest, cancel_job, start_job


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


def test_music_api_requires_key(studio_home: Path) -> None:
    from minimax_studio.worker.backends.music_api import generate_music_api
    from minimax_studio.worker.runtime import runtime

    runtime.config.minimax_api_key = None
    try:
        generate_music_api("x", JobRequest(kind="music", prompt="folk"))
        raise AssertionError("expected missing key")
    except RuntimeError as exc:
        assert "API key" in str(exc)
