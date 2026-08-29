import threading
import time
from pathlib import Path

from minimax_studio.worker.jobs import (
    JobRequest,
    cancel_job,
    get_job,
    iter_job_snapshots,
    start_job,
)


def test_stub_music_job_writes_history(studio_home: Path, monkeypatch) -> None:
    monkeypatch.setenv("MINIMAX_STUDIO_STUB", "1")
    job = start_job(JobRequest(kind="music", backend="stub", prompt="warm folk", lyrics="[Verse]\nhi"))
    deadline = time.time() + 5
    while time.time() < deadline:
        current = get_job(job["id"])
        if current["status"] in {"done", "error"}:
            break
        time.sleep(0.05)
    current = get_job(job["id"])
    assert current["status"] == "done", current.get("error")
    assert current["output_path"]
    assert Path(current["output_path"]).is_file()
    assert (studio_home / "history" / job["id"] / "meta.json").is_file()
    import json

    meta = json.loads((studio_home / "history" / job["id"] / "meta.json").read_text())
    assert meta["speed"] == "quality"
    assert "cfg" in meta


def test_job_queue_runs_second_after_first(studio_home: Path, monkeypatch) -> None:
    monkeypatch.setenv("MINIMAX_STUDIO_STUB", "1")
    first = start_job(JobRequest(kind="music", backend="stub", prompt="one"))
    second = start_job(JobRequest(kind="music", backend="stub", prompt="two"))
    deadline = time.time() + 8
    while time.time() < deadline:
        a = get_job(first["id"])
        b = get_job(second["id"])
        if a["status"] in {"done", "error"} and b["status"] in {"done", "error"}:
            break
        time.sleep(0.05)
    assert get_job(first["id"])["status"] == "done", get_job(first["id"]).get("error")
    assert get_job(second["id"])["status"] == "done", get_job(second["id"]).get("error")
    assert Path(get_job(first["id"])["output_path"]).is_file()
    assert Path(get_job(second["id"])["output_path"]).is_file()


def test_cancel_queued_job(studio_home: Path, monkeypatch) -> None:
    monkeypatch.setenv("MINIMAX_STUDIO_STUB", "1")
    from minimax_studio.worker.backends import music as music_mod
    from minimax_studio.worker.jobs import cancel_job

    blocker = threading.Event()
    original = music_mod.generate_music

    def hang_music(job_id, request):
        blocker.wait(timeout=3)
        return original(job_id, request)

    monkeypatch.setattr(music_mod, "generate_music", hang_music)
    first = start_job(JobRequest(kind="music", backend="stub", prompt="hold"))
    second = start_job(JobRequest(kind="music", backend="stub", prompt="later"))
    deadline = time.time() + 2
    while time.time() < deadline:
        if (
            get_job(first["id"])["status"] == "running"
            and get_job(second["id"])["status"] == "queued"
        ):
            break
        time.sleep(0.02)
    assert get_job(second["id"])["status"] == "queued"
    cancelled = cancel_job(second["id"])
    assert cancelled["status"] == "cancelled"
    blocker.set()
    deadline = time.time() + 5
    while time.time() < deadline:
        if get_job(first["id"])["status"] in {"done", "error", "cancelled"}:
            break
        time.sleep(0.05)
    assert get_job(first["id"])["status"] == "done"
    assert get_job(second["id"])["status"] == "cancelled"


def test_iter_job_snapshots_reaches_done(studio_home: Path, monkeypatch) -> None:
    monkeypatch.setenv("MINIMAX_STUDIO_STUB", "1")
    job = start_job(JobRequest(kind="music", backend="stub", prompt="snap", lyrics=""))
    snaps = [item for item in iter_job_snapshots(job["id"], heartbeat_s=0.05) if item]
    assert snaps
    assert snaps[-1]["status"] == "done"
    assert "seq" in snaps[-1]


def _wait_terminal(job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        current = get_job(job_id)
        if current["status"] in {"done", "error", "cancelled"}:
            return current
        time.sleep(0.05)
    return get_job(job_id)


def test_cancel_midrun_marks_cancelled_not_error(
    studio_home: Path, monkeypatch
) -> None:
    """A cancel that interrupts sampling must land in `cancelled`."""
    monkeypatch.setenv("MINIMAX_STUDIO_STUB", "1")
    import minimax_studio.worker.backends.music as music_mod
    from minimax_studio.worker import jobs

    def slow_but_cancellable(job_id, request):
        for _ in range(200):
            if jobs.is_cancelled(job_id):
                raise jobs.CancelledError("Cancelled")
            time.sleep(0.02)
        return {"output_path": "x.wav", "media_type": "audio"}

    monkeypatch.setattr(music_mod, "generate_music", slow_but_cancellable)
    job = start_job(JobRequest(kind="music", backend="stub", prompt="slow"))
    deadline = time.time() + 2
    while time.time() < deadline and get_job(job["id"])["status"] != "running":
        time.sleep(0.02)
    cancel_job(job["id"])
    final = _wait_terminal(job["id"])
    assert final["status"] == "cancelled", final
    assert not final.get("error")


def test_legacy_plain_cancelled_error_also_lands_cancelled(
    studio_home: Path, monkeypatch
) -> None:
    """Backends raising plain RuntimeError('Cancelled') stay quiet too."""
    monkeypatch.setenv("MINIMAX_STUDIO_STUB", "1")
    import minimax_studio.worker.backends.music as music_mod
    from minimax_studio.worker import jobs

    def legacy_cancel(job_id, request):
        for _ in range(200):
            if jobs.is_cancelled(job_id):
                raise RuntimeError("Cancelled")
            time.sleep(0.02)
        return {"output_path": "x.wav", "media_type": "audio"}

    monkeypatch.setattr(music_mod, "generate_music", legacy_cancel)
    job = start_job(JobRequest(kind="music", backend="stub", prompt="legacy"))
    deadline = time.time() + 2
    while time.time() < deadline and get_job(job["id"])["status"] != "running":
        time.sleep(0.02)
    cancel_job(job["id"])
    final = _wait_terminal(job["id"])
    assert final["status"] == "cancelled", final
    assert not final.get("error")


def test_real_error_during_cancel_stays_error(
    studio_home: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MINIMAX_STUDIO_STUB", "1")
    import minimax_studio.worker.backends.music as music_mod

    def boom(job_id, request):
        time.sleep(0.1)
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(music_mod, "generate_music", boom)
    job = start_job(JobRequest(kind="music", backend="stub", prompt="doomed"))
    deadline = time.time() + 2
    while time.time() < deadline and get_job(job["id"])["status"] != "running":
        time.sleep(0.02)
    cancel_job(job["id"])
    final = _wait_terminal(job["id"])
    assert final["status"] == "error"
    assert "out of memory" in (final.get("error") or "")
