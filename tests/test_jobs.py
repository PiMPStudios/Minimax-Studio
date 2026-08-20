from pathlib import Path
import threading
import time

from minimax_studio.worker.jobs import JobRequest, get_job, iter_job_snapshots, start_job


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
