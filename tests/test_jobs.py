from pathlib import Path
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


def test_iter_job_snapshots_reaches_done(studio_home: Path, monkeypatch) -> None:
    monkeypatch.setenv("MINIMAX_STUDIO_STUB", "1")
    job = start_job(JobRequest(kind="music", backend="stub", prompt="snap", lyrics=""))
    snaps = [item for item in iter_job_snapshots(job["id"], heartbeat_s=0.05) if item]
    assert snaps
    assert snaps[-1]["status"] == "done"
    assert "seq" in snaps[-1]
