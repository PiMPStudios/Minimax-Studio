import json
import time

from fastapi.testclient import TestClient

from minimax_studio.worker.server import app


def _wait_for_job(client: TestClient, job_id: str, timeout: float = 60.0) -> dict:
    """A generate job runs in its own thread. A fixed number of in-process
    polls races thread startup and loses on a loaded runner (it did on
    Windows CI) — wait on wall-clock time, then judge the outcome."""
    deadline = time.time() + timeout
    job: dict = {}
    while time.time() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job.get("status") in {"done", "error"}:
            return job
        time.sleep(0.05)
    return job


def test_packs_and_stub_job(studio_home) -> None:
    client = TestClient(app)
    packs = client.get("/packs")
    assert packs.status_code == 200
    ids = {item["id"] for item in packs.json()}
    assert "music3-cuda" in ids
    assert "music3-comfy" in ids
    assert "h3-fl2va" in ids
    created = client.post(
        "/jobs",
        json={"kind": "music", "backend": "stub", "prompt": "test", "lyrics": ""},
    )
    assert created.status_code == 200
    job_id = created.json()["id"]
    assert _wait_for_job(client, job_id)["status"] == "done"
    history = client.get("/history")
    assert history.status_code == 200
    assert any(item["id"] == job_id for item in history.json())


def test_job_sse_stream(studio_home, monkeypatch) -> None:
    monkeypatch.setenv("MINIMAX_STUDIO_STUB", "1")
    client = TestClient(app)
    created = client.post(
        "/jobs",
        json={"kind": "music", "backend": "stub", "prompt": "sse", "lyrics": ""},
    )
    assert created.status_code == 200
    job_id = created.json()["id"]
    events = []
    with client.stream("GET", f"/jobs/{job_id}/events") as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        for line in response.iter_lines():
            if isinstance(line, bytes):
                line = line.decode()
            if line.startswith("data: "):
                payload = line[6:].strip()
                if payload and payload != "{}":
                    events.append(json.loads(payload))
            if line.startswith("event: end"):
                break
    assert events
    assert events[-1]["status"] == "done"
    assert events[-1]["id"] == job_id


def test_job_sse_404(studio_home) -> None:
    client = TestClient(app)
    response = client.get("/jobs/no-such-job/events")
    assert response.status_code == 404


def test_comfy_detect_route(studio_home) -> None:
    client = TestClient(app)
    response = client.get("/comfy")
    assert response.status_code == 200
    body = response.json()
    assert "root" in body
    assert "running" in body


def test_worker_token_gate(studio_home, monkeypatch) -> None:
    from minimax_studio.worker.server import TOKEN_HEADER

    client = TestClient(app)
    assert client.get("/health").status_code == 200  # open dev mode

    monkeypatch.setenv("MINIMAX_STUDIO_WORKER_TOKEN", "s3cret")
    assert client.get("/health").status_code == 401
    assert client.post("/jobs", json={"kind": "music"}).status_code == 401
    ok = client.get("/health", headers={TOKEN_HEADER: "s3cret"})
    assert ok.status_code == 200
    assert client.get("/health", headers={TOKEN_HEADER: "wrong"}).status_code == 401


def test_lora_import_rejects_non_safetensors(studio_home, tmp_path) -> None:
    evil = tmp_path /"payload.bin"
    evil.write_bytes(b"not a lora")
    client = TestClient(app)
    response = client.post("/loras/import", json={"path": str(evil)})
    assert response.status_code == 400
    assert "safetensors" in response.json()["detail"]


def test_file_data_url_rejects_unknown_type(tmp_path) -> None:
    from minimax_studio.worker.backends.h3_api import _file_data_url

    secret = tmp_path / "id_rsa"
    secret.write_text("private")
    try:
        _file_data_url(str(secret))
        raise AssertionError("expected rejection of unknown extension")
    except RuntimeError as exc:
        assert "Unsupported asset type" in str(exc)


def test_history_trim_route(studio_home, tmp_path, monkeypatch) -> None:
    import sys
    from pathlib import Path

    from minimax_studio.worker.history import record_entry

    script = tmp_path / "ffmpeg_stub.py"
    script.write_text(
        "import pathlib, sys\npathlib.Path(sys.argv[-1]).write_bytes(b'trimmed')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "MINIMAX_STUDIO_FFMPEG_BIN", f'"{sys.executable}" "{script}"'
    )
    folder = studio_home / "history" / "parentparent"
    folder.mkdir(parents=True)
    src = folder / "audio.wav"
    src.write_bytes(b"RIFF")
    record_entry(
        {
            "id": "parentparent",
            "kind": "music",
            "prompt": "folk",
            "duration_s": 8,
            "output_path": str(src),
        }
    )
    client = TestClient(app)
    payload = client.post(
        "/history/parentparent/trim", json={"start_s": 0, "end_s": 2}
    )
    assert payload.status_code == 200, payload.text
    body = payload.json()
    assert body["trimmed_from"] == "parentparent"
    assert Path(body["output_path"]).read_bytes() == b"trimmed"

    monkeypatch.delenv("MINIMAX_STUDIO_FFMPEG_BIN", raising=False)
    monkeypatch.setattr(
        "minimax_studio.worker.trim.shutil.which", lambda _name: None
    )
    missing = client.post(
        "/history/parentparent/trim", json={"start_s": 0, "end_s": 1}
    )
    assert missing.status_code == 400
    assert "install ffmpeg" in missing.json()["detail"]
