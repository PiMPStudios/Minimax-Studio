import json

from fastapi.testclient import TestClient

from minimax_studio.worker.server import app


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
    for _ in range(50):
        job = client.get(f"/jobs/{job_id}")
        if job.json()["status"] in {"done", "error"}:
            break
    assert job.json()["status"] == "done"
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
