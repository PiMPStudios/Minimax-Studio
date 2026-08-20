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
