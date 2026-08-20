from minimax_studio.worker.probe import probe
from minimax_studio.worker.server import app
from fastapi.testclient import TestClient


def test_probe_has_os_fields() -> None:
    info = probe()
    assert info["os"]
    assert "cuda" in info
    assert "apple_silicon" in info


def test_health_and_probe_routes() -> None:
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["ok"] is True
    probed = client.get("/probe")
    assert probed.status_code == 200
    body = probed.json()
    assert "os" in body
    assert "sageattention" in body
    assert "packs_ready" in body
