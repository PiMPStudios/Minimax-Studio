from minimax_studio.worker.probe import probe, reset_probe_cache
from minimax_studio.worker.server import app
from fastapi.testclient import TestClient


def test_probe_has_os_fields() -> None:
    reset_probe_cache()
    info = probe()
    assert info["os"]
    assert "cuda" in info
    assert "apple_silicon" in info


def test_nvidia_smi_fallback_when_no_torch(monkeypatch) -> None:
    reset_probe_cache()
    monkeypatch.setattr(
        "minimax_studio.worker.probe._nvidia_smi_gpus",
        lambda: [{"name": "NVIDIA GeForce RTX 3080", "vram_gb": 10.0}],
    )
    info = probe()
    assert info["cuda"] is True
    assert info["cuda_name"] == "NVIDIA GeForce RTX 3080"
    assert info["vram_gb"] == 10.0
    assert info["cuda_source"] in {"nvidia-smi", "torch"}
    if not info["torch_available"]:
        assert info["cuda_source"] == "nvidia-smi"


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
