from minimax_studio.worker.backends.h3_api import _payload
from minimax_studio.worker.jobs import JobRequest
from minimax_studio.worker.presets import delete_preset, list_presets, save_preset


def test_preset_round_trip(studio_home) -> None:
    saved = save_preset({"name": "Folk", "kind": "music", "prompt": "banjo", "lyrics": "[Verse]\nhi"})
    assert saved["id"]
    names = [item["name"] for item in list_presets()]
    assert "Folk" in names
    delete_preset(saved["id"])
    assert all(item["id"] != saved["id"] for item in list_presets())


def test_h3_api_payload_t2va() -> None:
    payload = _payload(JobRequest(kind="h3", mode="t2va", prompt="a red fox", duration_s=8))
    assert payload["model"] == "MiniMax-H3"
    assert payload["ratio"] == "16:9"
    assert payload["duration"] == 8
    assert payload["content"][0]["text"] == "a red fox"


def test_context_ir_payload_drops_resolution_rules() -> None:
    i2v = _payload(
        JobRequest(kind="h3", mode="i2va", prompt="steam", duration_s=5)
    )
    assert i2v["ratio"] == "adaptive"
    assert i2v["resolution"] == "768P"


def test_preset_keeps_cfg_and_lora_stack(studio_home) -> None:
    saved = save_preset(
        {
            "name": "Stacked",
            "kind": "music",
            "cfg": 3.2,
            "loras": [
                {"id": "/models/loras/a.safetensors", "strength": 0.8},
                {"id": "/models/loras/b.safetensors", "strength": 0.5},
            ],
            "lora_id": "/models/loras/a.safetensors",
            "lora_strength": 0.8,
            "lora2_id": "/models/loras/b.safetensors",
            "lora2_strength": 0.5,
        }
    )
    reloaded = [item for item in list_presets() if item["id"] == saved["id"]][0]
    assert reloaded["cfg"] == 3.2
    assert len(reloaded["loras"]) == 2
    assert reloaded["lora2_id"] == "/models/loras/b.safetensors"
    assert reloaded["lora2_strength"] == 0.5
    delete_preset(saved["id"])


def test_preset_route_validates_payload(studio_home) -> None:
    from fastapi.testclient import TestClient

    from minimax_studio.worker.server import app

    client = TestClient(app)
    created = client.post(
        "/presets",
        json={
            "name": "Via route",
            "kind": "h3",
            "mode": "t2va",
            "cfg": 2.5,
            "loras": [{"id": "/x/turbo.safetensors", "strength": 1.0}],
        },
    )
    assert created.status_code == 200
    preset_id = created.json()["id"]
    rows = client.get("/presets").json()
    row = [item for item in rows if item["id"] == preset_id][0]
    assert row["cfg"] == 2.5
    assert row["loras"][0]["id"] == "/x/turbo.safetensors"

    bad = client.post("/presets", json={"name": "Nope", "cfg": "not-a-number"})
    assert bad.status_code == 422
