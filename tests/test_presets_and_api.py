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
