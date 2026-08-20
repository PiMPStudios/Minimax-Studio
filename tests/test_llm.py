from minimax_studio.worker.llm import enhance_prompt


def test_enhance_music_uses_chat_completions(studio_home, monkeypatch) -> None:
    calls = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"choices": [{"message": {"content": "Genre: folk. BPM: 92. Warm male vocal."}}]}

    class FakeClient:
        def __init__(self, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            calls["url"] = url
            calls["json"] = json
            return FakeResponse()

    monkeypatch.setattr("minimax_studio.worker.llm.httpx.Client", FakeClient)
    monkeypatch.setenv("MINIMAX_STUDIO_LLM_KEY", "test-key")
    from minimax_studio.worker.runtime import runtime

    runtime.config.llm_api_key = None
    runtime.config.llm_base_url = "http://127.0.0.1:8080/v1"
    runtime.config.llm_model = "qwen3.8-27b-q4kxl"
    text = enhance_prompt("music", "sad banjo song")
    assert "folk" in text.lower()
    assert calls["url"].endswith("/chat/completions")
    assert calls["json"]["model"] == "qwen3.8-27b-q4kxl"
