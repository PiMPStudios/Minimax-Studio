from __future__ import annotations

from typing import Any

import httpx

from minimax_studio.worker.runtime import runtime

MUSIC_SYSTEM = """You expand a short music idea into a MiniMax-Music3 caption.
Return ONLY the caption, no quotes, no markdown.
Include: genre, BPM, key if obvious, vocal description, and arrangement that develops over the song.
Keep it under 120 words. Do not write lyrics."""

VIDEO_SYSTEM = """You expand a short video idea into a MiniMax H3 prompt.
Return ONLY the prompt, no quotes, no markdown.
Structure as timed shots with camera, action, dialogue/SFX, and non-diegetic music.
Stay under 7000 characters. Do not invent copyrighted characters."""


def enhance_prompt(kind: str, text: str, extra: str = "") -> str:
    source = (text or "").strip()
    if not source:
        raise RuntimeError("Nothing to enhance.")
    config = runtime.config
    base = (config.llm_base_url or "http://127.0.0.1:8080/v1").rstrip("/")
    key = config.resolved_llm_key()
    if not key:
        raise RuntimeError(
            "No local LLM key. Put one in Settings or ~/.config/llama-api.key"
        )
    system = VIDEO_SYSTEM if kind == "h3" else MUSIC_SYSTEM
    user = source
    if extra.strip():
        user += f"\n\nExisting lyrics or notes (do not rewrite unless asked):\n{extra.strip()}"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload: dict[str, Any] = {
        "model": config.llm_model or "qwen3.8-27b-q4kxl",
        "temperature": 0.6,
        "max_tokens": 900,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    with httpx.Client(timeout=120.0) as client:
        response = client.post(f"{base}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected LLM response: {data}") from exc
    text_out = str(content).strip()
    if text_out.startswith("```"):
        text_out = text_out.strip("`")
        if "\n" in text_out:
            text_out = text_out.split("\n", 1)[1]
    return text_out.strip()
