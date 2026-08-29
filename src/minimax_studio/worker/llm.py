from __future__ import annotations

from typing import Any

import httpx

from minimax_studio.worker.runtime import runtime

MUSIC_SYSTEM = """You expand a short music idea into a MiniMax-Music3 caption.
Return ONLY the caption, no quotes, no markdown.
Include: genre, BPM, key if obvious, vocal description, and arrangement that develops over the song.
Keep it under 120 words. Do not write lyrics."""

LYRICS_SYSTEM = """You write song lyrics for MiniMax-Music3.
Return ONLY lyrics. Put structure tags on their own lines: [Intro], [Verse], [Pre-Chorus], [Chorus], [Post-Chorus], [Bridge], [Solo], [Outro].
Keep it singable, under 80 lines. No commentary."""

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
    if kind == "h3":
        system = VIDEO_SYSTEM
    elif kind == "lyrics":
        system = LYRICS_SYSTEM
    else:
        system = MUSIC_SYSTEM
    user = source
    if extra.strip():
        user += f"\n\nExisting lyrics or notes (do not rewrite unless asked):\n{extra.strip()}"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload: dict[str, Any] = {
        "model": config.llm_model or "qwen3.8-27b-q4kxl",
        "temperature": 0.6,
        "max_tokens": 1600,
        "thinking_budget_tokens": 512,
        "chat_template_kwargs": {
            "enable_thinking": True,
            "reasoning_effort": "medium",
        },
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
        message = data["choices"][0]["message"]
        content = message.get("content") or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected LLM response: {data}") from exc
    text_out = str(content).strip()
    if not text_out:
        raise RuntimeError("Local LLM returned empty content (thinking used the token budget).")
    if text_out.startswith("```"):
        text_out = text_out.strip("`")
        if "\n" in text_out:
            text_out = text_out.split("\n", 1)[1]
    return text_out.strip()
