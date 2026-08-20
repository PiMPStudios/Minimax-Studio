from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from minimax_studio.worker.jobs import JobRequest, update_job
from minimax_studio.worker.runtime import runtime


def generate_music_api(job_id: str, request: JobRequest) -> dict[str, Any]:
    key = runtime.config.minimax_api_key
    if not key:
        raise RuntimeError("Set a MiniMax API key in Settings for the Music API.")
    base = (runtime.config.minimax_api_base or "https://api.minimax.io").rstrip("/")
    lyrics = (request.lyrics or "").strip()
    instrumental = not lyrics
    payload: dict[str, Any] = {
        "model": "music-3.0",
        "prompt": (request.prompt or "")[:2000],
        "output_format": "hex",
        "is_instrumental": instrumental,
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 256000,
            "format": "wav",
        },
    }
    if lyrics:
        payload["lyrics"] = lyrics[:3500]
    elif not instrumental:
        payload["lyrics_optimizer"] = True
    update_job(job_id, message="Calling MiniMax Music 3.0 API", progress=0.2)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=180.0) as client:
        response = client.post(f"{base}/v1/music_generation", headers=headers, json=payload)
        response.raise_for_status()
        body = response.json()
    status = (body.get("base_resp") or {}).get("status_code")
    if status not in (0, None):
        raise RuntimeError(
            f"Music API error {status}: {(body.get('base_resp') or {}).get('status_msg')}"
        )
    hex_audio = (body.get("data") or {}).get("audio")
    if not hex_audio:
        raise RuntimeError(f"Music API returned no audio: {body}")
    dest = Path(runtime.config.history_root() / job_id)
    dest.mkdir(parents=True, exist_ok=True)
    wav_path = dest / "audio.wav"
    wav_path.write_bytes(bytes.fromhex(hex_audio))
    return {"output_path": str(wav_path), "backend": "api", "media_type": "audio"}
