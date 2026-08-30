from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import httpx

from minimax_studio.worker.jobs import CancelledError, JobRequest, is_cancelled, update_job
from minimax_studio.worker.runtime import runtime


def _cancellable_post(
    url: str,
    *,
    headers: dict[str, str],
    json: dict[str, Any],
    job_id: str,
    timeout: float = 180.0,
) -> httpx.Response:
    """POST that abortable-closes the client when the job is cancelled."""
    if is_cancelled(job_id):
        raise CancelledError("Cancelled")
    box: dict[str, Any] = {}
    done = threading.Event()
    client = httpx.Client(timeout=timeout)

    def _go() -> None:
        try:
            box["response"] = client.post(url, headers=headers, json=json)
        except Exception as exc:
            box["exc"] = exc
        finally:
            done.set()
            try:
                client.close()
            except Exception:
                pass

    threading.Thread(target=_go, daemon=True, name=f"music-api-{job_id}").start()
    while not done.wait(0.4):
        if is_cancelled(job_id):
            try:
                client.close()
            except Exception:
                pass
    if is_cancelled(job_id):
        raise CancelledError("Cancelled")
    if "exc" in box:
        raise box["exc"]
    response = box.get("response")
    if response is None:
        raise RuntimeError("Music API call returned no response")
    return response


def generate_music_api(job_id: str, request: JobRequest) -> dict[str, Any]:
    key = runtime.config.minimax_api_key
    if not key:
        raise RuntimeError("Set a MiniMax API key in Settings for the Music API.")
    base = (runtime.config.minimax_api_base or "https://api.minimax.io").rstrip("/")
    lyrics = (request.lyrics or "").strip()
    payload: dict[str, Any] = {
        "model": "music-3.0",
        "prompt": (request.prompt or "")[:2000],
        "output_format": "hex",
        "is_instrumental": False,
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 256000,
            "format": "wav",
        },
    }
    if lyrics:
        payload["lyrics"] = lyrics[:3500]
    else:
        # MiniMax's prompt-driven song: empty lyrics + optimizer writes lyrics
        # from the prompt. Forcing instrumental here made local vs API disagree.
        payload["lyrics_optimizer"] = True
    update_job(job_id, message="Calling MiniMax Music 3.0 API", progress=0.2)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    response = _cancellable_post(
        f"{base}/v1/music_generation",
        headers=headers,
        json=payload,
        job_id=job_id,
    )
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
