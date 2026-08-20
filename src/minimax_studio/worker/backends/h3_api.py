from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path
from typing import Any

import httpx

from minimax_studio.worker.jobs import JobRequest, is_cancelled, update_job
from minimax_studio.worker.runtime import runtime


def generate_h3_api(job_id: str, request: JobRequest) -> dict[str, Any]:
    key = runtime.config.minimax_api_key
    if not key:
        raise RuntimeError("Set a MiniMax API key in Settings for the API backend.")
    base = (runtime.config.minimax_api_base or "https://api.minimax.io").rstrip("/")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = _payload(request)
    update_job(job_id, message="Submitting MiniMax H3 API task", progress=0.1)
    with httpx.Client(timeout=60.0) as client:
        created = client.post(f"{base}/v2/video_generation", headers=headers, json=payload)
        created.raise_for_status()
        body = created.json()
        task_id = body.get("task_id") or (body.get("task") or {}).get("id")
        if not task_id:
            raise RuntimeError(f"API create returned no task_id: {body}")
        update_job(job_id, message=f"API task {task_id}", progress=0.2)
        url = _poll(client, base, headers, task_id, job_id)
        dest = runtime.config.history_root() / job_id
        dest.mkdir(parents=True, exist_ok=True)
        out_path = dest / "video.mp4"
        update_job(job_id, message="Downloading result", progress=0.9)
        video = client.get(url, timeout=120.0)
        video.raise_for_status()
        out_path.write_bytes(video.content)
    return {"output_path": str(out_path), "backend": "api", "media_type": "video"}


def _payload(request: JobRequest) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]
    for item in request.assets:
        path = item.get("path")
        role = item.get("role")
        if not path:
            continue
        data_url = _file_data_url(path)
        suffix = Path(path).suffix.lower()
        if role == "first_frame":
            content.append(
                {"type": "image_url", "role": "first_frame", "image_url": {"url": data_url}}
            )
        elif role == "last_frame":
            content.append(
                {"type": "image_url", "role": "last_frame", "image_url": {"url": data_url}}
            )
        elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif"}:
            content.append(
                {
                    "type": "image_url",
                    "role": "reference_image",
                    "image_url": {"url": data_url},
                }
            )
        elif suffix in {".mp4", ".mov", ".webm", ".mkv"}:
            content.append(
                {
                    "type": "video_url",
                    "role": "reference_video",
                    "video_url": {"url": data_url},
                }
            )
        else:
            content.append(
                {
                    "type": "audio_url",
                    "role": "reference_audio",
                    "audio_url": {"url": data_url},
                }
            )
    duration = int(max(4, min(15, round(request.duration_s))))
    payload: dict[str, Any] = {
        "model": "MiniMax-H3",
        "content": content,
        "duration": duration,
        "resolution": request.resolution or "768P",
    }
    if request.mode == "t2va":
        payload["ratio"] = request.ratio or "16:9"
    return payload


def _poll(
    client: httpx.Client,
    base: str,
    headers: dict[str, str],
    task_id: str,
    job_id: str,
) -> str:
    deadline = time.monotonic() + 20 * 60
    first = True
    while time.monotonic() < deadline:
        if is_cancelled(job_id):
            raise RuntimeError("Cancelled")
        if not first:
            time.sleep(5)
        first = False
        response = client.get(
            f"{base}/v2/query/video_generation/{task_id}",
            headers=headers,
            timeout=30.0,
        )
        response.raise_for_status()
        parsed = response.json()
        task = parsed.get("task") or parsed
        status = task.get("status")
        update_job(job_id, message=f"API {status}", progress=0.5)
        if status == "succeeded":
            url = (task.get("content") or {}).get("url")
            if not url:
                raise RuntimeError(f"API succeeded without url: {task}")
            return url
        if status in {"failed", "cancelled"}:
            raise RuntimeError(f"API task {status}: {task.get('error')}")
    raise RuntimeError("API task timed out after 20 minutes")


_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
}


def _file_data_url(path: str) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        raise RuntimeError(f"Asset not found: {path}")
    raw = file_path.read_bytes()
    mime = _MIME.get(file_path.suffix.lower()) or mimetypes.guess_type(path)[0] or "application/octet-stream"
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"
