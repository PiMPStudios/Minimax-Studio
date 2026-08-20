from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

from minimax_studio.worker.history import record_entry
from minimax_studio.worker.runtime import runtime


class JobRequest(BaseModel):
    kind: str
    backend: str = "auto"
    mode: str = "ttm"
    prompt: str = ""
    lyrics: str = ""
    duration_s: float = 30
    seed: int = -1
    steps: int = 30
    assets: list[dict[str, str]] = Field(default_factory=list)
    loras: list[dict[str, Any]] = Field(default_factory=list)
    speed: str = "quality"
    width: int = 960
    height: int = 544
    resolution: str = "768P"
    ratio: str = "16:9"


def start_job(request: JobRequest) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    record: dict[str, Any] = {
        "id": job_id,
        "kind": request.kind,
        "backend": request.backend,
        "mode": request.mode,
        "status": "queued",
        "progress": 0.0,
        "message": "Queued",
        "error": None,
        "output_path": None,
        "created_at": time.time(),
        "request": request.model_dump(),
    }
    with runtime.lock:
        runtime.jobs[job_id] = record
    thread = threading.Thread(
        target=_run_job, args=(job_id, request), daemon=True, name=f"job-{job_id}"
    )
    thread.start()
    return dict(record)


def get_job(job_id: str) -> dict[str, Any]:
    with runtime.lock:
        record = runtime.jobs.get(job_id)
        if record is None:
            raise KeyError(job_id)
        return dict(record)


def list_jobs() -> list[dict[str, Any]]:
    with runtime.lock:
        jobs = [dict(item) for item in runtime.jobs.values()]
    jobs.sort(key=lambda item: item.get("created_at") or 0, reverse=True)
    return jobs


def update_job(job_id: str, **fields: Any) -> None:
    with runtime.lock:
        if job_id in runtime.jobs:
            runtime.jobs[job_id].update(fields)


def cancel_job(job_id: str) -> dict[str, Any]:
    with runtime.lock:
        record = runtime.jobs.get(job_id)
        if record is None:
            raise KeyError(job_id)
        if record.get("status") in {"done", "error", "cancelled"}:
            return dict(record)
        record["status"] = "cancelling"
        record["message"] = "Cancel requested"
        return dict(record)


def is_cancelled(job_id: str) -> bool:
    with runtime.lock:
        record = runtime.jobs.get(job_id) or {}
        return record.get("status") in {"cancelling", "cancelled"}


def _run_job(job_id: str, request: JobRequest) -> None:
    update_job(job_id, status="running", message="Starting", progress=0.05)
    try:
        if is_cancelled(job_id):
            update_job(job_id, status="cancelled", message="Cancelled")
            return
        if request.kind == "music":
            from minimax_studio.worker.backends.music import generate_music

            result = generate_music(job_id, request)
        elif request.kind == "h3":
            from minimax_studio.worker.backends.h3 import generate_h3

            result = generate_h3(job_id, request)
        else:
            raise RuntimeError(f"unknown job kind: {request.kind}")
        if is_cancelled(job_id):
            update_job(job_id, status="cancelled", message="Cancelled")
            return
        entry = record_entry(
            {
                "id": job_id,
                "kind": request.kind,
                "mode": request.mode,
                "backend": result.get("backend") or request.backend,
                "prompt": request.prompt,
                "lyrics": request.lyrics,
                "duration_s": request.duration_s,
                "seed": request.seed,
                "steps": request.steps,
                "output_path": result["output_path"],
                "media_type": result.get("media_type", "audio"),
            }
        )
        update_job(
            job_id,
            status="done",
            progress=1.0,
            message="Done",
            output_path=result["output_path"],
            history=entry,
        )
    except Exception as exc:
        update_job(job_id, status="error", message="Failed", error=str(exc), progress=0.0)
