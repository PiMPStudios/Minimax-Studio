from __future__ import annotations

import json
import os
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.requests import Request

from minimax_studio import __version__
from minimax_studio.config import AppConfig, save_config
from minimax_studio.worker import downloads
from minimax_studio.worker.backends.h3_api import run_context_ir
from minimax_studio.worker.history import delete_entry, get_entry, list_history
from minimax_studio.worker.jobs import (
    JobRequest,
    cancel_job,
    get_job,
    iter_job_snapshots,
    list_jobs,
    start_job,
)
from minimax_studio.worker.llm import enhance_prompt
from minimax_studio.worker.loras import import_lora, list_loras
from minimax_studio.worker.ping import ping_services
from minimax_studio.worker.preflight import preflight as run_preflight
from minimax_studio.worker.presets import delete_preset, list_presets, save_preset
from minimax_studio.worker.probe import probe
from minimax_studio.worker.runtime import runtime

app = FastAPI(title="MiniMax Studio Worker", version=__version__)

AUTH_ENV = "MINIMAX_STUDIO_WORKER_TOKEN"
TOKEN_HEADER = "X-Minimax-Studio-Token"


@app.middleware("http")
async def require_token(request: Request, call_next):
    """Shared-secret gate. The GUI generates a token per launch and passes it
    via env; without it any local process could read tokens from /settings
    or queue jobs. Unset env (``--worker-only`` dev mode) keeps the worker
    open on purpose.
    """
    token = os.environ.get(AUTH_ENV, "")
    if token and request.headers.get(TOKEN_HEADER) != token:
        return JSONResponse(
            {"detail": "missing or wrong worker token"}, status_code=401
        )
    return await call_next(request)


class SettingsIn(BaseModel):
    output_dir: str | None = None
    models_dir: str | None = None
    hf_token: str | None = None
    minimax_api_key: str | None = None
    minimax_api_base: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: str | None = None
    comfy_models_dir: str | None = None
    comfy_url: str | None = None
    comfy_root: str | None = None
    comfy_extra_args: str | None = None
    cuda_device: int | None = None
    use_os_keyring: bool | None = None


@app.get("/health")
def health() -> dict[str, object]:
    return {"ok": True, "version": __version__, "service": "minimax-studio-worker"}


@app.get("/probe")
def hardware_probe() -> dict[str, object]:
    info = probe()
    try:
        packs = downloads.list_packs()
        ready = [item for item in packs if item.get("ready")]
        info["packs_ready"] = [item["id"] for item in ready]
        info["packs_ready_titles"] = [item["title"] for item in ready]
        info["packs_from_comfy"] = sum(1 for item in ready if item.get("source") == "comfy")
    except Exception:
        info["packs_ready"] = []
        info["packs_ready_titles"] = []
        info["packs_from_comfy"] = 0
    return info


@app.get("/settings")
def get_settings() -> dict[str, object]:
    return runtime.config.model_dump()


@app.get("/ping")
def ping() -> dict[str, object]:
    return ping_services()


@app.get("/comfy")
def comfy_status() -> dict[str, object]:
    from minimax_studio.worker.comfy_launch import detect_comfy

    return detect_comfy()


@app.post("/comfy/start")
def comfy_start() -> dict[str, object]:
    from minimax_studio.worker.comfy_launch import start_comfy

    try:
        return start_comfy()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/preflight")
def preflight(
    kind: str = "h3",
    backend: str = "auto",
    mode: str = "t2va",
    speed: str = "quality",
    resolution: str = "768P",
) -> dict[str, object]:
    return run_preflight(kind, backend, mode, speed, resolution)


@app.post("/settings")
def post_settings(body: SettingsIn) -> dict[str, object]:
    data = runtime.config.model_dump()
    incoming = body.model_dump()
    for key, value in incoming.items():
        if value == "":
            data[key] = None
        elif value is not None:
            data[key] = value
    runtime.h3_pipe = None
    runtime.h3_pipe_path = None
    runtime.music_pipe = None
    runtime.music_pipe_path = None
    from minimax_studio.worker.backends.h3_comfy import (
        reset_comfy_object_cache,
        reset_comfy_reach_cache,
    )

    reset_comfy_reach_cache()
    reset_comfy_object_cache()
    config = AppConfig.model_validate(data)
    save_config(config)
    runtime.config = config
    from minimax_studio.worker.model_paths import reset_bytes_cache

    reset_bytes_cache()
    try:
        config.ensure_dirs()
    except RuntimeError:
        pass
    return config.model_dump()


@app.get("/packs")
def packs() -> list[dict[str, object]]:
    try:
        return downloads.list_packs()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class DownloadIn(BaseModel):
    pack_id: str
    force: bool = False


@app.delete("/packs/{pack_id}")
def remove_pack(pack_id: str, delete_shared: bool = False) -> dict[str, object]:
    try:
        return downloads.delete_pack(pack_id, delete_shared=delete_shared)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/downloads")
def create_download(body: DownloadIn) -> dict[str, object]:
    try:
        return downloads.start_download(body.pack_id, force=body.force)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/downloads")
def all_downloads() -> list[dict[str, object]]:
    return downloads.list_downloads()


@app.post("/downloads/{job_id}/cancel")
def stop_download(job_id: str) -> dict[str, object]:
    try:
        return downloads.cancel_download(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/downloads/{job_id}")
def one_download(job_id: str) -> dict[str, object]:
    try:
        return downloads.get_download(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/jobs")
def create_job(body: JobRequest) -> dict[str, object]:
    try:
        return start_job(body)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/jobs")
def all_jobs() -> list[dict[str, object]]:
    return list_jobs()


@app.get("/jobs/{job_id}")
def one_job(job_id: str) -> dict[str, object]:
    try:
        return get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/jobs/{job_id}/events")
def job_events(job_id: str) -> StreamingResponse:
    try:
        get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    def generate():
        try:
            for snap in iter_job_snapshots(job_id):
                if snap is None:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(snap)}\n\n"
            yield "event: end\ndata: {}\n\n"
        except KeyError:
            yield f"event: error\ndata: {json.dumps({'detail': 'job not found'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/jobs/{job_id}/cancel")
def stop_job(job_id: str) -> dict[str, object]:
    try:
        return cancel_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/history")
def history() -> list[dict[str, object]]:
    try:
        return list_history()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/history/{entry_id}")
def history_item(entry_id: str) -> dict[str, object]:
    try:
        return get_entry(entry_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/history/{entry_id}")
def remove_history(entry_id: str) -> dict[str, object]:
    try:
        get_entry(entry_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    delete_entry(entry_id)
    return {"ok": True, "id": entry_id}


class PresetIn(BaseModel):
    id: str | None = None
    name: str = "Untitled"
    kind: str = "music"
    backend: str = "auto"
    mode: str | None = None
    prompt: str = ""
    lyrics: str = ""
    duration_s: float = 30
    seed: int = -1
    steps: int = 30
    width: int = 960
    height: int = 544
    resolution: str = "768P"
    ratio: str = "16:9"
    speed: str = "quality"
    attention: str = "default"
    ref_image_size: str = "match"
    quality: str = "native"
    cfg: float = 1.7
    assets: list[dict[str, str]] = []
    loras: list[dict[str, Any]] = []
    lora_id: str = ""
    lora_strength: float = 1.0
    lora2_id: str = ""
    lora2_strength: float = 1.0


@app.get("/presets")
def presets() -> list[dict[str, object]]:
    return list_presets()


@app.post("/presets")
def create_preset(body: PresetIn) -> dict[str, object]:
    return save_preset(body.model_dump())


@app.delete("/presets/{preset_id}")
def remove_preset(preset_id: str) -> dict[str, object]:
    delete_preset(preset_id)
    return {"ok": True, "id": preset_id}


@app.get("/loras")
def loras() -> list[dict[str, object]]:
    try:
        return list_loras()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class LoraImportIn(BaseModel):
    path: str


@app.post("/loras/import")
def lora_import(body: LoraImportIn) -> dict[str, object]:
    try:
        return import_lora(body.path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class LyricsIn(BaseModel):
    prompt: str
    notes: str = ""


@app.post("/lyrics")
def write_lyrics(body: LyricsIn) -> dict[str, object]:
    from minimax_studio.worker.llm import enhance_prompt

    seed = (
        "Write MiniMax-Music3 lyrics only. Use section tags like [Verse] and [Chorus] "
        "on their own lines. Theme:\n"
        + body.prompt
    )
    try:
        return {"text": enhance_prompt("lyrics", seed, body.notes)}
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class EnhanceIn(BaseModel):
    kind: str = "music"
    text: str
    extra: str = ""


class ContextIRIn(BaseModel):
    prompt: str
    mode: str = "t2va"
    duration_s: float = 8
    ratio: str = "16:9"
    assets: list[dict[str, str]] = []


@app.post("/context-ir")
def context_ir(body: ContextIRIn) -> dict[str, object]:
    request = JobRequest(
        kind="h3",
        mode=body.mode,
        prompt=body.prompt,
        duration_s=body.duration_s,
        ratio=body.ratio,
        assets=body.assets,
    )
    try:
        return {"text": run_context_ir(request)}
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Context-IR failed: {exc}") from exc


@app.post("/enhance")
def enhance(body: EnhanceIn) -> dict[str, object]:
    try:
        return {"text": enhance_prompt(body.kind, body.text, body.extra)}
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Local LLM failed: {exc}") from exc
