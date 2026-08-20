from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from minimax_studio import __version__
from minimax_studio.config import AppConfig, save_config
from minimax_studio.worker import downloads
from minimax_studio.worker.history import get_entry, list_history
from minimax_studio.worker.jobs import JobRequest, get_job, list_jobs, start_job
from minimax_studio.worker.probe import probe
from minimax_studio.worker.runtime import runtime

app = FastAPI(title="MiniMax Studio Worker", version=__version__)


class SettingsIn(BaseModel):
    output_dir: str | None = None
    models_dir: str | None = None
    hf_token: str | None = None
    minimax_api_key: str | None = None
    minimax_api_base: str | None = None


@app.get("/health")
def health() -> dict[str, object]:
    return {"ok": True, "version": __version__, "service": "minimax-studio-worker"}


@app.get("/probe")
def hardware_probe() -> dict[str, object]:
    return probe()


@app.get("/settings")
def get_settings() -> dict[str, object]:
    return runtime.config.model_dump()


@app.post("/settings")
def post_settings(body: SettingsIn) -> dict[str, object]:
    data = runtime.config.model_dump()
    incoming = body.model_dump(exclude_none=True)
    data.update(incoming)
    config = AppConfig.model_validate(data)
    save_config(config)
    runtime.config = config
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


@app.post("/downloads")
def create_download(body: DownloadIn) -> dict[str, object]:
    try:
        return downloads.start_download(body.pack_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/downloads")
def all_downloads() -> list[dict[str, object]]:
    return downloads.list_downloads()


@app.get("/downloads/{job_id}")
def one_download(job_id: str) -> dict[str, object]:
    try:
        return downloads.get_download(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/jobs")
def create_job(body: JobRequest) -> dict[str, object]:
    return start_job(body)


@app.get("/jobs")
def all_jobs() -> list[dict[str, object]]:
    return list_jobs()


@app.get("/jobs/{job_id}")
def one_job(job_id: str) -> dict[str, object]:
    try:
        return get_job(job_id)
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
