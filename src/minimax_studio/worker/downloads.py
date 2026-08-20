from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from minimax_studio.worker.catalog import PACKS, Pack, pack_or_raise
from minimax_studio.worker.fsutil import dir_bytes, first_existing
from minimax_studio.worker.model_paths import pack_status as _pack_status
from minimax_studio.worker.model_paths import search_roots
from minimax_studio.worker.runtime import runtime


SnapshotFn = Callable[..., str]


def pack_status(pack: Pack, models_root: Path) -> dict[str, Any]:
    extra = None
    try:
        extra = search_roots(models_root, runtime.config.comfy_models_dir)
    except Exception:
        extra = [models_root]
    return _pack_status(pack, models_root, extra_roots=extra)


def list_packs() -> list[dict[str, Any]]:
    from minimax_studio.worker.probe import probe

    root = runtime.config.models_root()
    hw = probe()
    recommended = set()
    if hw.get("cuda"):
        recommended.update(
            {"music3-cuda", "music3-comfy", "h3-fl2va", "h3-ref2va", "h3-diffusers-fl2va"}
        )
    if hw.get("apple_silicon"):
        recommended.add("music3-mlx")
    rows = []
    for pack in PACKS.values():
        row = pack_status(pack, root)
        row["recommended"] = pack.id in recommended
        rows.append(row)
    return rows


def start_download(pack_id: str, snapshot: SnapshotFn | None = None) -> dict[str, Any]:
    pack = pack_or_raise(pack_id)
    dest = runtime.config.models_root() / pack.local_dir
    dest.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex[:12]
    record: dict[str, Any] = {
        "id": job_id,
        "pack_id": pack.id,
        "status": "queued",
        "bytes": 0,
        "total_bytes": int(pack.approx_gb * (1024**3)),
        "message": "Starting download",
        "error": None,
        "path": str(dest),
    }
    with runtime.lock:
        runtime.downloads[job_id] = record
    thread = threading.Thread(
        target=_run_download,
        args=(job_id, pack, dest, snapshot),
        daemon=True,
        name=f"download-{pack.id}",
    )
    thread.start()
    return dict(record)


def get_download(job_id: str) -> dict[str, Any]:
    with runtime.lock:
        record = runtime.downloads.get(job_id)
        if record is None:
            raise KeyError(job_id)
        return dict(record)


def list_downloads() -> list[dict[str, Any]]:
    with runtime.lock:
        return [dict(item) for item in runtime.downloads.values()]


def _update(job_id: str, **fields: Any) -> None:
    with runtime.lock:
        if job_id in runtime.downloads:
            runtime.downloads[job_id].update(fields)


def _run_download(
    job_id: str,
    pack: Pack,
    dest: Path,
    snapshot: SnapshotFn | None,
) -> None:
    stop = threading.Event()
    watcher = threading.Thread(target=_watch_size, args=(job_id, dest, stop), daemon=True)
    watcher.start()
    _update(job_id, status="running", message=f"Fetching {pack.repo_id}")
    try:
        fn = snapshot or _hf_snapshot
        fn(
            repo_id=pack.repo_id,
            local_dir=str(dest),
            token=runtime.config.hf_token or None,
            allow_patterns=list(pack.allow_patterns) if pack.allow_patterns else None,
            ignore_patterns=list(pack.ignore_patterns) if pack.ignore_patterns else None,
        )
        status = pack_status(pack, runtime.config.models_root())
        if not status["ready"] and pack.marker_files:
            # Some MLX packs use different markers; accept any file present.
            if first_existing(dest, pack.marker_files) is None and dir_bytes(dest) == 0:
                raise RuntimeError("download finished but pack files are missing")
        _update(
            job_id,
            status="done",
            message="Ready",
            bytes=dir_bytes(dest),
            error=None,
        )
    except Exception as exc:
        _update(job_id, status="error", message="Download failed", error=str(exc))
    finally:
        stop.set()


def _watch_size(job_id: str, dest: Path, stop: threading.Event) -> None:
    while not stop.wait(0.4):
        _update(job_id, bytes=dir_bytes(dest))


def _hf_snapshot(
    repo_id: str,
    local_dir: str,
    token: str | None,
    allow_patterns: list[str] | None,
    ignore_patterns: list[str] | None,
) -> str:
    from huggingface_hub import snapshot_download

    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "local_dir": local_dir,
        "token": token,
        "resume_download": True,
    }
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns
    if ignore_patterns:
        kwargs["ignore_patterns"] = ignore_patterns
    return snapshot_download(**kwargs)
