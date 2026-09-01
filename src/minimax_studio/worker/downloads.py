from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from minimax_studio.worker.catalog import PACKS, Pack, pack_or_raise
from minimax_studio.worker.fsutil import dir_bytes, first_existing
from minimax_studio.worker.model_paths import pack_status as _pack_status
from minimax_studio.worker.model_paths import search_roots
from minimax_studio.worker.runtime import runtime

SnapshotFn = Callable[..., str]


class _DownloadCancelled(Exception):
    """Stop requested while the Hugging Face snapshot process was still running."""


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
    vram = float(hw.get("vram_gb") or 0)
    ram = float(hw.get("ram_gb") or 0)
    if hw.get("cuda"):
        recommended.update({"music3-comfy", "h3-fl2va", "h3-turbo"})
        if vram >= 16 or ram >= 64:
            recommended.add("h3-ref2va")
        if vram >= 24:
            recommended.add("h3-train")
        if vram >= 40:
            recommended.add("h3-diffusers-fl2va")
            recommended.add("music3-cuda")
    if hw.get("apple_silicon"):
        recommended.add("music3-mlx")
    rows = []
    for pack in PACKS.values():
        row = pack_status(pack, root)
        row["recommended"] = pack.id in recommended
        rows.append(row)
    return rows


def start_download(
    pack_id: str, snapshot: SnapshotFn | None = None, force: bool = False
) -> dict[str, Any]:
    pack = pack_or_raise(pack_id)
    dest = runtime.config.models_root() / pack.local_dir
    dest.mkdir(parents=True, exist_ok=True)
    if pack.approx_gb and not force:
        import shutil

        try:
            free_gb = shutil.disk_usage(dest).free / (1024**3)
        except OSError:
            free_gb = float("inf")
        if free_gb < pack.approx_gb:
            raise RuntimeError(
                f"Not enough free disk on the models volume: {free_gb:.0f} GB free, "
                f"“{pack.title}” needs about {pack.approx_gb:.0f} GB. Free up space "
                "or choose Download anyway."
            )
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
        runtime.download_stops[job_id] = threading.Event()
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


def cancel_download(job_id: str) -> dict[str, Any]:
    with runtime.lock:
        record = runtime.downloads.get(job_id)
        if record is None:
            raise KeyError(job_id)
        if record.get("status") in {"done", "error", "cancelled"}:
            return dict(record)
        record["status"] = "cancelling"
        record["message"] = "Cancel requested — Hugging Face may finish the current file"
        stop = runtime.download_stops.get(job_id)
        if stop:
            stop.set()
        return dict(record)


def _stopped(job_id: str) -> bool:
    stop = runtime.download_stops.get(job_id)
    return bool(stop and stop.is_set())


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
        kwargs = {
            "repo_id": pack.repo_id,
            "local_dir": str(dest),
            "token": runtime.config.hf_token or None,
            "allow_patterns": list(pack.allow_patterns) if pack.allow_patterns else None,
            "ignore_patterns": list(pack.ignore_patterns) if pack.ignore_patterns else None,
        }
        if snapshot is not None:
            snapshot(**kwargs)
        else:
            _cancellable_hf_snapshot(job_id, **kwargs)
        if _stopped(job_id):
            _update(job_id, status="cancelled", message="Cancelled")
            return
        status = pack_status(pack, runtime.config.models_root())
        if not status["ready"] and pack.marker_files:
            # Some MLX packs use different markers; accept any file present.
            if first_existing(dest, pack.marker_files) is None and dir_bytes(dest) == 0:
                raise RuntimeError("download finished but pack files are missing")
        _ensure_license(pack.repo_id, dest, runtime.config.hf_token)
        from minimax_studio.worker.model_paths import reset_bytes_cache

        reset_bytes_cache()
        _update(
            job_id,
            status="done",
            message="Ready",
            bytes=dir_bytes(dest),
            error=None,
        )
    except _DownloadCancelled:
        _update(job_id, status="cancelled", message="Cancelled")
        return
    except Exception as exc:
        if _stopped(job_id):
            _update(job_id, status="cancelled", message="Cancelled")
            return
        _update(job_id, status="error", message="Download failed", error=str(exc))
    finally:
        stop.set()
        from minimax_studio.worker.model_paths import reset_bytes_cache

        reset_bytes_cache()


def _watch_size(job_id: str, dest: Path, stop: threading.Event) -> None:
    while not stop.wait(0.4):
        _update(job_id, bytes=dir_bytes(dest))


def kill_active_downloads() -> None:
    """Worker shutdown: pack pulls must not outlive Studio."""
    for job_id, proc in list(runtime.download_procs.items()):
        _kill_snapshot(proc)
        runtime.download_procs.pop(job_id, None)


def _cancellable_hf_snapshot(job_id: str, **kwargs: Any) -> str:
    """``snapshot_download`` has no cancel hook; run it in a child we can kill."""
    import json
    import subprocess
    import sys

    payload = {key: value for key, value in kwargs.items() if value is not None}
    dest = Path(str(kwargs.get("local_dir") or "."))
    dest.mkdir(parents=True, exist_ok=True)
    err_path = dest / ".hf-snapshot.err"
    err_handle = err_path.open("wb")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import json,sys; from huggingface_hub import snapshot_download; "
            "snapshot_download(**json.load(sys.stdin))",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=err_handle,
        start_new_session=True,
    )
    runtime.download_procs[job_id] = proc
    assert proc.stdin is not None
    try:
        proc.stdin.write(json.dumps(payload).encode("utf-8"))
        proc.stdin.close()
        while proc.poll() is None:
            if _stopped(job_id):
                _kill_snapshot(proc)
                raise _DownloadCancelled()
            try:
                proc.wait(timeout=0.4)
            except subprocess.TimeoutExpired:
                continue
        if proc.returncode not in (0, None):
            if _stopped(job_id):
                raise _DownloadCancelled()
            err = b""
            try:
                err = err_path.read_bytes()[-400:]
            except OSError:
                pass
            raise RuntimeError(
                f"download process failed: {err.decode('utf-8', 'replace')}"
            )
    finally:
        runtime.download_procs.pop(job_id, None)
        try:
            err_handle.close()
        except OSError:
            pass
        if proc.poll() is None:
            _kill_snapshot(proc)
        try:
            err_path.unlink()
        except OSError:
            pass
    return str(kwargs.get("local_dir") or "")


def _kill_snapshot(proc: Any) -> None:
    import os
    import signal
    import subprocess

    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=8,
            )
        else:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=2)
    except (ProcessLookupError, PermissionError, OSError, subprocess.SubprocessError):
        pass


def _ensure_license(repo_id: str, dest: Path, token: str | None) -> None:
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "NOTICE"):
        if (dest / name).is_file():
            return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "NOTICE"):
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=name,
                local_dir=str(dest),
                token=token,
            )
            return
        except Exception:
            continue


def _requires_chain(pack: Pack) -> list[Pack]:
    """Transitive `requires` closure of a pack (same or other folders)."""
    seen: set[str] = set()
    queue = list(pack.requires)
    chain: list[Pack] = []
    while queue:
        pack_id = queue.pop()
        if pack_id in seen:
            continue
        seen.add(pack_id)
        item = PACKS.get(pack_id)
        if item is None:
            continue
        chain.append(item)
        queue.extend(item.requires)
    return chain


def delete_pack(pack_id: str, delete_shared: bool = False) -> dict[str, Any]:
    """Remove a pack's Studio copy. Never touches a ComfyUI folder.

    When other installed packs (or their requirements) share the folder,
    only files they do not need are removed — unless ``delete_shared`` says
    otherwise. Reports bytes actually freed.
    """
    pack = pack_or_raise(pack_id)
    root = runtime.config.models_root().resolve()
    dest = (root / pack.local_dir).resolve()
    if dest != root and root not in dest.parents:
        raise RuntimeError("refusing to delete outside the Studio models folder")
    result: dict[str, Any] = {
        "ok": True,
        "id": pack_id,
        "removed": False,
        "removed_bytes": 0,
        "folder_kept": False,
        "kept_for": [],
        "kept_files": [],
    }
    if not dest.exists():
        return result
    before = dir_bytes(dest)
    # Installed means every marker is present. Configs that h3-train shares
    # with official FL2VA must not keep the 63 GB Qwen tree after that pack
    # is deleted, and must not make the 130 GB generate pack look ready.
    in_use = [
        other
        for other in PACKS.values()
        if other.local_dir == pack.local_dir
        and other.id != pack.id
        and other.marker_files
        and all((dest / marker).is_file() for marker in other.marker_files)
    ]
    if in_use and not delete_shared:
        # A pack with no allow_patterns is a full-folder snapshot (official
        # FL2VA). The training-files slice lives in that same tree — deleting
        # unique train markers would gut the generate pack.
        owns_folder = any(other.allow_patterns is None for other in in_use)
        protected: set[str] = set()
        for other in in_use:
            protected.update(other.marker_files)
            for needed in _requires_chain(other):
                if needed.local_dir == pack.local_dir:
                    protected.update(needed.marker_files)
        if not owns_folder:
            for marker in pack.marker_files or ():
                if marker in protected:
                    result["kept_files"].append(marker)
                    continue
                path = dest / marker
                if path.is_file():
                    try:
                        path.unlink()
                    except OSError:
                        result["kept_files"].append(marker)
        else:
            result["kept_files"] = list(pack.marker_files or ())
        result["removed"] = not owns_folder
        result["removed_bytes"] = max(0, before - dir_bytes(dest))
        result["folder_kept"] = True
        result["shared"] = True
        result["kept_for"] = sorted({other.title for other in in_use})
        from minimax_studio.worker.model_paths import reset_bytes_cache

        reset_bytes_cache()
        return result
    import shutil

    shutil.rmtree(dest)
    result["removed"] = True
    result["removed_bytes"] = before
    result["shared"] = bool(in_use)
    result["deleted_shared"] = bool(in_use) and delete_shared
    from minimax_studio.worker.model_paths import reset_bytes_cache

    reset_bytes_cache()
    return result
