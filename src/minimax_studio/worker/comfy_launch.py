"""Detect and start a user-owned ComfyUI process. Studio does not embed Comfy."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from minimax_studio.worker.runtime import runtime

_PYTHON_RELS = (
    ".venv/bin/python",
    ".venv/Scripts/python.exe",
    "venv/bin/python",
    "venv/Scripts/python.exe",
    "python_embeded/python.exe",
)


def parse_comfy_listen(url: str) -> tuple[str, int]:
    raw = url.strip() or "http://127.0.0.1:8188"
    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    host = parsed.hostname or "127.0.0.1"
    if host == "localhost":
        host = "127.0.0.1"
    port = parsed.port or 8188
    return host, int(port)


def comfy_python(root: Path) -> Path | None:
    for rel in _PYTHON_RELS:
        candidate = root / rel
        if candidate.is_file():
            return candidate
    return None


def guess_comfy_app_roots() -> list[Path]:
    home = Path.home()
    guesses = [
        home / "ai" / "ComfyUI",
        home / "ComfyUI",
        home / "Documents" / "ComfyUI",
    ]
    env = os.environ.get("COMFYUI_PATH")
    if env:
        path = Path(env).expanduser()
        guesses.insert(0, path)
        if path.name == "models":
            guesses.insert(0, path.parent)
    from minimax_studio.worker.model_paths import extra_model_path_files

    for yaml_path in extra_model_path_files():
        if yaml_path.is_file():
            guesses.append(yaml_path.parent)
    cfg = runtime.config
    if cfg.comfy_root:
        guesses.insert(0, Path(cfg.comfy_root).expanduser())
    if cfg.comfy_models_dir:
        models = Path(cfg.comfy_models_dir).expanduser()
        guesses.insert(0, models)
        if models.name == "models":
            guesses.insert(0, models.parent)
    seen: set[str] = set()
    out: list[Path] = []
    for item in guesses:
        try:
            resolved = item.expanduser().resolve()
        except OSError:
            continue
        key = str(resolved)
        if key in seen or not resolved.is_dir():
            continue
        seen.add(key)
        out.append(resolved)
    return out


def build_comfy_argv(
    python: Path,
    root: Path,
    url: str,
    extra: str = "",
) -> list[str]:
    argv = [str(python), str(root / "main.py")]
    extra_parts = shlex.split(extra or "", posix=os.name != "nt")
    flags = {part for part in extra_parts if part.startswith("-")}
    host, port = parse_comfy_listen(url)
    if "--listen" not in flags and "-l" not in flags:
        argv.extend(["--listen", host])
    if "--port" not in flags and "-p" not in flags:
        argv.extend(["--port", str(port)])
    argv.extend(extra_parts)
    return argv


def detect_comfy() -> dict[str, Any]:
    url = runtime.config.comfy_url or "http://127.0.0.1:8188"
    proc_state = _process_state()
    if proc_state["starting"] or proc_state["dead"]:
        from minimax_studio.worker.backends.h3_comfy import reset_comfy_reach_cache

        reset_comfy_reach_cache()
    running = _comfy_running()
    if running:
        proc_state["starting"] = False
        proc_state["dead"] = False
    found = {
        "root": None,
        "python": None,
        "url": url,
        "running": running,
        "starting": bool(proc_state["starting"]) and not running,
        "dead": bool(proc_state["dead"]) and not running,
        "pid": proc_state["pid"],
        "exit_code": proc_state["exit_code"],
        "argv": None,
        "log_tail": _log_tail(),
        "detail": (
            "No ComfyUI install found. Looked for main.py under ~/ai/ComfyUI, "
            "~/ComfyUI, Settings → ComfyUI folder, and COMFYUI_PATH."
        ),
    }
    for root in guess_comfy_app_roots():
        if not (root / "main.py").is_file():
            continue
        python = comfy_python(root)
        extra = runtime.config.comfy_extra_args or ""
        argv = (
            build_comfy_argv(python, root, url, extra) if python else None
        )
        detail = str(root)
        if python is None:
            detail = (
                f"Found ComfyUI at {root} but no venv python "
                "(.venv, venv, or python_embeded)."
            )
        elif running:
            detail = f"ComfyUI is up at {url} ({root})"
        elif found["starting"]:
            detail = f"Starting ComfyUI (pid {found['pid']}) from {root}…"
        elif found["dead"]:
            detail = (
                f"ComfyUI exited ({found['exit_code']}) from {root}. "
                + (found["log_tail"] or "No log.")
            )
        found.update(
            {
                "root": str(root),
                "python": str(python) if python else None,
                "argv": argv,
                "detail": detail,
            }
        )
        return found
    if found["dead"] and found["log_tail"]:
        found["detail"] = (
            f"ComfyUI exited ({found['exit_code']}). " + found["log_tail"]
        )
    return found


def start_comfy() -> dict[str, Any]:
    url = runtime.config.comfy_url or "http://127.0.0.1:8188"
    if _comfy_running():
        return {
            "ok": True,
            "already": True,
            "pid": None,
            "detail": f"ComfyUI is already running at {url}.",
        }
    existing = runtime.comfy_proc
    if existing is not None and existing.poll() is None:
        return {
            "ok": True,
            "already": True,
            "pid": existing.pid,
            "detail": f"ComfyUI start is in flight (pid {existing.pid}).",
        }
    info = detect_comfy()
    root = info.get("root")
    python = info.get("python")
    argv = info.get("argv")
    if not root:
        raise RuntimeError(str(info.get("detail") or "No ComfyUI install found."))
    if not python or not argv:
        raise RuntimeError(str(info.get("detail") or "No ComfyUI venv python."))
    log_handle = _open_comfy_log()
    kwargs: dict[str, Any] = {
        "cwd": root,
        "stdout": log_handle,
        "stderr": subprocess.STDOUT,
    }
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP
        detached = getattr(subprocess, "DETACHED_PROCESS", 0)
        kwargs["creationflags"] = flags | detached
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(argv, **kwargs)
    runtime.comfy_proc = proc
    runtime.comfy_log = log_handle
    from minimax_studio.worker.backends.h3_comfy import reset_comfy_reach_cache

    reset_comfy_reach_cache()
    log_path = str(getattr(log_handle, "name", "") or "")
    return {
        "ok": True,
        "already": False,
        "starting": True,
        "pid": proc.pid,
        "argv": argv,
        "log": log_path,
        "detail": (
            f"Starting ComfyUI (pid {proc.pid}) from {root}. "
            f"Waiting until it answers at {url}."
        ),
    }


def _comfy_running() -> bool:
    from minimax_studio.worker.backends.h3_comfy import comfy_reachable

    return comfy_reachable()


def _process_state() -> dict[str, Any]:
    proc = runtime.comfy_proc
    if proc is None:
        return {"pid": None, "starting": False, "dead": False, "exit_code": None}
    code = proc.poll()
    if code is None:
        return {"pid": proc.pid, "starting": True, "dead": False, "exit_code": None}
    return {"pid": proc.pid, "starting": False, "dead": True, "exit_code": code}


def _log_tail(limit: int = 2000) -> str:
    path = None
    handle = runtime.comfy_log
    name = getattr(handle, "name", None)
    if name:
        path = Path(name)
    elif runtime.config.output_dir:
        path = Path(runtime.config.output_dir) / "comfy-studio.log"
    if path is None or not path.is_file():
        return ""
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    text = data[-limit:].decode("utf-8", errors="replace").strip()
    return text


def _open_comfy_log():
    try:
        if runtime.config.output_dir:
            path = Path(runtime.config.output_dir) / "comfy-studio.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("ab")
            runtime.comfy_log = handle
            return handle
    except OSError:
        pass
    return subprocess.DEVNULL
