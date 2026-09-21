"""The one file that lets a process outside Studio find the running worker.

Studio keeps an ephemeral port and an in-memory token on purpose (`app.py`), so
that no other local process can queue jobs or read keys out of ``/settings``.
This file is a deliberate, *switchable* exception to that design — written only
while Settings → "Allow agent access" is on, and deleted the moment the switch
goes off or the worker shuts down. Off means no file exists at all, so a stale
handoff cannot outlive the intent that created it.

It carries the per-launch worker token, so it is created 0600 *at creation
time* (a chmod after the write is a window), replaced atomically, and kept next
to ``config.json`` — the directory that already holds HF / MiniMax keys in
plaintext unless the optional ``keyring`` extra is switched on. It adds no
secret that is not already on disk; it advertises where to use one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from minimax_studio import __version__
from minimax_studio.config import default_config_path

HANDOFF_NAME = "agent-handoff.json"
SERVICE = "minimax-studio-worker"
# The GUI exports these two into the worker child. The URL is the address the
# GUI itself is talking to, so the handoff can never advertise something the
# GUI could not reach.
URL_ENV = "MINIMAX_STUDIO_WORKER_URL"
TOKEN_ENV = "MINIMAX_STUDIO_WORKER_TOKEN"


def handoff_path(config_path: Path | None = None) -> Path:
    return (config_path or default_config_path()).parent / HANDOFF_NAME


def write(payload: dict[str, Any], config_path: Path | None = None) -> Path:
    path = handoff_path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def clear(config_path: Path | None = None) -> None:
    """Idempotent. A worker mid-restart can hold the file open on Windows."""
    try:
        handoff_path(config_path).unlink(missing_ok=True)
    except OSError:
        pass


def read(config_path: Path | None = None) -> dict[str, Any] | None:
    """Shape-checked only.

    Liveness is the caller's problem: a file cannot know whether the worker
    that wrote it is still running, and on a dead worker's port anything may
    be listening. Callers prove it by asking — GET /health with the token.
    """
    try:
        data = json.loads(handoff_path(config_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("service") != SERVICE or not data.get("url"):
        return None
    if "token" not in data:
        return None
    return data


def sync(
    enabled: bool,
    url: str | None = None,
    token: str | None = None,
    config_path: Path | None = None,
) -> Path | None:
    """Write the handoff, or remove it. The only-while-on rule lives here.

    Both the worker's startup hook and its settings save call this, so the
    switch cannot be applied in one path and forgotten in the other.

    A tokenless worker — ``--worker-only`` dev mode — gets no file even if the
    switch is on: that listener is already open to every local process, and
    advertising it would make a dev run look like a normal Studio install to
    whatever agent is reading this.
    """
    resolved_url = url or os.environ.get(URL_ENV, "")
    resolved_token = token if token is not None else os.environ.get(TOKEN_ENV, "")
    if not enabled or not resolved_url or not resolved_token:
        clear(config_path)
        return None
    return write(
        {
            "service": SERVICE,
            "url": resolved_url,
            "token": resolved_token,
            "pid": os.getpid(),
            "worker_version": __version__,
        },
        config_path,
    )
