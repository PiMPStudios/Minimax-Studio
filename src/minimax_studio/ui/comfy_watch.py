"""Poll GET /comfy until a launched ComfyUI answers or dies."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QWidget

from minimax_studio.worker_client import WorkerClient

OnText = Callable[[str], None]
OnDone = Callable[[bool, dict], None]


def watch_comfy_start(
    parent: QWidget,
    client: WorkerClient,
    on_text: OnText,
    on_done: OnDone | None = None,
    timeout_s: int = 90,
) -> None:
    try:
        result = client.start_comfy()
    except Exception as exc:
        on_text(str(exc))
        if on_done:
            on_done(False, {"detail": str(exc)})
        return
    on_text(str(result.get("detail") or "Starting ComfyUI…"))
    if result.get("already") and not result.get("starting"):
        if on_done:
            on_done(True, result)
        return

    timer = QTimer(parent)
    tries = {"n": 0}

    def tick() -> None:
        tries["n"] += 1
        try:
            info = client.comfy_status()
        except Exception as exc:
            on_text(str(exc))
            timer.stop()
            if on_done:
                on_done(False, {"detail": str(exc)})
            return
        if info.get("running"):
            on_text(str(info.get("detail") or "ComfyUI is up."))
            timer.stop()
            if on_done:
                on_done(True, info)
            return
        if info.get("dead"):
            tail = info.get("log_tail") or info.get("detail") or "ComfyUI exited."
            on_text(str(tail))
            timer.stop()
            if on_done:
                on_done(False, info)
            return
        on_text(f"Starting ComfyUI… ({tries['n']}s)")
        if tries["n"] >= timeout_s:
            tail = info.get("log_tail") or ""
            on_text(
                "Timed out waiting for ComfyUI to answer. "
                + (tail or "Check Settings extra args and comfy-studio.log.")
            )
            timer.stop()
            if on_done:
                on_done(False, info)

    timer.timeout.connect(tick)
    timer.start(1000)
    parent._comfy_watch_timer = timer  # type: ignore[attr-defined]
