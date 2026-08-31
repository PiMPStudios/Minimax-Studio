from __future__ import annotations

import time
from typing import Any

from PySide6.QtWidgets import QInputDialog, QMessageBox, QWidget

from minimax_studio.worker_client import WorkerClient

_LORA_FAMILY_LABELS = ("Music", "H3 (Video)")


def ask_lora_family(parent: QWidget, source_path: str) -> str | None:
    """Music vs H3 for a hand-imported adapter. Cancel returns None.

    Folders named ``h3-comfy`` / ``minimax-h3`` already say H3, so we don't ask.
    """
    from minimax_studio.worker.adapters import kind_from_path

    if kind_from_path(source_path) == "h3":
        return "h3"
    label, ok = QInputDialog.getItem(
        parent,
        "Import LoRA",
        "Which generator is this adapter for?\n"
        "The wrong family queues the wrong generate job (a song vs a video clip).",
        _LORA_FAMILY_LABELS,
        0,
        False,
    )
    if not ok:
        return None
    return "h3" if str(label).startswith("H3") else "music"


def classify_preflight(check: dict[str, Any]) -> str:
    if not check.get("ok"):
        return "block"
    if check.get("warnings"):
        return "warn"
    return "ok"


_PREFLIGHT_TTL_S = 8.0
_preflight_cache: dict[str, Any] = {}
_preflight_cache_at = 0.0


def remember_preflight(
    check: dict[str, Any],
    *,
    speed: str = "quality",
    resolution: str = "768P",
) -> None:
    """Inspector route check is already off-thread; Generate reuses it."""
    global _preflight_cache, _preflight_cache_at
    _preflight_cache = {
        **check,
        "_speed": str(speed),
        "_resolution": str(resolution),
    }
    _preflight_cache_at = time.monotonic()


def _cached_preflight(
    kind: str, backend: str, mode: str, speed: str, resolution: str
) -> dict[str, Any] | None:
    if time.monotonic() - _preflight_cache_at > _PREFLIGHT_TTL_S:
        return None
    check = _preflight_cache
    if (
        str(check.get("kind") or "") == str(kind)
        and str(check.get("requested") or "") == str(backend)
        and str(check.get("mode") or "") == str(mode)
        and str(check.get("_speed") or "") == str(speed)
        and str(check.get("_resolution") or "") == str(resolution)
    ):
        return check
    return None


def confirm_generate(
    parent: QWidget,
    client: WorkerClient,
    kind: str,
    backend: str,
    mode: str = "t2va",
    speed: str = "quality",
    resolution: str = "768P",
) -> bool:
    try:
        check = _cached_preflight(kind, backend, mode, speed, resolution)
        if check is None:
            check = client.preflight(kind, backend, mode, speed, resolution)
            remember_preflight(check, speed=speed, resolution=resolution)
    except Exception as exc:
        QMessageBox.warning(parent, "Worker unreachable", str(exc))
        return False
    verdict = classify_preflight(check)
    if verdict == "ok":
        return True
    if verdict == "warn":
        warnings = check.get("warnings") or []
        text = "\n".join(str(item) for item in warnings)
        answer = QMessageBox.question(
            parent,
            "Generate anyway?",
            text + "\n\nGenerate anyway?",
        )
        return answer == QMessageBox.StandardButton.Yes
    QMessageBox.warning(
        parent,
        "Can't generate yet",
        str(check.get("detail") or "No backend is ready."),
    )
    return False


def notify_job_result(parent: QWidget, job: dict[str, Any], on_retry=None) -> None:
    if job.get("status") != "error":
        return
    text = str(job.get("error") or job.get("message") or "Failed")
    if on_retry is None:
        QMessageBox.warning(parent, "Generate failed", text)
        return
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("Generate failed")
    box.setText(text)
    retry = box.addButton("Retry", QMessageBox.ButtonRole.AcceptRole)
    box.addButton(QMessageBox.StandardButton.Close)
    box.exec()
    if box.clickedButton() is retry:
        on_retry()


def format_queue_line(
    jobs: list[dict[str, Any]], kind: str, current_id: str | None
) -> str:
    queued = [
        item
        for item in jobs
        if item.get("status") == "queued" and item.get("kind") == kind
    ]
    running = [
        item
        for item in jobs
        if item.get("status") in {"running", "cancelling"} and item.get("kind") == kind
    ]
    bits: list[str] = []
    if running and running[0].get("id") != current_id:
        bits.append("another job is running")
    if queued:
        bits.append(f"{len(queued)} queued")
    return " · ".join(bits)
