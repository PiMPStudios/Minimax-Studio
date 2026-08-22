from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QMessageBox, QWidget

from minimax_studio.worker_client import WorkerClient


def classify_preflight(check: dict[str, Any]) -> str:
    if not check.get("ok"):
        return "block"
    if check.get("warnings"):
        return "warn"
    return "ok"


def confirm_generate(
    parent: QWidget,
    client: WorkerClient,
    kind: str,
    backend: str,
    mode: str = "t2va",
    speed: str = "quality",
) -> bool:
    try:
        check = client.preflight(kind, backend, mode, speed)
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
