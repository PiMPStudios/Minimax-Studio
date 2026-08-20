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
) -> bool:
    try:
        check = client.preflight(kind, backend, mode)
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


def notify_job_result(parent: QWidget, job: dict[str, Any]) -> None:
    if job.get("status") != "error":
        return
    QMessageBox.warning(
        parent,
        "Generate failed",
        str(job.get("error") or job.get("message") or "Failed"),
    )
