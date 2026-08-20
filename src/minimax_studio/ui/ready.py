from __future__ import annotations

from PySide6.QtWidgets import QMessageBox, QWidget

from minimax_studio.worker_client import WorkerClient


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
    if check.get("ok"):
        return True
    QMessageBox.warning(
        parent,
        "Can't generate yet",
        str(check.get("detail") or "No backend is ready."),
    )
    return False
