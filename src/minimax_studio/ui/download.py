"""Start a pack download; turn InsufficientDisk into Download anyway?"""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QMessageBox, QWidget

from minimax_studio.errors import InsufficientDisk
from minimax_studio.worker_client import WorkerClient


def start_download_or_ask(
    parent: QWidget, client: WorkerClient, pack_id: str
) -> dict[str, Any] | None:
    """Return the download job, or None if the user cancelled / it failed.

    Catches :class:`InsufficientDisk` by type so rewording the worker message
    cannot silently drop the “Download anyway?” hatch.
    """
    try:
        return client.start_download(pack_id)
    except InsufficientDisk as exc:
        answer = QMessageBox.question(
            parent,
            "Low disk space",
            f"{exc}\n\nDownload anyway?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return None
        try:
            return client.start_download(pack_id, force=True)
        except Exception as exc2:
            QMessageBox.warning(parent, "Download failed", str(exc2))
            return None
    except Exception as exc:
        QMessageBox.warning(parent, "Download failed", str(exc))
        return None
