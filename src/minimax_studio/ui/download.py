"""License notice, then start a pack download; InsufficientDisk → Download anyway?"""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QMessageBox, QWidget

from minimax_studio.errors import InsufficientDisk
from minimax_studio.worker_client import WorkerClient


def confirm_and_download(
    parent: QWidget,
    client: WorkerClient,
    pack: dict[str, Any],
    *,
    noun: str = "pack",
) -> dict[str, Any] | None:
    """License notice → download → disk-guard retry.

    Returns the download job, or None if the person backed out / it failed.
    Models and Adapters button handlers are the callers — they differ only
    in the noun (“pack” vs “adapter”) and what they do with the job.
    """
    notice = pack.get("territory_notice")
    if notice:
        answer = QMessageBox.question(
            parent,
            str(pack.get("license_name") or "License"),
            f"{notice}\n\nDownload this {noun} anyway?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return None
    pack_id = str(pack.get("id") or "")
    if not pack_id:
        return None
    return start_download_or_ask(parent, client, pack_id)


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
