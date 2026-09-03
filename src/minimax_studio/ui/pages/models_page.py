from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from minimax_studio.worker_client import WorkerClient


class ModelsPage(QWidget):
    def __init__(self, client: WorkerClient) -> None:
        super().__init__()
        self._client = client
        self._cards: dict[str, _PackCard] = {}
        self._poll_thread = None
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        title = QLabel("Models")
        title.setObjectName("pageTitle")
        self._sub = QLabel(
            "Download only what you need. MiniMax H3 and MiniMax-Music3 weights "
            "stay on disk here — they are not shipped in the app."
        )
        self._sub.setObjectName("pageSubtitle")
        self._sub.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(self._sub)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._host = QWidget()
        self._list = QVBoxLayout(self._host)
        self._list.setSpacing(12)
        self._list.addStretch(1)
        scroll.setWidget(self._host)
        root.addWidget(scroll, 1)
        self.refresh()

    def refresh(self) -> None:
        try:
            packs = self._client.list_packs()
            downloads = {item["pack_id"]: item for item in self._client.list_downloads()}
        except Exception:
            return
        self._apply(packs, downloads)

    def poll(self) -> None:
        """Timer path: pack list off the GUI thread. Overlapping polls skip."""
        from minimax_studio.ui.enhance import start_background

        def work() -> tuple[list, dict]:
            packs = self._client.list_packs()
            downloads = {
                item["pack_id"]: item for item in self._client.list_downloads()
            }
            return packs, downloads

        def done(payload: object) -> None:
            if not isinstance(payload, tuple) or len(payload) != 2:
                return
            packs, downloads = payload
            if isinstance(packs, list) and isinstance(downloads, dict):
                self._apply(packs, downloads)

        start_background(self, work, done, attr="_poll_thread")

    def _apply(self, packs: list[dict[str, Any]], downloads: dict[str, Any]) -> None:
        known = {pack["id"] for pack in packs}
        for pack_id, card in list(self._cards.items()):
            if pack_id not in known:
                card.setParent(None)
                del self._cards[pack_id]
        # Keep stretch at the end.
        if self._list.count():
            stretch = self._list.takeAt(self._list.count() - 1)
        else:
            stretch = None
        for pack in packs:
            card = self._cards.get(pack["id"])
            if card is None:
                card = _PackCard(
                    pack, self._start, self._remove, self._cancel_dl, self._reveal
                )
                self._cards[pack["id"]] = card
                self._list.addWidget(card)
            card.update_pack(pack, downloads.get(pack["id"]))
        if stretch:
            self._list.addItem(stretch)
        else:
            self._list.addStretch(1)
        ready = [pack for pack in packs if pack.get("ready")]
        from_comfy = sum(1 for pack in ready if pack.get("source") == "comfy")
        if ready:
            extra = f" ({from_comfy} from a ComfyUI folder)" if from_comfy else ""
            self._sub.setText(
                f"{len(ready)} pack{'s' if len(ready) != 1 else ''} ready on disk{extra}. "
                "Download only what you still need — weights are never shipped in the app."
            )
        else:
            self._sub.setText(
                "Download only what you need. MiniMax H3 and MiniMax-Music3 weights "
                "stay on disk here — they are not shipped in the app. Comfy-Org INT8 "
                "is the consumer CUDA default."
            )

    def _reveal(self, pack: dict[str, Any]) -> None:
        from pathlib import Path

        from minimax_studio.ui.reveal import reveal_path

        src = pack.get("path")
        if not src or not Path(src).exists():
            QMessageBox.information(self, "Show in folder", "This pack is not on disk yet.")
            return
        try:
            reveal_path(src)
        except OSError as exc:
            QMessageBox.warning(self, "Show in folder failed", str(exc))

    def _start(self, pack: dict[str, Any]) -> None:
        notice = pack.get("territory_notice")
        if notice:
            answer = QMessageBox.question(
                self,
                pack.get("license_name") or "License",
                notice + "\n\nDownload this pack anyway?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        from minimax_studio.ui.download import start_download_or_ask

        if start_download_or_ask(self, self._client, pack["id"]) is None:
            return
        self.refresh()

    def _cancel_dl(self, pack: dict[str, Any], download_id: str | None) -> None:
        if not download_id:
            return
        try:
            self._client.cancel_download(download_id)
        except Exception as exc:
            QMessageBox.warning(self, "Cancel failed", str(exc))
        self.refresh()

    def _remove(self, pack: dict[str, Any]) -> None:
        if pack.get("source") == "comfy":
            QMessageBox.information(
                self,
                "Comfy pack",
                "This pack was found in a ComfyUI models folder. "
                "Studio will not delete files outside its own models directory.",
            )
            return
        answer = QMessageBox.question(
            self,
            "Remove pack",
            f"Delete the Studio copy of “{pack.get('title')}” from disk?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            result = self._client.delete_pack(pack["id"])
        except Exception as exc:
            QMessageBox.warning(self, "Remove failed", str(exc))
            return
        if result.get("folder_kept"):
            freed = float(result.get("removed_bytes") or 0) / (1024**3)
            kept_for = ", ".join(result.get("kept_for") or []) or "other packs"
            answer = QMessageBox.question(
                self,
                "Shared folder kept",
                f"Freed about {freed:.1f} GB. Files still needed by {kept_for} "
                "were kept.\n\nDelete the whole folder anyway? That breaks "
                "those packs too.",
            )
            if answer == QMessageBox.StandardButton.Yes:
                try:
                    self._client.delete_pack(pack["id"], delete_shared=True)
                except Exception as exc:
                    QMessageBox.warning(self, "Remove failed", str(exc))
        self.refresh()


class _PackCard(QFrame):
    def __init__(
        self, pack: dict[str, Any], on_download, on_remove, on_cancel, on_reveal
    ) -> None:
        super().__init__()
        self._pack = pack
        self._on_download = on_download
        self._on_remove = on_remove
        self._on_cancel = on_cancel
        self._on_reveal = on_reveal
        self._download_id: str | None = None
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        self._title = QLabel()
        self._title.setStyleSheet("font-weight: 700; font-size: 15px;")
        self._summary = QLabel()
        self._summary.setWordWrap(True)
        self._summary.setObjectName("pageSubtitle")
        self._meta = QLabel()
        self._meta.setObjectName("pageSubtitle")
        self._meta.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.hide()
        row = QHBoxLayout()
        self._button = QPushButton("Download")
        self._button.setObjectName("primary")
        self._button.clicked.connect(lambda: self._on_download(self._pack))
        self._remove = QPushButton("Remove")
        self._remove.clicked.connect(lambda: self._on_remove(self._pack))
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.hide()
        self._cancel_btn.clicked.connect(
            lambda: self._on_cancel(self._pack, self._download_id)
        )
        self._reveal = QPushButton("Show in folder")
        self._reveal.clicked.connect(lambda: self._on_reveal(self._pack))
        row.addWidget(self._button)
        row.addWidget(self._cancel_btn)
        row.addWidget(self._remove)
        row.addWidget(self._reveal)
        row.addStretch(1)
        layout.addWidget(self._title)
        layout.addWidget(self._summary)
        layout.addWidget(self._meta)
        layout.addWidget(self._bar)
        layout.addLayout(row)

    def update_pack(self, pack: dict[str, Any], download: dict[str, Any] | None) -> None:
        self._pack = pack
        title = pack["title"]
        if pack.get("recommended"):
            title += "  ·  recommended"
        self._title.setText(title)
        self._summary.setText(pack["summary"])
        gb = (pack.get("bytes_on_disk") or 0) / (1024**3)
        status = "Ready" if pack.get("ready") else "Not installed"
        if pack.get("ready") and pack.get("source") == "comfy":
            status = "Ready (ComfyUI models folder)"
        if download and download.get("status") in {"queued", "running", "cancelling"}:
            status = download.get("message") or "Downloading"
            total = download.get("total_bytes") or 0
            done = download.get("bytes") or 0
            pct = int(min(99, (done / total) * 100)) if total else 0
            self._bar.show()
            self._bar.setValue(pct)
            self._button.setEnabled(False)
            self._button.setText("Downloading…")
            self._download_id = str(download.get("id") or "") or None
            cancelling = download.get("status") == "cancelling"
            self._cancel_btn.show()
            self._cancel_btn.setEnabled(not cancelling)
            if cancelling:
                self._cancel_btn.setText("Cancelling…")
            else:
                self._cancel_btn.setText("Cancel")
        elif download and download.get("status") == "error":
            status = f"Error: {download.get('error')}"
            self._bar.hide()
            self._button.setEnabled(True)
            self._button.setText("Retry")
            self._download_id = None
            self._cancel_btn.hide()
        else:
            self._bar.hide()
            self._button.setEnabled(True)
            self._download_id = None
            self._cancel_btn.hide()
            if pack.get("ready"):
                self._button.setText("Re-download")
            elif pack.get("partial") or gb > 0.01:
                self._button.setText("Resume")
            else:
                self._button.setText("Download")
        studio_copy = bool(pack.get("ready")) and pack.get("source") != "comfy"
        self._remove.setEnabled(studio_copy)
        if pack.get("source") == "comfy":
            self._remove.setToolTip("Won’t delete files in your ComfyUI models folder.")
        else:
            self._remove.setToolTip("Delete the Studio copy of this pack.")
        from pathlib import Path

        path = pack.get("path")
        self._reveal.setEnabled(bool(path and Path(path).exists()))
        self._meta.setText(
            f"{status}  ·  ~{pack.get('approx_gb')} GB  ·  {pack.get('license_name')}\n"
            f"{pack.get('path')}"
        )
