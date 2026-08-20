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
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        title = QLabel("Models")
        title.setObjectName("pageTitle")
        sub = QLabel(
            "Download only what you need. MiniMax H3 and MiniMax-Music3 weights "
            "stay on disk here — they are not shipped in the app."
        )
        sub.setObjectName("pageSubtitle")
        sub.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(sub)
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
        except Exception as exc:
            return
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
                card = _PackCard(pack, self._start)
                self._cards[pack["id"]] = card
                self._list.addWidget(card)
            card.update_pack(pack, downloads.get(pack["id"]))
        if stretch:
            self._list.addItem(stretch)
        else:
            self._list.addStretch(1)

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
        try:
            self._client.start_download(pack["id"])
        except Exception as exc:
            QMessageBox.warning(self, "Download failed", str(exc))
            return
        self.refresh()


class _PackCard(QFrame):
    def __init__(self, pack: dict[str, Any], on_download) -> None:
        super().__init__()
        self._pack = pack
        self._on_download = on_download
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
        row.addWidget(self._button)
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
        if download and download.get("status") in {"queued", "running"}:
            status = download.get("message") or "Downloading"
            total = download.get("total_bytes") or 0
            done = download.get("bytes") or 0
            pct = int(min(99, (done / total) * 100)) if total else 0
            self._bar.show()
            self._bar.setValue(pct)
            self._button.setEnabled(False)
            self._button.setText("Downloading…")
        elif download and download.get("status") == "error":
            status = f"Error: {download.get('error')}"
            self._bar.hide()
            self._button.setEnabled(True)
            self._button.setText("Retry")
        else:
            self._bar.hide()
            self._button.setEnabled(True)
            if pack.get("ready"):
                self._button.setText("Re-download")
            elif pack.get("partial") or gb > 0.01:
                self._button.setText("Resume")
            else:
                self._button.setText("Download")
        self._meta.setText(
            f"{status}  ·  ~{pack.get('approx_gb')} GB  ·  {pack.get('license_name')}\n"
            f"{pack.get('path')}"
        )
