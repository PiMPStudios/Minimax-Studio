from __future__ import annotations

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from minimax_studio.ui.state import StudioState
from minimax_studio.worker_client import WorkerClient


class PresetsPage(QWidget):
    def __init__(self, client: WorkerClient, state: StudioState) -> None:
        super().__init__()
        self._client = client
        self._state = state
        self._items: list[dict] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        title = QLabel("Presets")
        title.setObjectName("pageTitle")
        sub = QLabel("Save from Generate Music or Generate Video, apply here.")
        sub.setObjectName("pageSubtitle")
        layout.addWidget(title)
        layout.addWidget(sub)
        self._list = QListWidget()
        layout.addWidget(self._list, 1)
        row = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        apply_btn.setObjectName("primary")
        apply_btn.clicked.connect(self._apply)
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(self._delete)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        row.addWidget(apply_btn)
        row.addWidget(delete_btn)
        row.addWidget(refresh_btn)
        row.addStretch(1)
        layout.addLayout(row)

    def refresh(self) -> None:
        try:
            self._items = self._client.list_presets()
        except Exception:
            self._items = []
        self._list.clear()
        for item in self._items:
            self._list.addItem(
                QListWidgetItem(f"{item.get('name')}  ·  {item.get('kind')}")
            )

    def _current(self) -> dict | None:
        row = self._list.currentRow()
        if row < 0 or row >= len(self._items):
            return None
        return self._items[row]

    def _apply(self) -> None:
        item = self._current()
        if not item:
            return
        if item.get("kind") == "h3":
            self._state.restore_video.emit(item)
        else:
            self._state.restore_music.emit(item)

    def _delete(self) -> None:
        item = self._current()
        if not item:
            return
        try:
            self._client.delete_preset(str(item["id"]))
        except Exception:
            return
        self.refresh()
