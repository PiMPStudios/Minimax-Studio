from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from minimax_studio.ui.state import StudioState
from minimax_studio.worker_client import WorkerClient


def _history_detail(entry: dict) -> str:
    bits = [
        f"{entry.get('kind')} / {entry.get('backend') or '?'}",
        str(entry.get("output_path") or ""),
    ]
    meta = []
    if entry.get("mode"):
        meta.append(str(entry["mode"]))
    if entry.get("duration_s"):
        meta.append(f"{entry['duration_s']}s")
    if entry.get("seed") is not None and int(entry.get("seed") or -1) >= 0:
        meta.append(f"seed {entry['seed']}")
    if entry.get("steps"):
        meta.append(f"{entry['steps']} steps")
    if entry.get("speed") and entry.get("speed") != "quality":
        meta.append(str(entry["speed"]))
    if entry.get("ratio"):
        meta.append(str(entry["ratio"]))
    if entry.get("quality"):
        meta.append(str(entry["quality"]))
    if meta:
        bits.append(" · ".join(meta))
    bits.append("")
    bits.append(str(entry.get("prompt") or ""))
    lyrics = str(entry.get("lyrics") or "").strip()
    if lyrics:
        bits.append("")
        bits.append(lyrics)
    return "\n".join(bits)


class HistoryPage(QWidget):
    def __init__(self, client: WorkerClient, state: StudioState) -> None:
        super().__init__()
        self._client = client
        self._state = state
        self._entries: list[dict] = []
        self._visible: list[dict] = []
        self._player = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        title = QLabel("History")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        tools = QHBoxLayout()
        self._kind = QComboBox()
        self._kind.addItems(["All", "Video", "Music"])
        self._kind.currentTextChanged.connect(self._rebuild_list)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Filter prompt…")
        self._search.textChanged.connect(self._rebuild_list)
        tools.addWidget(self._kind)
        tools.addWidget(self._search, 1)
        layout.addLayout(tools)
        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._show)
        layout.addWidget(self._list, 1)
        self._video = None
        try:
            from PySide6.QtMultimediaWidgets import QVideoWidget

            self._video = QVideoWidget()
            self._video.setMinimumHeight(180)
            layout.addWidget(self._video)
        except Exception:
            self._video = None
        self._detail = QLabel("Select a take.")
        self._detail.setWordWrap(True)
        self._detail.setObjectName("pageSubtitle")
        layout.addWidget(self._detail)
        row = QHBoxLayout()
        self._play = QPushButton("Play")
        self._play.clicked.connect(self._play_current)
        restore = QPushButton("Restore to Generate")
        restore.clicked.connect(self._restore_current)
        delete = QPushButton("Delete")
        delete.clicked.connect(self._delete_current)
        export = QPushButton("Export…")
        export.clicked.connect(self._export_current)
        show = QPushButton("Show in folder")
        show.clicked.connect(self._show_in_folder)
        copy = QPushButton("Copy prompt")
        copy.clicked.connect(self._copy_prompt)
        row.addWidget(self._play)
        row.addWidget(restore)
        row.addWidget(export)
        row.addWidget(show)
        row.addWidget(copy)
        row.addWidget(delete)
        row.addStretch(1)
        layout.addLayout(row)
        self._init_player()

    def _init_player(self) -> None:
        try:
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

            self._player = QMediaPlayer(self)
            output = QAudioOutput(self)
            self._player.setAudioOutput(output)
            if self._video is not None:
                self._player.setVideoOutput(self._video)
        except Exception:
            self._player = None
            self._play.setEnabled(False)
            self._play.setToolTip("Qt Multimedia is not available.")

    def refresh(self) -> None:
        try:
            self._entries = self._client.list_history()
        except Exception:
            self._entries = []
        self._rebuild_list()

    def _rebuild_list(self) -> None:
        kind_filter = self._kind.currentText()
        query = self._search.text().strip().lower()
        self._visible = []
        self._list.clear()
        for entry in self._entries:
            kind = entry.get("kind", "?")
            if kind_filter == "Video" and kind != "h3":
                continue
            if kind_filter == "Music" and kind != "music":
                continue
            haystack = f"{entry.get('prompt') or ''} {entry.get('lyrics') or ''}".lower()
            if query and query not in haystack:
                continue
            self._visible.append(entry)
            prompt = (entry.get("prompt") or "")[:80]
            bits = [str(kind)]
            if entry.get("backend"):
                bits.append(str(entry["backend"]))
            if entry.get("duration_s"):
                bits.append(f"{int(entry['duration_s'])}s")
            self._list.addItem(QListWidgetItem(f"{'  ·  '.join(bits)}  ·  {prompt}"))
        if self._list.count() and self._list.currentRow() < 0:
            self._list.setCurrentRow(0)

    def _current(self) -> dict | None:
        row = self._list.currentRow()
        if row < 0 or row >= len(self._visible):
            return None
        return self._visible[row]

    def _show(self, _row: int) -> None:
        entry = self._current()
        if not entry:
            self._detail.setText("Select a take.")
            return
        self._detail.setText(_history_detail(entry))
        path = entry.get("output_path")
        if self._player and path and Path(path).is_file():
            self._player.setSource(QUrl.fromLocalFile(path))

    def _play_current(self) -> None:
        entry = self._current()
        if not entry or not self._player:
            return
        path = entry.get("output_path")
        if not path or not Path(path).is_file():
            return
        self._player.setSource(QUrl.fromLocalFile(path))
        self._player.play()

    def _restore_current(self) -> None:
        entry = self._current()
        if not entry:
            return
        if entry.get("kind") == "music":
            self._state.restore_music.emit(entry)
        else:
            self._state.restore_video.emit(entry)

    def _export_current(self) -> None:
        import shutil

        entry = self._current()
        if not entry:
            return
        src = entry.get("output_path")
        if not src or not Path(src).is_file():
            QMessageBox.information(self, "Export", "This take has no file on disk.")
            return
        suffix = Path(src).suffix or ".bin"
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Export take", Path(src).name, f"Media (*{suffix})"
        )
        if not chosen:
            return
        dest = Path(chosen)
        if dest.suffix.lower() != suffix.lower():
            dest = dest.with_suffix(suffix)
        try:
            shutil.copy2(src, dest)
        except OSError as exc:
            QMessageBox.warning(self, "Export failed", str(exc))

    def _copy_prompt(self) -> None:
        from PySide6.QtGui import QGuiApplication

        entry = self._current()
        if not entry:
            return
        parts = [str(entry.get("prompt") or "").strip()]
        lyrics = str(entry.get("lyrics") or "").strip()
        if lyrics:
            parts.append(lyrics)
        text = "\n\n".join(part for part in parts if part)
        if not text:
            QMessageBox.information(self, "Copy prompt", "This take has no prompt.")
            return
        QGuiApplication.clipboard().setText(text)

    def _show_in_folder(self) -> None:
        from minimax_studio.ui.reveal import reveal_path

        entry = self._current()
        if not entry:
            return
        src = entry.get("output_path")
        if not src or not Path(src).exists():
            QMessageBox.information(self, "Show in folder", "This take has no file on disk.")
            return
        try:
            reveal_path(src)
        except OSError as exc:
            QMessageBox.warning(self, "Show in folder failed", str(exc))

    def _delete_current(self) -> None:
        entry = self._current()
        if not entry:
            return
        if (
            QMessageBox.question(self, "Delete take", "Remove this take from history?")
            != QMessageBox.StandardButton.Yes
        ):
            return
        try:
            self._client.delete_history(str(entry["id"]))
        except Exception as exc:
            QMessageBox.warning(self, "Delete failed", str(exc))
            return
        self.refresh()
