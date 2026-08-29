"""Datasets — the S1 half of the Build pages (PLAN-V2).

SimpleTuner reads plain folders natively: ``track.wav`` + ``track.txt`` caption
(+ optional ``track.lyrics``). Studio owns every write, and imports **copy** —
cleaning up the originals must never gut a dataset. This page's whole job is
honesty at clip level: ✓ per clip that can train, and the specific reason for
the ones that can't, before anyone burns an hour of VRAM finding out.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from minimax_studio.ui.reveal import reveal_path
from minimax_studio.worker_client import WorkerClient

# Mirrors worker/datasets.py MEDIA_BY_KIND; used only to filter the History
# picker, so a Music dataset never offers you an mp4 to add.
MEDIA_BY_KIND = {
    "music": (".wav", ".flac", ".mp3"),
    "video": (".mp4", ".mov", ".webm"),
}
_OK = QBrush(QColor("#32d74b"))
_BAD = QBrush(QColor("#ff453a"))


def validation_status_short(row: dict[str, Any]) -> str:
    """One line of truth about a dataset's last validation, in the list."""
    last = row.get("last_validation")
    if not last:
        return "not checked"
    if last.get("ok"):
        return "ready"
    return f"{last.get('with_problems', '?')} problems"


def _detail_text(row: dict[str, Any], detail: dict[str, Any]) -> str:
    folder = Path(str(detail.get("path") or ""))
    # Labels here are rich text, and dataset names are user input.
    name = html.escape(str(row.get("name")))
    notes = html.escape(str(row.get("notes") or "").strip())
    lines = [
        f"<b>{name}</b>  ·  {html.escape(str(row.get('kind')))} dataset",
        f"{row.get('clip_count', 0)} clips on disk  · {validation_status_short(row)}",
        f"Folder: {html.escape(str(folder))}",
    ]
    if notes:
        lines.append(notes)
    report = detail.get("validation") or {}
    if not report:
        lines.append("")
        lines.append(
            "Never validated. Press <b>Validate</b> — training refuses a "
            "dataset that does not check out, so this is the cheap moment to "
            "find a missing caption."
        )
    return "\n".join(lines).replace("\n", "<br>")


class DatasetsPage(QWidget):
    """List, import, validate. Training lives in :mod:`train_page`."""

    #: asks the window to open the Train page with this dataset selected
    train_requested = Signal(str)

    def __init__(self, client: WorkerClient) -> None:
        super().__init__()
        self._client = client
        self._rows: list[dict[str, Any]] = []
        self._detail: dict[str, Any] = {}
        self._selected: str | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        title = QLabel("Datasets")
        title.setObjectName("pageTitle")
        subtitle = QLabel(
            "A dataset is a plain folder — <code>track.wav</code> plus a "
            "<code>track.txt</code> caption, and optionally "
            "<code>track.lyrics</code>. Imports <b>copy</b> files in, so "
            "cleaning up your originals never guts a dataset. Music only for "
            "now; H3 training is the next slice. "
            "<span style='color:#ff9f0a'>Experimental.</span>"
        )
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)
        subtitle.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(title)
        root.addWidget(subtitle)

        tools = QHBoxLayout()
        self._new_btn = QPushButton("New dataset…")
        self._new_btn.setObjectName("primary")
        self._new_btn.clicked.connect(self._create)
        self._import_btn = QPushButton("Import folder…")
        self._import_btn.clicked.connect(self._import_folder)
        self._from_history_btn = QPushButton("Add from History…")
        self._from_history_btn.clicked.connect(self._add_from_history)
        self._validate_btn = QPushButton("Validate")
        self._validate_btn.clicked.connect(self._validate)
        self._reveal_btn = QPushButton("Show in folder")
        self._reveal_btn.clicked.connect(self._reveal_folder)
        self._train_btn = QPushButton("Train with this →")
        self._train_btn.clicked.connect(
            lambda: self._selected and self.train_requested.emit(self._selected)
        )
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.clicked.connect(self._delete)
        for button in (
            self._new_btn,
            self._import_btn,
            self._from_history_btn,
            self._validate_btn,
            self._train_btn,
            self._reveal_btn,
            self._delete_btn,
        ):
            tools.addWidget(button)
        tools.addStretch(1)
        root.addLayout(tools)

        self._status = QLabel("No datasets yet — make one to start.")
        self._status.setObjectName("pageSubtitle")
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        split = QSplitter()
        self._list = QListWidget()
        self._list.setFixedWidth(260)
        self._list.currentRowChanged.connect(self._row_changed)
        split.addWidget(self._list)
        right = QWidget()
        right_row = QVBoxLayout(right)
        right_row.setContentsMargins(12, 0, 0, 0)
        self._detail_label = QLabel("Select a dataset.")
        self._detail_label.setObjectName("pageSubtitle")
        self._detail_label.setWordWrap(True)
        self._detail_label.setTextFormat(Qt.TextFormat.RichText)
        self._detail_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        right_row.addWidget(self._detail_label)
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Clip", "Status"])
        self._tree.setRootIsDecorated(False)
        self._tree.setAlternatingRowColors(True)
        self._tree.header().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self._tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        right_row.addWidget(self._tree, 1)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        root.addWidget(split, 1)
        self._set_buttons_enabled(bool(self._rows))
        self.refresh()

    # --- data ---------------------------------------------------------------

    def refresh(self) -> None:
        try:
            self._rows = self._client.list_datasets()
        except Exception:
            return
        wanted = self._selected
        self._list.blockSignals(True)
        self._list.clear()
        row_for_id: dict[str, int] = {}
        for row in self._rows:
            dataset_id = str(row.get("id"))
            item = QListWidgetItem(
                f"{row.get('name')}\n{row.get('clip_count', 0)} clips · "
                f"{validation_status_short(row)}"
            )
            item.setData(Qt.ItemDataRole.UserRole, dataset_id)
            self._list.addItem(item)
            row_for_id[dataset_id] = self._list.count() - 1
        self._list.blockSignals(False)
        if wanted and wanted in row_for_id:
            self._list.setCurrentRow(row_for_id[wanted])
        elif self._list.count():
            self._list.setCurrentRow(0)
        else:
            self._selected = None
            self._show()
        self._set_buttons_enabled(bool(self._rows))

    def select(self, dataset_id: str) -> None:
        for index in range(self._list.count()):
            if self._list.item(index).data(Qt.ItemDataRole.UserRole) == dataset_id:
                self._list.setCurrentRow(index)
                return

    def _set_buttons_enabled(self, have_selection: bool) -> None:
        for button in (
            self._import_btn,
            self._from_history_btn,
            self._validate_btn,
            self._train_btn,
            self._reveal_btn,
            self._delete_btn,
        ):
            button.setEnabled(have_selection)
        row = self._current_row()
        # The H3 trainer lands in S4; a video dataset is a place to put clips
        # today, but "Train with this" would only ever fail.
        self._train_btn.setEnabled(bool(row) and row.get("kind") == "music")
        if row and row.get("kind") != "music":
            self._train_btn.setToolTip(
                "H3 training arrives in PLAN-V2 S4 — Music LoRAs only for now."
            )
        else:
            self._train_btn.setToolTip("")

    def _current_row(self) -> dict[str, Any] | None:
        if not self._selected:
            return None
        return next((row for row in self._rows if row.get("id") == self._selected), None)

    def _row_changed(self, row: int) -> None:
        item = self._list.item(row) if row >= 0 else None
        self._selected = item.data(Qt.ItemDataRole.UserRole) if item else None
        self._show()

    def _show(self) -> None:
        row = self._current_row()
        self._tree.clear()
        self._set_buttons_enabled(bool(row))
        if not row:
            self._detail_label.setText("Select a dataset.")
            self._status.setText("No datasets yet — make one to start.")
            return
        try:
            self._detail = self._client.get_dataset(str(row["id"]))
        except Exception as exc:
            self._detail = {}
            self._detail_label.setText(f"Could not read this dataset: {exc}")
            return
        self._detail_label.setText(_detail_text(row, self._detail))
        report = self._detail.get("validation") or {}
        rows = sorted(
            report.get("rows", []), key=lambda item: bool(item.get("ok"))
        )
        for entry in rows:
            problems = entry.get("problems") or []
            item = QTreeWidgetItem(
                [
                    str(entry.get("file")),
                    "✓ ready" if entry.get("ok") else "✗ " + "; ".join(problems),
                ]
            )
            brush = _OK if entry.get("ok") else _BAD
            item.setForeground(1, brush)
            if problems:
                item.setToolTip(1, "\n".join(problems))
            self._tree.addTopLevelItem(item)
        self._status.setText(self._status_line(row, report))

    def _status_line(self, row: dict[str, Any], report: dict[str, Any]) -> str:
        clips = int(row.get("clip_count") or 0)
        if not report:
            return (
                f"{clips} clip{'s' if clips != 1 else ''}, never validated. "
                "Validate before training — it is free, the GPU is not."
            )
        bad = [entry for entry in report.get("rows", []) if not entry.get("ok")]
        if report.get("ok"):
            return (
                f"Validated {report.get('checked', 0)} clips — all good. "
                "Ready to train."
            )
        first = (bad[0]["problems"][0] if bad and bad[0].get("problems") else "see below")
        return (
            f"{len(bad)} of {max(report.get('checked', 0), len(bad))} entries "
            f"have problems — first: {first}. Training will refuse this "
            "dataset until they are fixed."
        )

    # --- actions ------------------------------------------------------------

    def _create(self) -> None:
        dialog = _NewDatasetDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            created = self._client.create_dataset(dialog.dataset_name(), dialog.dataset_kind(), dialog.dataset_notes())
        except Exception as exc:
            QMessageBox.warning(self, "Could not create dataset", str(exc))
            return
        self._selected = str(created.get("id"))
        self.refresh()
        self._status.setText(
            f"Created “{created.get('name')}” at {created.get('path')}. "
            "Import clips or add them from History, then Validate."
        )

    def _import_folder(self) -> None:
        row = self._current_row()
        if not row:
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Import clips — files are copied, not linked", str(Path.home())
        )
        if not folder:
            return
        try:
            result = self._client.import_dataset_folder(str(row["id"]), folder)
        except Exception as exc:
            QMessageBox.warning(self, "Import failed", str(exc))
            return
        copied = len(result.get("copied") or [])
        captions = int(result.get("captions") or 0)
        self.refresh()
        self._validate(quiet=True)
        missing = copied - captions
        note = (
            f" — {captions} brought a caption with them"
            if captions
            else " — none carried a .txt caption, so write one per clip "
            "(SimpleTuner reads them verbatim)"
        )
        self._status.setText(
            f"Copied {copied} clip{'s' if copied != 1 else ''} from "
            f"{Path(folder).name}{note}."
            + (" Missing captions are listed below." if missing > captions else "")
        )

    def _add_from_history(self) -> None:
        row = self._current_row()
        if not row:
            return
        try:
            entries = [
                entry
                for entry in self._client.list_history()
                if Path(str(entry.get("output_path") or "")).suffix.lower()
                in MEDIA_BY_KIND.get(str(row.get("kind") or "music"), ())
            ]
        except Exception as exc:
            QMessageBox.warning(self, "History unavailable", str(exc))
            return
        if not entries:
            QMessageBox.information(
                self,
                "Nothing to add yet",
                f"No {'music' if row.get('kind') == 'music' else 'video'} "
                "generations are in History yet. Generate a few, keep the "
                "good ones, then add them here.",
            )
            return
        picker = _HistoryPicker(entries, str(row.get("kind") or "music"), self)
        if picker.exec() != QDialog.DialogCode.Accepted or not picker.selected_entry():
            return
        try:
            added = self._client.add_dataset_from_history(
                str(row["id"]), str(picker.selected_entry().get("id"))
            )
        except Exception as exc:
            QMessageBox.warning(self, "Could not add that take", str(exc))
            return
        self.refresh()
        self._validate(quiet=True)
        self._status.setText(
            f"Added {added.get('added')} from History, with its caption and "
            "lyrics carried over."
        )

    def _validate(self, quiet: bool = False) -> None:
        row = self._current_row()
        if not row:
            return
        try:
            report = self._client.validate_dataset(str(row["id"]))
        except Exception as exc:
            QMessageBox.warning(self, "Validation failed", str(exc))
            return
        self.refresh()
        if not quiet and not report.get("ok"):
            bad = [entry for entry in report.get("rows", []) if not entry.get("ok")]
            QMessageBox.warning(
                self,
                "Not ready to train",
                "\n".join(
                    f"{entry.get('file')}: {problem}"
                    for entry in bad[:8]
                    for problem in (entry.get("problems") or [])
                )[:2000]
                or "Nothing checked.",
            )

    def _reveal_folder(self) -> None:
        path = str(self._detail.get("path") or "")
        if not path or not Path(path).is_dir():
            QMessageBox.information(
                self, "Show in folder", "This dataset folder is not on disk."
            )
            return
        try:
            reveal_path(path)
        except OSError as exc:
            QMessageBox.warning(self, "Show in folder failed", str(exc))

    def _delete(self) -> None:
        row = self._current_row()
        if not row:
            return
        answer = QMessageBox.question(
            self,
            "Delete dataset",
            f"Delete “{row.get('name')}” and every clip copied into it?\n\n"
            "Your original files and History are untouched — only the copies "
            f"in {self._detail.get('path')} go.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._client.delete_dataset(str(row["id"]))
        except Exception as exc:
            QMessageBox.warning(self, "Delete failed", str(exc))
            return
        self._selected = None
        self.refresh()


class _NewDatasetDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New dataset")
        layout = QFormLayout(self)
        self._name = QLineEdit()
        self._name.setPlaceholderText("summer sessions")
        self._kind = QComboBox()
        self._kind.addItem("Music — trains a LoRA today", "music")
        self._kind.addItem("Video (H3) — collection only until S4", "video")
        self._notes = QLineEdit()
        self._notes.setPlaceholderText("what this one is for")
        layout.addRow("Name", self._name)
        layout.addRow("Kind", self._kind)
        layout.addRow("Notes", self._notes)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def dataset_name(self) -> str:
        return self._name.text().strip() or "untitled"

    def dataset_kind(self) -> str:
        return str(self._kind.currentData() or "music")

    def dataset_notes(self) -> str:
        return self._notes.text().strip()


class _HistoryPicker(QDialog):
    """Pick one finished generation to feed a dataset. Caption and lyrics
    ride along — the reason History-to-dataset is nearly free."""

    def __init__(
        self, entries: list[dict[str, Any]], kind: str, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add a generation to this dataset")
        self._entries = entries
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                f"Your {kind} generations. The prompt becomes the caption and "
                "the lyrics are copied too, so the clip arrives labelled."
            )
        )
        self._list = QListWidget()
        for entry in entries:
            prompt = str(entry.get("prompt") or "").strip() or "(no prompt)"
            name = Path(str(entry.get("output_path") or "?")).name
            self._list.addItem(
                QListWidgetItem(f"{name}\n{prompt[:90]}")
            )
        if entries:
            self._list.setCurrentRow(0)
        layout.addWidget(self._list, 1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_entry(self) -> dict[str, Any] | None:
        row = self._list.currentRow()
        if row < 0 or row >= len(self._entries):
            return None
        return self._entries[row]
