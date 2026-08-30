"""Adapters — provenance and the audition loop (PLAN-V2 S3).

Every ``.safetensors`` the app can load is listed here, whether Studio trained
it, you imported it, or it was simply already in a LoRA folder — ``trained`` /
``imported`` / ``untracked`` say which, and the untracked ones are the honest
part: the picker has always loaded them, so pretending not to know would be a
lie by omission.

The button that closes the loop is **Audition**: one 30-second render at 0.8
strength with the caption the adapter was actually trained on, queued as an
ordinary job and badged in History. That is what 3 hours of GPU was for, in
your ears, without setting up anything.
"""

from __future__ import annotations

import html
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from minimax_studio.ui.reveal import reveal_path
from minimax_studio.worker_client import WorkerClient

_OK = QBrush(QColor("#32d74b"))
_WARN = QBrush(QColor("#ff9f0a"))
_BAD = QBrush(QColor("#ff453a"))
_DIM = QBrush(QColor("#8e8e93"))

SOURCE_LABEL = {
    "trained": "trained here",
    "imported": "imported",
    "untracked": "found on disk",
}


def _when(timestamp: Any) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(timestamp)))
    except (TypeError, ValueError):
        return "—"


def _detail_text(row: dict[str, Any]) -> str:
    """Everything a filename hides. Blank fields are left out rather than
    shown as “None”, which is how inventories start lying."""
    lines = [f"<b>{html.escape(str(row.get('name')))}</b>"]
    bits = [SOURCE_LABEL.get(str(row.get("source")), str(row.get("source") or "?"))]
    if row.get("created_at"):
        bits.append(_when(row.get("created_at")))
    if not row.get("on_disk"):
        bits.append("<span style='color:#ff453a'>file is gone</span>")
    lines.append("  ·  ".join(bits))

    dataset = row.get("dataset") or {}
    if dataset.get("path"):
        lines.append("")
        lines.append(
            f"<b>Trained on</b> {html.escape(str(Path(str(dataset.get('path'))).name))}"
            f" — {dataset.get('clip_count', 0)} clips"
            + (
                f" · manifest {dataset.get('manifest_hash')}"
                if dataset.get("manifest_hash")
                else ""
            )
        )
        if not row.get("dataset_exists"):
            lines.append(
                "<span style='color:#ff9f0a'>That dataset folder is gone — the "
                "provenance still reads, but an audition needs a prompt typed "
                "in.</span>"
            )
        else:
            lines.append(
                f"<span style='color:#8e8e93'>{html.escape(str(dataset.get('path')))}</span>"
            )
    for label, key in (
        ("Run", "run_name"),
        ("Trainer", "trainer"),
        ("Base pack", "base_pack"),
        ("Preset", "preset"),
        ("Rank", "rank"),
        ("Steps", "steps"),
    ):
        value = row.get(key)
        if value not in (None, ""):
            lines.append(f"<b>{label}:</b> {html.escape(str(value))}")
    if row.get("path"):
        lines.append(
            f"<span style='color:#8e8e93'>{html.escape(str(row.get('path')))}</span>"
        )
    return "<br>".join(lines)


class AdaptersPage(QWidget):
    def __init__(self, client: WorkerClient) -> None:
        super().__init__()
        self._client = client
        self._rows: list[dict[str, Any]] = []
        self._visible: list[dict[str, Any]] = []
        self._selected: str | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        title = QLabel("Adapters")
        title.setObjectName("pageTitle")
        subtitle = QLabel(
            "Every LoRA the picker can load, with where it came from. "
            "<b>Audition</b> queues a short render at 0.8 strength using the "
            "caption the adapter was trained on — 30 seconds that tell you "
            "whether the run was worth it."
        )
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Adapter", "Source", "Kind", "Trained on", "When"])
        self._tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._tree.currentItemChanged.connect(lambda *_: self._row_selected())
        root.addWidget(self._tree, 3)

        self._detail = QLabel("Select an adapter.")
        self._detail.setObjectName("pageSubtitle")
        self._detail.setWordWrap(True)
        self._detail.setTextFormat(Qt.TextFormat.RichText)
        self._detail.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        root.addWidget(self._detail)

        row = QHBoxLayout()
        self._prompt = QLineEdit()
        self._prompt.setPlaceholderText("Audition prompt — blank reuses the dataset caption")
        self._duration = QSpinBox()
        self._duration.setRange(5, 120)
        self._duration.setValue(30)
        self._duration.setSuffix(" s")
        self._duration.setToolTip(
            "Auditions are short on purpose — you are checking for the voice, "
            "not mastering a track."
        )
        self._audition_btn = QPushButton("Audition")
        self._audition_btn.setObjectName("primary")
        self._audition_btn.clicked.connect(self._audition)
        self._import_btn = QPushButton("Import .safetensors…")
        self._import_btn.clicked.connect(self._import)
        self._reveal_btn = QPushButton("Show in folder")
        self._reveal_btn.clicked.connect(self._reveal)
        self._forget_btn = QPushButton("Forget")
        self._forget_btn.setToolTip(
            "Remove Studio's provenance row for this file. The .safetensors "
            "itself is left alone."
        )
        self._forget_btn.clicked.connect(self._forget)
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self.refresh)
        row.addWidget(self._prompt, 1)
        row.addWidget(self._duration)
        row.addWidget(self._audition_btn)
        row.addWidget(self._import_btn)
        row.addWidget(self._reveal_btn)
        row.addWidget(self._forget_btn)
        row.addWidget(self._refresh_btn)
        root.addLayout(row)

        filters = QHBoxLayout()
        self._mine_only = QCheckBox("Only what Studio trained")
        self._mine_only.toggled.connect(self._rebuild_rows)
        self._missing_only = QCheckBox("Only missing files")
        self._missing_only.setToolTip(
            "Registry rows whose .safetensors was deleted from disk."
        )
        self._missing_only.toggled.connect(self._rebuild_rows)
        filters.addWidget(self._mine_only)
        filters.addWidget(self._missing_only)
        filters.addStretch(1)
        root.addLayout(filters)

        self._status = QLabel("")
        self._status.setObjectName("pageSubtitle")
        self._status.setWordWrap(True)
        root.addWidget(self._status)
        self._set_buttons_enabled(False)
        self.refresh()

    # --- data ---------------------------------------------------------------

    def refresh(self) -> None:
        try:
            self._rows = self._client.list_adapters()
        except Exception:
            return
        self._rebuild_rows()

    def _rebuild_rows(self) -> None:
        wanted = self._selected
        self._visible = [
            row
            for row in self._rows
            if (not self._mine_only.isChecked() or row.get("source") == "trained")
            and (not self._missing_only.isChecked() or not row.get("on_disk"))
        ]
        self._tree.blockSignals(True)
        self._tree.clear()
        for row in self._visible:
            dataset = row.get("dataset") or {}
            trained_on = (
                f"{dataset.get('clip_count', 0)} clips"
                if dataset.get("path")
                else "—"
            )
            if dataset.get("manifest_hash"):
                trained_on += f" · {dataset['manifest_hash']}"
            item = QTreeWidgetItem(
                [
                    str(row.get("name")),
                    SOURCE_LABEL.get(str(row.get("source")), str(row.get("source") or "?")),
                    str(row.get("kind") or "?"),
                    trained_on,
                    _when(row.get("created_at")),
                ]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, str(row.get("id")))
            source = str(row.get("source"))
            if not row.get("on_disk"):
                item.setForeground(0, _BAD)
            elif source == "trained":
                item.setForeground(1, _OK)
            elif source == "imported":
                item.setForeground(1, _WARN)
            else:
                item.setForeground(1, _DIM)
            self._tree.addTopLevelItem(item)
        self._tree.blockSignals(False)
        restore = next(
            (
                index
                for index in range(self._tree.topLevelItemCount())
                if self._tree.topLevelItem(index).data(0, Qt.ItemDataRole.UserRole)
                == wanted
            ),
            None,
        )
        if restore is not None:
            self._tree.setCurrentItem(self._tree.topLevelItem(restore))
        elif self._tree.topLevelItemCount():
            self._tree.setCurrentItem(self._tree.topLevelItem(0))
        else:
            self._selected = None
            self._show()

    def _current(self) -> dict[str, Any] | None:
        return next(
            (row for row in self._visible if row.get("id") == self._selected), None
        )

    def _row_selected(self) -> None:
        item = self._tree.currentItem()
        self._selected = item.data(0, Qt.ItemDataRole.UserRole) if item else None
        self._show()

    def _show(self) -> None:
        row = self._current()
        self._set_buttons_enabled(bool(row))
        if not row:
            self._detail.setText(
                "Nothing here yet. Train a LoRA and <b>Install adapter</b>, or "
                "import a .safetensors — either way it shows up in the picker "
                "on Generate Music too."
            )
            self._prompt.setPlaceholderText("Audition prompt")
            return
        self._detail.setText(_detail_text(row))
        caption = str(row.get("audition_prompt") or "")
        self._prompt.setPlaceholderText(
            (
                f"Audition prompt — blank reuses “{caption[:70]}”"
                if caption
                else "No dataset caption behind this one — type an audition prompt"
            )
        )

    def _set_buttons_enabled(self, have_selection: bool) -> None:
        row = self._current()
        for button in (self._reveal_btn, self._forget_btn):
            button.setEnabled(have_selection)
        self._audition_btn.setEnabled(bool(row and row.get("can_audition")))
        if row and not row.get("can_audition"):
            self._audition_btn.setToolTip(
                "One-click audition renders a clip: Music adapters only. An H3 "
                "adapter has no preview wired up yet — load it on Generate Video."
                if row.get("kind") != "music"
                else "The file is not on disk — reinstall it from its run."
            )
        else:
            self._audition_btn.setToolTip(
                "One short generate job at 0.8 strength, badged in History."
            )

    # --- actions ------------------------------------------------------------

    def _audition(self) -> None:
        row = self._current()
        if not row:
            return
        try:
            queued = self._client.audition_adapter(
                str(row["id"]),
                prompt=self._prompt.text().strip(),
                duration_s=float(self._duration.value()),
            )
        except Exception as exc:
            QMessageBox.warning(self, "Audition did not start", str(exc))
            return
        self._prompt.clear()
        self._status.setText(
            f"Audition queued for “{row.get('name')}” — "
            f"{int(queued.get('duration_s') or 0)} s at "
            f"{float(queued.get('strength') or 0):.1f} strength, job "
            f"{queued.get('job_id')}. It lands in History badged as an "
            "audition, and Restore to Generate works on it like any take."
        )

    def _import(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Import a LoRA",
            str(Path.home()),
            "LoRA files (*.safetensors)",
        )
        if not path:
            return
        try:
            row = self._client.import_lora(path)
        except Exception as exc:
            QMessageBox.warning(self, "Import failed", str(exc))
            return
        self._status.setText(
            f"Imported “{row.get('name')}”. It is listed as imported — Studio "
            "knows it did not train it, and the picker has it."
        )
        self.refresh()

    def _reveal(self) -> None:
        row = self._current()
        path = str((row or {}).get("path") or "")
        if not path or not Path(path).is_file():
            QMessageBox.information(
                self, "Show in folder", "This adapter file is not on disk."
            )
            return
        try:
            reveal_path(path)
        except OSError as exc:
            QMessageBox.warning(self, "Show in folder failed", str(exc))

    def _forget(self) -> None:
        row = self._current()
        if not row:
            return
        answer = QMessageBox.question(
            self,
            "Forget adapter",
            f"Forget Studio’s record of “{row.get('name')}”?\n\n"
            "The .safetensors stays on disk and stays in the LoRA picker — it "
            "just comes back as “found on disk”, with no provenance.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._client.forget_adapter(str(row["id"]))
        except Exception as exc:
            QMessageBox.warning(self, "Forget failed", str(exc))
            return
        self._selected = None
        self.refresh()
