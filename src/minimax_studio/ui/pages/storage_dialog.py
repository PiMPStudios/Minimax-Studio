"""Storage — the disk half of a long training run (PLAN-V2 S5).

A run that goes well is tens of gigabytes: a checkpoint every N steps, plus a
VAE cache and a text-embedding cache that are each larger than the dataset they
came from. SimpleTuner has no reason to tidy that up, so Studio does — from one
dialog, with the numbers named *before* anything is deleted.

Two rules here are not negotiable:

* **Nothing is deleted while a run is live.** A running trainer holds those
  files open; on Windows the result is not freed disk but a half-deleted run.
* **An installed checkpoint is never pruned.** There is no eval score in this
  contract, so "best" means *the one you chose to keep*, said out loud rather
  than invented.
"""

from __future__ import annotations

import html
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from minimax_studio.ui.reveal import reveal_path
from minimax_studio.worker_client import WorkerClient

_LIVE = {"running", "queued", "lost"}
_KEPT = QBrush(QColor("#32d74b"))
_DIM = QBrush(QColor("#8e8e93"))


def human_bytes(count: Any) -> str:
    """One honest unit, not “4194304”."""
    try:
        size = float(count)
    except (TypeError, ValueError):
        return "—"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _when(timestamp: Any) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(timestamp)))
    except (TypeError, ValueError):
        return "—"


class StorageDialog(QDialog):
    """One run's footprint, and the four things worth doing about it."""

    #: a run folder was deleted — the caller must drop it from its list
    run_deleted = Signal()

    def __init__(self, client: WorkerClient, run: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self._client = client
        self._run = dict(run)
        self._run_id = str(self._run.get("id") or "")
        self._report: dict[str, Any] = {}
        self.setWindowTitle(f"Storage — {self._run.get('name') or self._run_id}")
        self.resize(680, 520)

        root = QVBoxLayout(self)
        heading = QLabel(f"<b>{html.escape(str(self._run.get('name') or self._run_id))}</b>")
        heading.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(heading)
        self._path_label = QLabel(str(self._run.get("path") or ""))
        self._path_label.setObjectName("pageSubtitle")
        self._path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._path_label.setWordWrap(True)
        root.addWidget(self._path_label)
        self._totals = QLabel("Measuring…")
        self._totals.setObjectName("pageSubtitle")
        self._totals.setWordWrap(True)
        root.addWidget(self._totals)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(4)
        self._tree.setHeaderLabels(["Checkpoint", "Size", "Written", "Kept as adapter"])
        self._tree.setRootIsDecorated(False)
        self._tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        root.addWidget(self._tree, 1)

        form = QFormLayout()
        self._keep = QSpinBox()
        self._keep.setRange(1, 50)
        self._keep.setValue(3)
        self._keep.setToolTip(
            "Newest checkpoints to keep. Anything you installed as an adapter "
            "is kept on top of this, whatever the number."
        )
        form.addRow("Keep the newest", self._keep)
        self._include_cache = QCheckBox("Include caches when exporting")
        self._include_cache.setToolTip(
            "Off by default: caches are most of the bytes and the receiving "
            "machine rebuilds them anyway."
        )
        form.addRow(self._include_cache)
        root.addLayout(form)

        row = QHBoxLayout()
        self._prune_btn = QPushButton("Prune checkpoints")
        self._prune_btn.clicked.connect(self._prune)
        self._cache_btn = QPushButton("Clear caches")
        self._cache_btn.setToolTip(
            "Delete the VAE and text-embedding caches. Derived data — the next "
            "run rebuilds them, slowly."
        )
        self._cache_btn.clicked.connect(self._clear_caches)
        self._export_btn = QPushButton("Export…")
        self._export_btn.setToolTip(
            "Copy the run — state, config, checkpoints, log — to a folder, "
            "without its caches."
        )
        self._export_btn.clicked.connect(self._export)
        self._resume_btn = QPushButton("Resume from selected")
        self._resume_btn.setToolTip(
            "Continue training from the checkpoint highlighted above (the "
            "newest one if nothing is highlighted). Same folder, same caches."
        )
        self._resume_btn.clicked.connect(self._resume_selected)
        self._folder_btn = QPushButton("Open folder")
        self._folder_btn.clicked.connect(self._open_folder)
        self._delete_btn = QPushButton("Delete run folder…")
        self._delete_btn.clicked.connect(self._delete_run)
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self.refresh)
        for button in (
            self._prune_btn,
            self._cache_btn,
            self._resume_btn,
            self._export_btn,
            self._folder_btn,
            self._delete_btn,
            self._refresh_btn,
        ):
            row.addWidget(button)
        row.addStretch(1)
        root.addLayout(row)

        self._result = QLabel("")
        self._result.setObjectName("pageSubtitle")
        self._result.setWordWrap(True)
        self._result.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(self._result)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        if buttons.button(QDialogButtonBox.StandardButton.Close) is not None:
            buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(
                self.accept
            )
        root.addWidget(buttons)

        self.refresh()

    # --- the numbers --------------------------------------------------------

    def refresh(self) -> None:
        self._tree.clear()
        try:
            self._report = self._client.train_run_storage(self._run_id)
        except Exception as exc:
            self._totals.setText(f"Could not measure this run: {exc}")
            self._set_enabled(False)
            return
        live = str(self._report.get("status") or "") in _LIVE
        self._totals.setText(
            f"<b>{human_bytes(self._report.get('bytes'))}</b> in this run · "
            f"checkpoints {human_bytes(self._report.get('checkpoint_bytes'))} · "
            f"caches {human_bytes(self._report.get('cache_bytes'))} · "
            f"{self._report.get('free_gb')} GB free on that volume"
        )
        for row in self._report.get("checkpoints") or []:
            item = QTreeWidgetItem(
                [
                    str(row.get("path")),
                    human_bytes(row.get("bytes")),
                    _when(row.get("written_at")),
                    "yes" if row.get("installed") else "",
                ]
            )
            item.setToolTip(0, str(row.get("abs") or ""))
            if row.get("installed"):
                item.setForeground(3, _KEPT)
            self._tree.addTopLevelItem(item)
        if not self._report.get("checkpoints"):
            self._totals.setText(
                self._totals.text() + " · no checkpoints written yet"
            )
        self._set_enabled(True)
        self._resume_btn.setEnabled(bool(self._report.get("checkpoints")))
        if live:
            reason = (
                "Training is live — Studio will not delete files under a "
                "running trainer."
            )
            for button in (self._prune_btn, self._cache_btn, self._delete_btn, self._resume_btn):
                button.setEnabled(False)
                button.setToolTip(reason)
            self._result.setText(f"<span style='color:#ff9f0a'>{reason}</span>")
        else:
            for button in (self._prune_btn, self._cache_btn, self._delete_btn):
                button.setEnabled(True)
            self._resume_btn.setToolTip(
                "Continue training from the checkpoint highlighted above (the "
                "newest one if nothing is highlighted). Same folder, same caches."
            )

    def _set_enabled(self, flag: bool) -> None:
        for button in (
            self._prune_btn,
            self._cache_btn,
            self._resume_btn,
            self._export_btn,
            self._delete_btn,
        ):
            button.setEnabled(flag)

    def _kept_count(self) -> int:
        return len(self._installed_rows())

    def _installed_rows(self) -> list[dict[str, Any]]:
        return [row for row in self._report.get("checkpoints") or [] if row.get("installed")]

    # --- actions ------------------------------------------------------------

    def _prune(self) -> None:
        """Ask first, with the worker's own numbers, then do it.

        The figure comes from a dry run over the same code path rather than from
        arithmetic the dialog does on the row sizes it happens to have — those
        rows are the weight files, and a pruned step folder also carries
        SimpleTuner's optimiser state.
        """
        keep = int(self._keep.value())
        try:
            plan = self._client.prune_train_checkpoints(self._run_id, keep=keep, dry_run=True)
        except Exception as exc:
            self._say(f"<span style='color:#ff453a'>Could not plan the prune: {html.escape(str(exc))}</span>")
            return
        doomed = plan.get("removed") or []
        if not doomed:
            self._say(
                f"Nothing to prune — {len(self._report.get('checkpoints') or [])} "
                f"checkpoint(s), all inside the {keep} you keep (plus "
                f"{self._kept_count()} installed)."
            )
            return
        answer = QMessageBox.question(
            self,
            "Prune checkpoints",
            f"Delete {len(doomed)} older checkpoint(s), about "
            f"{human_bytes(plan.get('freed_bytes'))}?\n\n"
            f"Kept: the newest {keep} and the {self._kept_count()} installed as "
            "adapters. A pruned step goes whole — its folder with SimpleTuner's "
            "optimiser state included. This cannot be undone.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            result = self._client.prune_train_checkpoints(self._run_id, keep=keep)
        except Exception as exc:
            self._say(f"<span style='color:#ff453a'>Prune failed: {html.escape(str(exc))}</span>")
            return
        self._say(
            f"Freed {human_bytes(result.get('freed_bytes'))} — kept "
            f"{len(result.get('kept') or [])} checkpoint(s)."
        )
        self.refresh()

    def _clear_caches(self) -> None:
        size = human_bytes(self._report.get("cache_bytes"))
        answer = QMessageBox.question(
            self,
            "Clear caches",
            f"Delete this run's VAE and text-embedding caches ({size})?\n\n"
            "Nothing trained is lost — they are derived data. The next run "
            "rebuilds them, which costs an hour or two of GPU, not the weights.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            result = self._client.clear_train_cache(self._run_id)
        except Exception as exc:
            self._say(f"<span style='color:#ff453a'>Clear failed: {html.escape(str(exc))}</span>")
            return
        self._say(
            "No cache folder to clear."
            if not result.get("cleared")
            else f"Freed {human_bytes(result.get('freed_bytes'))} of caches."
        )
        self.refresh()

    def _resume_selected(self) -> None:
        """The resume picker: any checkpoint of this run, or the newest one.

        The number of the checkpoint is the whole UI here — SimpleTuner re-reads
        the optimiser state from the file, so the choice is about *where to pick
        up*, not about configuration.
        """
        rows = self._report.get("checkpoints") or []
        if not rows:
            self._say("Nothing to resume from — this run has not written a checkpoint.")
            return
        item = self._tree.currentItem()
        index = self._tree.indexOfTopLevelItem(item) if item is not None else -1
        row = rows[index] if 0 <= index < len(rows) else rows[0]
        answer = QMessageBox.question(
            self,
            "Resume training",
            f"Continue “{self._run.get('name')}” from "
            f"{row.get('path')} ({human_bytes(row.get('bytes'))})?\n\n"
            "Same run folder, same caches — a new trainer process starts and "
            "the log keeps going.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            state = self._client.resume_train_run(self._run_id, str(row.get("abs") or row.get("path")))
        except Exception as exc:
            self._say(f"<span style='color:#ff453a'>Resume failed: {html.escape(str(exc))}</span>")
            return
        self._say(
            f"Resumed as pid {state.get('pid')} (resume "
            f"#{state.get('resume_count')}) — the log is in the run folder."
        )
        self.refresh()

    def _export(self) -> None:
        dest = QFileDialog.getExistingDirectory(
            self, "Export run to…", str(Path.home())
        )
        if not dest:
            return
        try:
            result = self._client.export_train_run(
                self._run_id, dest, include_cache=self._include_cache.isChecked()
            )
        except Exception as exc:
            self._say(f"<span style='color:#ff453a'>Export failed: {html.escape(str(exc))}</span>")
            return
        self._say(
            f"Exported {result.get('files')} file(s), "
            f"{human_bytes(result.get('bytes'))} → {html.escape(str(result.get('path')))}"
            + (
                ""
                if self._include_cache.isChecked()
                else " (caches left behind — import and train, they rebuild)"
            )
        )

    def _delete_run(self) -> None:
        size = human_bytes(self._report.get("bytes"))
        answer = QMessageBox.question(
            self,
            "Delete run folder",
            f"Delete the whole “{self._run.get('name')}” folder ({size})?\n\n"
            "Checkpoints, config and the log all go. Adapters you installed "
            "stay in the LoRA folder — they are copies.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            result = self._client.delete_train_run(self._run_id)
        except Exception as exc:
            self._say(f"<span style='color:#ff453a'>Delete failed: {html.escape(str(exc))}</span>")
            return
        self.run_deleted.emit()
        self._say(f"Deleted. Freed {human_bytes(result.get('freed_bytes'))}.")
        self._set_enabled(False)

    def _open_folder(self) -> None:
        path = str(self._report.get("path") or self._run.get("path") or "")
        if not path:
            self._say("This run folder is not on disk.")
            return
        try:
            reveal_path(path)
        except OSError as exc:
            self._say(f"<span style='color:#ff453a'>Could not open it: {html.escape(str(exc))}</span>")

    def _say(self, text: str) -> None:
        self._result.setText(text)
