"""Train LoRA — the S2 half of the Build pages (PLAN-V2).

Studio writes SimpleTuner's two config files, launches it as **its own process
group**, and then stays out of the way: close the app, quit, crash — the run
keeps going and this page reattaches from ``state.json`` next time. That
property is the whole reason training is not a job in the generate queue.

Preflight runs before the GPU gets burned, and says so in numbers: which pack
is missing, how much VRAM is free versus needed, how much cache disk is left.
"""

from __future__ import annotations

import html
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from minimax_studio.ui.pages.datasets_page import validation_status_short
from minimax_studio.ui.pages.storage_dialog import StorageDialog
from minimax_studio.ui.reveal import reveal_path
from minimax_studio.worker_client import WorkerClient

_BAD = QBrush(QColor("#ff453a"))
_WARN = QBrush(QColor("#ff9f0a"))
_OK = QBrush(QColor("#32d74b"))
_LIVE = {"running", "queued", "lost"}

# Fallback only: the real list arrives from /train/preflight so the picker can
# never advertise a preset the worker would refuse.
_FALLBACK_PRESETS = {
    "24g": {
        "title": "24 GB — conservative LoRA",
        "vram_floor_gb": 24,
        "lora_rank": 16,
        "note": "int8 everywhere + gradient checkpointing",
        "family": "music",
    },
    "h3-24g": {
        "title": "24 GB — H3 LoRA with RamTorch",
        "vram_floor_gb": 24,
        "lora_rank": 16,
        "note": "RamTorch CPU-offload, ConvRot INT8, 480p buckets",
        "family": "h3",
    },
}


def _eta(step: int | None, total: int | None, started_at: float | None) -> str:
    """Honest ETA from measured step time, or an honest shrug."""
    if not step or not total or not started_at:
        return ""
    elapsed = max(1.0, time.time() - float(started_at))
    remaining = (total - step) * (elapsed / step)
    if remaining <= 0:
        return "finishing up"
    hours, minutes = divmod(int(remaining), 3600)
    minutes, seconds = divmod(minutes, 60)
    if hours:
        return f"about {hours}h {minutes:02d}m left at this pace"
    if minutes:
        return f"about {minutes}m {seconds:02d}s left at this pace"
    return f"about {seconds}s left at this pace"


def _run_line(run: dict[str, Any]) -> tuple[str, str, str]:
    """List row: the run dir is the only truth, so steps/loss stay “—” here."""
    total = run.get("steps") or "?"
    return f"— / {total}", str(run.get("status") or ""), "—"


class TrainPage(QWidget):
    #: an adapter was copied into the LoRA folder — refresh the picker
    adapter_installed = Signal()
    #: a run folder was deleted from the Storage dialog
    run_deleted = Signal()

    def __init__(self, client: WorkerClient) -> None:
        super().__init__()
        self._client = client
        self._datasets: list[dict[str, Any]] = []
        self._runs: list[dict[str, Any]] = []
        self._selected_run: str | None = None
        self._detail: dict[str, Any] = {}
        self._presets: dict[str, dict[str, Any]] = {
            name: row
            for name, row in _FALLBACK_PRESETS.items()
            if str(row.get("family") or "music") == "music"
        }
        # The whole table as the worker described it, kept apart from the
        # filtered view above: filtering a filtered list loses the other family.
        self._all_presets: dict[str, dict[str, Any]] = {}
        self._preflight: dict[str, Any] = {}
        self._poll_thread = None

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        title = QLabel("Train LoRA")
        title.setObjectName("pageTitle")
        subtitle = QLabel(
            "Studio generates the SimpleTuner config and runs it as its own "
            "process — <b>close Studio and the run keeps going</b>, and this "
            "page reattaches to it. "
            "<span style='color:#ff9f0a'>Experimental:</span> Music 3 and H3 "
            "LoRAs, CUDA only, and a 24 GB card is the floor. The VRAM presets "
            "change to match the dataset you pick — the two trainers never "
            "share a run."
        )
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)
        subtitle.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(title)
        root.addWidget(subtitle)

        split = QSplitter()

        left = QWidget()
        left_row = QVBoxLayout(left)
        left_row.setContentsMargins(0, 0, 12, 0)
        form_box = QGroupBox("New run")
        form = QFormLayout(form_box)
        self._dataset = QComboBox()
        self._dataset.setToolTip(
            "Music and H3 datasets both appear here. Picking one changes the "
            "VRAM presets to the trainer that reads it."
        )
        self._dataset.currentIndexChanged.connect(self._dataset_changed)
        self._name = QLineEdit()
        self._name.setPlaceholderText("my summer LoRA")
        self._preset = QComboBox()
        self._preset.currentIndexChanged.connect(self._preset_changed)
        self._steps = QSpinBox()
        self._steps.setRange(10, 200000)
        self._steps.setValue(1000)
        self._steps.setToolTip(
            "Small datasets learn in a few hundred steps. Every checkpoint is "
            "kept in the run folder; install any of them."
        )
        self._rank = QSpinBox()
        self._rank.setRange(1, 256)
        self._rank.setValue(16)
        self._rank.setToolTip(
            "LoRA rank — capacity versus size. The preset's default is "
            "SimpleTuner's conservative choice for this VRAM tier."
        )
        self._validation_prompt = QLineEdit()
        self._validation_prompt.setPlaceholderText(
            "bright synth pop with clean vocal melody (default if blank)"
        )
        self._validation_prompt.setToolTip(
            "SimpleTuner renders something with this prompt during the run, so "
            "you see progress instead of watching a loss number."
        )
        self._validation_duration = QSpinBox()
        self._validation_duration.setRange(5, 60)
        self._validation_duration.setValue(15)
        self._validation_duration.setSuffix(" s")
        self._validation_duration.setToolTip(
            "How long the Music validation clip is. H3 validation samples stay "
            "at SimpleTuner's own defaults — we do not guess a frame count."
        )
        self._dataset_label = QLabel("Music dataset")
        form.addRow(self._dataset_label, self._dataset)
        form.addRow("Run name", self._name)
        form.addRow("VRAM preset", self._preset)
        form.addRow("Steps", self._steps)
        form.addRow("LoRA rank", self._rank)
        form.addRow("Validation prompt", self._validation_prompt)
        form.addRow("Validation length", self._validation_duration)
        left_row.addWidget(form_box)

        check_row = QHBoxLayout()
        self._check_btn = QPushButton("Check again")
        self._check_btn.clicked.connect(self.preflight)
        check_row.addWidget(self._check_btn)
        check_row.addStretch(1)
        left_row.addLayout(check_row)

        self._preflight_label = QLabel("Checking requirements…")
        self._preflight_label.setObjectName("pageSubtitle")
        self._preflight_label.setWordWrap(True)
        self._preflight_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        left_row.addWidget(self._preflight_label)

        self._start_btn = QPushButton("Start training")
        self._start_btn.setObjectName("primary")
        self._start_btn.clicked.connect(self._start_run)
        left_row.addWidget(self._start_btn)
        self._form_status = QLabel("")
        self._form_status.setObjectName("pageSubtitle")
        self._form_status.setWordWrap(True)
        left_row.addWidget(self._form_status)
        left_row.addStretch(1)
        split.addWidget(left)

        right = QWidget()
        right_row = QVBoxLayout(right)
        right_row.setContentsMargins(0, 0, 0, 0)
        runs_box = QGroupBox("Runs")
        runs_layout = QVBoxLayout(runs_box)
        self._runs_tree = QTreeWidget()
        self._runs_tree.setHeaderLabels(["Run", "Status", "Step", "Loss"])
        self._runs_tree.setColumnWidth(0, 220)
        self._runs_tree.setColumnWidth(1, 170)
        self._runs_tree.header().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch
        )
        self._runs_tree.currentItemChanged.connect(lambda *_: self._run_selected())
        runs_layout.addWidget(self._runs_tree, 2)
        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.setValue(0)
        self._progress.setFormat("%p%")
        runs_layout.addWidget(self._progress)
        self._run_status = QLabel("Select a run to see its log.")
        self._run_status.setObjectName("pageSubtitle")
        self._run_status.setWordWrap(True)
        runs_layout.addWidget(self._run_status)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(600)
        self._log.setStyleSheet("font-family: monospace; font-size: 11px;")
        runs_layout.addWidget(self._log, 3)
        row = QHBoxLayout()
        self._cancel_btn = QPushButton("Cancel run")
        self._cancel_btn.clicked.connect(self._cancel_run)
        self._install_btn = QPushButton("Install adapter")
        self._install_btn.setToolTip(
            "Copy the newest checkpoint into the LoRA folder so the Generate "
            "Music or Generate Video picker can load it."
        )
        self._install_btn.clicked.connect(self._install_adapter)
        self._folder_btn = QPushButton("Open folder")
        self._folder_btn.clicked.connect(self._open_folder)
        self._resume_btn = QPushButton("Resume from checkpoint")
        self._resume_btn.setToolTip(
            "Continue this run from its newest checkpoint — same folder, same "
            "caches, new process. For the run that died at 4 a.m."
        )
        self._resume_btn.clicked.connect(self._resume_run)
        self._storage_btn = QPushButton("Storage…")
        self._storage_btn.setToolTip(
            "What this run is using, and what can safely be freed: prune old "
            "checkpoints, clear caches, export the run, delete the folder."
        )
        self._storage_btn.clicked.connect(self._open_storage)
        self._import_btn = QPushButton("Import run folder")
        self._import_btn.setToolTip(
            "Pick up a run folder Studio exported — or one copied off another "
            "machine — and put it back in the list with its history."
        )
        self._import_btn.clicked.connect(self._import_run)
        row.addWidget(self._install_btn)
        row.addWidget(self._cancel_btn)
        row.addWidget(self._resume_btn)
        row.addWidget(self._storage_btn)
        row.addWidget(self._import_btn)
        row.addWidget(self._folder_btn)
        row.addStretch(1)
        runs_layout.addLayout(row)
        right_row.addWidget(runs_box, 1)
        split.addWidget(right)
        split.setStretchFactor(1, 2)
        root.addWidget(split, 1)

        self.refresh()
        self.preflight()

    # --- data ---------------------------------------------------------------

    def refresh(self) -> None:
        self._load_datasets()
        try:
            runs = self._client.list_train_runs()
        except Exception:
            self._run_status.setText("Could not list runs — keeping the last list.")
            return
        self._apply_runs(runs)
        self._show_run()

    def poll(self) -> None:
        """Timer path: list + log tail off the GUI thread. Overlapping polls skip."""
        selected = self._selected_run
        from minimax_studio.ui.enhance import start_background

        def work() -> tuple[list, object, object]:
            runs = self._client.list_train_runs()
            datasets = None
            try:
                datasets = self._client.list_datasets()
            except Exception:
                datasets = None
            detail = None
            if selected:
                try:
                    detail = self._client.get_train_run(selected, tail=80)
                except Exception:
                    detail = None
            return runs, detail, datasets

        def done(payload: object) -> None:
            if not isinstance(payload, tuple) or len(payload) != 3:
                return
            runs, detail, datasets = payload
            if isinstance(datasets, list):
                self._load_datasets(datasets)
            if isinstance(runs, list):
                self._apply_runs(runs)
            if self._selected_run != selected:
                self._show_run()
            elif isinstance(detail, dict):
                self._paint_run(detail)
            elif selected:
                self._run_status.setText(
                    "Could not read this run — keeping the last log."
                )
                for button in (
                    self._cancel_btn,
                    self._install_btn,
                    self._resume_btn,
                    self._storage_btn,
                ):
                    button.setEnabled(False)

        def fail() -> None:
            self._run_status.setText("Could not list runs — keeping the last list.")

        start_background(self, work, done, fail, attr="_poll_thread")

    def _apply_runs(self, runs: list[dict[str, Any]]) -> None:
        signature = [(row.get("id"), row.get("status")) for row in runs]
        previous = [(row.get("id"), row.get("status")) for row in self._runs]
        self._runs = runs
        if (
            signature == previous
            and self._runs_tree.topLevelItemCount() == len(runs)
        ):
            return
        wanted = self._selected_run
        self._runs_tree.blockSignals(True)
        self._runs_tree.clear()
        items: dict[str, QTreeWidgetItem] = {}
        for run in self._runs:
            step, status, loss = _run_line(run)
            item = QTreeWidgetItem(
                [
                    str(run.get("name") or run.get("id")),
                    status,
                    step,
                    loss,
                ]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, str(run.get("id")))
            if run.get("status") in _LIVE:
                item.setForeground(1, _OK)
            elif run.get("status") == "cancelled":
                item.setForeground(1, _WARN)
            elif run.get("status") != "completed":
                item.setForeground(1, _BAD)
            self._runs_tree.addTopLevelItem(item)
            items[str(run.get("id"))] = item
        if wanted and wanted in items:
            self._runs_tree.setCurrentItem(items[wanted])
            self._selected_run = wanted
        elif self._runs_tree.topLevelItemCount():
            first = self._runs_tree.topLevelItem(0)
            self._runs_tree.setCurrentItem(first)
            self._selected_run = first.data(0, Qt.ItemDataRole.UserRole)
        else:
            self._selected_run = None
        self._runs_tree.blockSignals(False)
        if not self._runs:
            self._paint_empty()

    def _load_datasets(self, rows: list[dict[str, Any]] | None = None) -> None:
        if rows is None:
            try:
                rows = self._client.list_datasets()
            except Exception:
                return
        trainable = [row for row in rows if row.get("kind") in ("music", "video")]
        # Rebuilding a picker the user is using loses their selection, so only
        # do it when the list actually changed. (An empty list still has to
        # flow through: that is the "make a dataset first" state.)
        signature = [
            (row.get("id"), row.get("clip_count"), validation_status_short(row))
            for row in trainable
        ]
        previous = [
            (row.get("id"), row.get("clip_count"), validation_status_short(row))
            for row in self._datasets
        ]
        if trainable and signature == previous:
            self._datasets = trainable
            return
        previous_kind = self._family()
        self._datasets = trainable
        current = self._dataset.currentData()
        self._dataset.blockSignals(True)
        self._dataset.clear()
        for row in trainable:
            # "2 problems" and "not validated yet" are different facts; the
            # picker must not blur them into one warning word.
            count = int(row.get("clip_count") or 0)
            if row.get("kind") == "video":
                label = (
                    f"{row.get('name')} — {count} stills/clips · "
                    f"{validation_status_short(row)} [H3]"
                )
            else:
                label = (
                    f"{row.get('name')} — {count} clips · "
                    f"{validation_status_short(row)}"
                )

            # The whole row as item data: the id is what validate takes, the
            # path is what training takes. Neither is ours to rebuild.
            self._dataset.addItem(label, row)
        if current:
            self._dataset.setCurrentIndex(max(0, self._dataset.findData(current)))
        self._dataset.blockSignals(False)
        self._dataset.setEnabled(bool(trainable))
        self._start_btn.setEnabled(bool(trainable))
        if self._family() != previous_kind:
            self._adopt_presets(self._all_presets or self._presets)
        self._show_family_fields()
        if not trainable:
            self._form_status.setText(
                "No dataset yet — make one on the Datasets page: import "
                "5–20 clips for Music, or stills and short clips for H3, then "
                "Validate."
            )

    def _family(self) -> str:
        """Which trainer the picked dataset is for. One place decides, so the
        preset list, the form and the payload cannot disagree."""
        row = self._dataset.currentData() or {}
        return "h3" if str(row.get("kind")) == "video" else "music"

    def _show_family_fields(self) -> None:
        """An H3 run has no audio length to promise; hiding the box beats a
        control that does nothing."""
        music = self._family() == "music"
        self._validation_duration.setVisible(music)
        self._dataset_label.setText(
            "Music dataset" if music else "H3 dataset (stills and short clips)"
        )
        self._validation_prompt.setPlaceholderText(
            "bright synth pop with clean vocal melody (default if blank)"
            if music
            else "a steady camera push across a neon street at dusk (default if blank)"
        )

    def _dataset_changed(self) -> None:
        """Changing dataset changes the trainer: re-pick the tier list from the
        full table and ask the worker to check this pair, not the previous one."""
        self._adopt_presets(self._all_presets or self._presets)
        self._show_family_fields()
        self.preflight()

    def select_dataset(self, dataset_id: str) -> None:
        for index in range(self._dataset.count()):
            row = self._dataset.itemData(index) or {}
            if row.get("id") == dataset_id:
                self._dataset.setCurrentIndex(index)
                return

    def preflight(self, _recheck: bool = False) -> None:
        preset = str(self._preset.currentData() or "24g")
        dataset = self._dataset.currentData() or {}
        dataset_dir = str(dataset.get("path") or "") or None
        try:
            check = self._client.train_preflight(preset, dataset_dir)
        except Exception as exc:
            self._preflight = {}
            self._preflight_label.setText(f"Could not reach the check: {exc}")
            return
        self._preflight = check
        presets = check.get("presets")
        if isinstance(presets, dict) and presets:
            switched = self._adopt_presets(presets)
            if switched and not _recheck:
                # We asked about the tier we were showing, and the returned table
                # then moved that selection (a dataset switch). Numbers about a
                # preset we no longer display would be someone else's verdict.
                return self.preflight(_recheck=True)
        lines: list[str] = []
        if check.get("ok"):
            lines.append(
                f"<b style='color:#32d74b'>Ready.</b> {html.escape(str(check.get('detail') or ''))}"
            )
        else:
            lines.append("<b style='color:#ff453a'>Not yet.</b>")
        for problem in check.get("problems") or []:
            lines.append(f"• {html.escape(str(problem))}")
        for warning in check.get("warnings") or []:
            lines.append(
                f"<span style='color:#ff9f0a'>• {html.escape(str(warning))}</span>"
            )
        self._preflight_label.setText("<br>".join(lines))

    def _adopt_presets(self, presets: dict[str, dict[str, Any]]) -> bool:
        """Build the VRAM picker from what the worker actually supports — for
        the model this dataset belongs to. A Music tier list under an H3 dataset
        would be a trap with a number on it.

        Returns whether the *selected* preset changed, so the caller can tell a
        fresh check apart from a verdict about a tier it is no longer showing.
        """
        if presets:
            self._all_presets = presets
        family = self._family()
        rows = {
            name: preset
            for name, preset in (presets or self._presets).items()
            if str(preset.get("family") or "music") == family
        }
        if not rows:
            rows = {
                name: preset
                for name, preset in _FALLBACK_PRESETS.items()
                if str(preset.get("family") or "music") == family
            } or dict(_FALLBACK_PRESETS)
        if rows == self._presets:
            return False
        before = self._preset.currentData()
        self._presets = rows
        current = self._preset.currentData()
        self._preset.blockSignals(True)
        self._preset.clear()
        for name, preset in sorted(
            rows.items(), key=lambda item: item[1].get("vram_floor_gb") or 0
        ):
            self._preset.addItem(
                f"{preset.get('title')} · rank {preset.get('lora_rank')}", name
            )
        if current and self._preset.findData(current) >= 0:
            self._preset.setCurrentIndex(self._preset.findData(current))
        self._preset.blockSignals(False)
        self._sync_rank_from_preset()
        return self._preset.currentData() != before

    def _sync_rank_from_preset(self) -> None:
        preset = self._presets.get(str(self._preset.currentData() or ""), {})
        rank = preset.get("lora_rank")
        if rank:
            self._rank.setValue(int(rank))

    def _preset_changed(self) -> None:
        self._sync_rank_from_preset()
        self.preflight()

    def _percent(self, step: int | None, total: int) -> int:
        if not total:
            return 0
        return int(max(0.0, min(1.0, int(step or 0) / float(total))) * 1000)

    # --- the selected run ---------------------------------------------------

    def _run_selected(self) -> None:
        item = self._runs_tree.currentItem()
        self._selected_run = item.data(0, Qt.ItemDataRole.UserRole) if item else None
        self._show_run()

    def _paint_empty(self) -> None:
        self._log.setPlainText("")
        self._progress.setValue(0)
        self._run_status.setText("Select a run to see its log.")
        self._detail = {}
        for button in (
            self._cancel_btn,
            self._install_btn,
            self._folder_btn,
            self._resume_btn,
            self._storage_btn,
        ):
            button.setEnabled(False)

    def _show_run(self) -> None:
        run_id = self._selected_run
        if not run_id:
            self._paint_empty()
            return
        try:
            detail = self._client.get_train_run(run_id, tail=80)
        except Exception as exc:
            self._run_status.setText(f"Could not read this run: {exc}")
            for button in (
                self._cancel_btn,
                self._install_btn,
                self._resume_btn,
                self._storage_btn,
            ):
                button.setEnabled(False)
            self._folder_btn.setEnabled(bool((self._detail or {}).get("path")))
            return
        self._paint_run(detail)

    def _paint_run(self, detail: dict[str, Any]) -> None:
        self._detail = detail
        self._log.setPlainText("\n".join(detail.get("log_tail") or []))
        progress = detail.get("progress") or {}
        total = int(progress.get("total_steps") or detail.get("steps") or 0)
        step = progress.get("step")
        self._progress.setValue(self._percent(step, total))
        bits = []
        if step is not None:
            bits.append(f"step {step} of {total or '?'}")
        if progress.get("loss") is not None:
            bits.append(f"loss {float(progress['loss']):.4f}")
        checkpoints = progress.get("checkpoints") or []
        if checkpoints:
            bits.append(f"{len(checkpoints)} checkpoint(s) on disk")
        eta = _eta(step, total, detail.get("started_at"))
        if eta:
            bits.append(eta)
        status = str(detail.get("status") or "")
        if status not in _LIVE and detail.get("exit_code") not in (None, 0):
            bits.append(f"exit code {detail['exit_code']}")
        self._run_status.setText(
            f"<b>{html.escape(str(detail.get('name')))}</b> — {status}. "
            + " · ".join(bits)
        )
        self._run_status.setTextFormat(Qt.TextFormat.RichText)
        self._cancel_btn.setEnabled(status in _LIVE and not detail.get("cancel_requested"))
        self._install_btn.setEnabled(bool(checkpoints))
        self._folder_btn.setEnabled(bool(detail.get("path")))
        # Resume is for the run that stopped, not the one that is going: two
        # trainers on one folder corrupt the checkpoint series, not the log.
        self._resume_btn.setEnabled(bool(checkpoints) and status not in _LIVE)
        self._storage_btn.setEnabled(bool(detail.get("path")))

    # --- actions ------------------------------------------------------------

    def _start_run(self) -> None:
        dataset = self._dataset.currentData() or {}
        dataset_dir = str(dataset.get("path") or "")
        dataset_id = str(dataset.get("id") or "")
        if not dataset_dir or not dataset_id:
            QMessageBox.warning(
                self,
                "No dataset",
                "Pick a dataset — make one on the Datasets page first. Music "
                "takes clips; H3 takes stills or short clips.",
            )
            return
        name = self._name.text().strip() or "training run"
        preset = str(self._preset.currentData() or "24g")
        preset_title = self._preset.currentText()
        steps = int(self._steps.value())
        rank = int(self._rank.value())
        validation: dict[str, Any] = {
            "prompt": self._validation_prompt.text().strip()
        }
        if self._family() == "music":
            # An H3 run has no audio length to promise; sending one would be a
            # key the trainer either ignores or rejects.
            validation["duration"] = int(self._validation_duration.value())
        self._form_status.setText("Checking requirements…")
        self._start_btn.setEnabled(False)
        from minimax_studio.ui.enhance import start_background

        def checks() -> dict[str, Any]:
            try:
                check = self._client.train_preflight(preset, str(dataset_dir))
            except Exception as exc:
                return {
                    "phase": "error",
                    "title": "Could not check requirements",
                    "message": str(exc),
                }
            if not check.get("ok"):
                return {"phase": "preflight", "check": check}
            try:
                report = self._client.validate_dataset(dataset_id)
            except Exception as exc:
                return {
                    "phase": "error",
                    "title": "Dataset check failed",
                    "message": str(exc),
                }
            return {"phase": "checked", "check": check, "report": report}

        def after_checks(payload: object) -> None:
            self.preflight()
            if not isinstance(payload, dict):
                self._start_btn.setEnabled(True)
                return
            if payload.get("phase") == "error":
                self._start_btn.setEnabled(True)
                QMessageBox.warning(
                    self,
                    str(payload.get("title") or "Could not check requirements"),
                    str(payload.get("message") or "Worker unreachable"),
                )
                return
            if payload.get("phase") == "preflight":
                self._start_btn.setEnabled(True)
                check = payload.get("check") or {}
                QMessageBox.warning(
                    self,
                    "Not ready — nothing was started",
                    str(check.get("detail") or " ".join(check.get("problems") or [])),
                )
                return
            report = payload.get("report") or {}
            if not report.get("ok"):
                self._start_btn.setEnabled(True)
                bad = [entry for entry in report.get("rows", []) if not entry.get("ok")]
                first = bad[0] if bad else {}
                QMessageBox.warning(
                    self,
                    "This dataset is not ready",
                    f"{len(bad)} of {report.get('checked', 0)} clips have problems — "
                    f"first: {first.get('file')} "
                    f"{'; '.join(first.get('problems') or [])}. Fix them on the "
                    "Datasets page; hours of VRAM is a rough way to find a missing "
                    "caption.",
                )
                return
            clips = report.get("checked", 0)
            answer = QMessageBox.question(
                self,
                "Start training",
                f"Train “{name}” for {steps} steps on {clips} clips "
                f"({preset_title})?\n\n"
                "The run is its own process: closing Studio will not stop it. "
                "Cancel any time — it resumes from the last checkpoint.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                self._start_btn.setEnabled(True)
                return
            try:
                run = self._client.start_train_run(
                    {
                        "name": name,
                        "dataset_dir": str(dataset_dir),
                        "preset": preset,
                        "steps": steps,
                        "rank": rank,
                        "validation": validation,
                    }
                )
            except Exception as exc:
                self._start_btn.setEnabled(True)
                QMessageBox.warning(self, "Training did not start", str(exc))
                return
            self._start_btn.setEnabled(True)
            self._form_status.setText(
                f"Started “{run.get('name')}” (pid {run.get('pid')}). "
                "It survives Studio closing — this page reattaches."
            )
            self._selected_run = str(run.get("id"))
            self.refresh()

        def checks_fail() -> None:
            self._start_btn.setEnabled(True)
            QMessageBox.warning(
                self, "Could not check requirements", "Worker unreachable"
            )

        if not start_background(
            self, checks, after_checks, checks_fail, attr="_check_thread"
        ):
            self._start_btn.setEnabled(True)

    def _cancel_run(self) -> None:
        run_id = self._selected_run
        if not run_id:
            return
        answer = QMessageBox.question(
            self,
            "Cancel run",
            "Stop the training run? Its checkpoints stay in the run folder, "
            "so nothing trained so far is lost.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._client.cancel_train_run(run_id)
        except Exception as exc:
            QMessageBox.warning(self, "Cancel failed", str(exc))
            return
        self.refresh()

    def _install_adapter(self) -> None:
        run_id = self._selected_run
        if not run_id:
            return
        try:
            row = self._client.install_train_adapter(run_id)
        except Exception as exc:
            QMessageBox.warning(self, "Install failed", str(exc))
            return
        self.adapter_installed.emit()
        family = str(self._detail.get("family") or "")
        if family == "h3":
            where = (
                "Pick it in the LoRA dropdown on Generate Video, or Audition "
                "it from Adapters — a short still-pair (or text-to-video) at "
                "0.8 strength, badged in History."
            )
        else:
            where = (
                "Pick it in the LoRA dropdown on Generate Music — around 0.8 "
                "strength is a good first audition."
            )
        QMessageBox.information(
            self,
            "Adapter installed",
            f"“{row.get('name')}” is in the LoRA folder now. {where}",
        )

    def _open_folder(self) -> None:
        item = self._runs_tree.currentItem()
        run_id = str(item.data(0, Qt.ItemDataRole.UserRole)) if item else ""
        run = next((row for row in self._runs if row.get("id") == run_id), None)
        path = str((run or {}).get("path") or "")
        if not path:
            QMessageBox.information(
                self, "Open folder", "This run folder is not on disk."
            )
            return
        try:
            reveal_path(path)
        except OSError as exc:
            QMessageBox.warning(self, "Open folder failed", str(exc))

    def _resume_run(self) -> None:
        run_id = self._selected_run
        if not run_id:
            return
        answer = QMessageBox.question(
            self,
            "Resume training",
            f"Continue “{self._detail.get('name')}” from its newest checkpoint?\n\n"
            "Same run folder, same caches — a new trainer process starts and "
            "the log keeps going. Nothing is re-trained from scratch. "
            "Storage… can resume from a specific checkpoint instead.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            run = self._client.resume_train_run(run_id)
        except Exception as exc:
            QMessageBox.warning(self, "Resume failed", str(exc))
            return
        self._form_status.setText(
            f"Resumed “{run.get('name')}” from "
            f"{Path(str(run.get('resumed_from') or '')).name} as pid "
            f"{run.get('pid')} (resume #{run.get('resume_count')})."
        )
        self.refresh()

    def _open_storage(self) -> None:
        run_id = self._selected_run
        if not run_id:
            return
        run = next((row for row in self._runs if row.get("id") == run_id), None) or {}
        dialog = StorageDialog(self._client, run, self)
        dialog.run_deleted.connect(self._run_was_deleted)
        dialog.exec()
        # Sizes changed even if nothing was deleted (a prune, an export).
        self.refresh()

    def _run_was_deleted(self) -> None:
        self._selected_run = None
        self._detail = {}
        self.run_deleted.emit()
        self.refresh()

    def _import_run(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Import a Studio run folder", str(Path.home())
        )
        if not folder:
            return
        try:
            run = self._client.import_train_run(folder)
        except Exception as exc:
            QMessageBox.warning(self, "Import failed", str(exc))
            return
        self._selected_run = str(run.get("id"))
        self._form_status.setText(
            f"Imported “{run.get('name')}” into {run.get('path')}."
        )
        self.refresh()
