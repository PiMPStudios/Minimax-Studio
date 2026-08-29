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
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QComboBox,
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
from minimax_studio.ui.reveal import reveal_path
from minimax_studio.worker_client import WorkerClient

_BAD = QBrush(QColor("#ff453a"))
_WARN = QBrush(QColor("#ff9f0a"))
_OK = QBrush(QColor("#32d74b"))
_LIVE = {"running", "queued"}

# Fallback only: the real list arrives from /train/preflight so the picker can
# never advertise a preset the worker would refuse.
_FALLBACK_PRESETS = {
    "24g": {
        "title": "24 GB — conservative LoRA",
        "vram_floor_gb": 24,
        "lora_rank": 16,
        "note": "int8 everywhere + gradient checkpointing",
    }
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

    def __init__(self, client: WorkerClient) -> None:
        super().__init__()
        self._client = client
        self._datasets: list[dict[str, Any]] = []
        self._runs: list[dict[str, Any]] = []
        self._selected_run: str | None = None
        self._presets: dict[str, dict[str, Any]] = dict(_FALLBACK_PRESETS)
        self._preflight: dict[str, Any] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        title = QLabel("Train LoRA")
        title.setObjectName("pageTitle")
        subtitle = QLabel(
            "Studio generates the SimpleTuner config and runs it as its own "
            "process — <b>close Studio and the run keeps going</b>, and this "
            "page reattaches to it. "
            "<span style='color:#ff9f0a'>Experimental:</span> Music 3 LoRAs "
            "only, CUDA only, and a 24 GB card is the floor."
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
            "Only Music datasets appear — the H3 trainer arrives in PLAN-V2 S4."
        )
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
            "SimpleTuner renders a clip with this prompt during the run, so "
            "you hear progress instead of watching a loss number."
        )
        self._validation_duration = QSpinBox()
        self._validation_duration.setRange(5, 60)
        self._validation_duration.setValue(15)
        self._validation_duration.setSuffix(" s")
        form.addRow("Dataset", self._dataset)
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
            "Music picker can load it."
        )
        self._install_btn.clicked.connect(self._install_adapter)
        self._folder_btn = QPushButton("Open folder")
        self._folder_btn.clicked.connect(self._open_folder)
        row.addWidget(self._install_btn)
        row.addWidget(self._cancel_btn)
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
            self._runs = self._client.list_train_runs()
        except Exception:
            self._runs = []
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
        self._runs_tree.blockSignals(False)
        if wanted and wanted in items:
            self._runs_tree.setCurrentItem(items[wanted])
        elif self._runs_tree.topLevelItemCount():
            self._runs_tree.setCurrentItem(self._runs_tree.topLevelItem(0))
        else:
            self._selected_run = None
            self._show_run()

    def _load_datasets(self) -> None:
        try:
            rows = self._client.list_datasets()
        except Exception:
            return
        trainable = [row for row in rows if row.get("kind") == "music"]
        # Rebuilding a picker the user is using loses their selection, so only
        # do it when the list actually changed. (An empty list still has to
        # flow through: that is the "make a dataset first" state.)
        if trainable and [row.get("id") for row in trainable] == [
            row.get("id") for row in self._datasets
        ]:
            return
        self._datasets = trainable
        current = self._dataset.currentData()
        self._dataset.blockSignals(True)
        self._dataset.clear()
        for row in trainable:
            # "2 problems" and "not validated yet" are different facts; the
            # picker must not blur them into one warning word.
            label = f"{row.get('name')} — {row.get('clip_count', 0)} clips · {validation_status_short(row)}"

            # The whole row as item data: the id is what validate takes, the
            # path is what training takes. Neither is ours to rebuild.
            self._dataset.addItem(label, row)
        if current:
            self._dataset.setCurrentIndex(max(0, self._dataset.findData(current)))
        self._dataset.blockSignals(False)
        self._dataset.setEnabled(bool(trainable))
        self._start_btn.setEnabled(bool(trainable))
        if not trainable:
            self._form_status.setText(
                "No Music dataset yet — make one on the Datasets page: import "
                "5–20 clips you like and Validate."
            )

    def select_dataset(self, dataset_id: str) -> None:
        for index in range(self._dataset.count()):
            row = self._dataset.itemData(index) or {}
            if row.get("id") == dataset_id:
                self._dataset.setCurrentIndex(index)
                return

    def preflight(self) -> None:
        preset = str(self._preset.currentData() or "24g")
        try:
            check = self._client.train_preflight(preset)
        except Exception as exc:
            self._preflight = {}
            self._preflight_label.setText(f"Could not reach the check: {exc}")
            return
        self._preflight = check
        presets = check.get("presets")
        if isinstance(presets, dict) and presets:
            self._adopt_presets(presets)
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

    def _adopt_presets(self, presets: dict[str, dict[str, Any]]) -> None:
        """Build the VRAM picker from what the worker actually supports."""
        if presets == self._presets:
            return
        self._presets = presets
        current = self._preset.currentData()
        self._preset.blockSignals(True)
        self._preset.clear()
        for name, preset in sorted(
            presets.items(), key=lambda item: item[1].get("vram_floor_gb") or 0
        ):
            self._preset.addItem(
                f"{preset.get('title')} · rank {preset.get('lora_rank')}", name
            )
        if current:
            self._preset.setCurrentIndex(max(0, self._preset.findData(current)))
        self._preset.blockSignals(False)
        self._preset_changed()

    def _preset_changed(self) -> None:
        preset = self._presets.get(str(self._preset.currentData() or ""), {})
        rank = preset.get("lora_rank")
        if rank:
            self._rank.setValue(int(rank))

    def _percent(self, step: int | None, total: int) -> int:
        if not total:
            return 0
        return int(max(0.0, min(1.0, int(step or 0) / float(total))) * 1000)

    # --- the selected run ---------------------------------------------------

    def _run_selected(self) -> None:
        item = self._runs_tree.currentItem()
        self._selected_run = item.data(0, Qt.ItemDataRole.UserRole) if item else None
        self._show_run()

    def _show_run(self) -> None:
        run_id = self._selected_run
        if not run_id:
            self._log.setPlainText("")
            self._progress.setValue(0)
            self._run_status.setText("Select a run to see its log.")
            for button in (self._cancel_btn, self._install_btn, self._folder_btn):
                button.setEnabled(False)
            return
        try:
            detail = self._client.get_train_run(run_id, tail=80)
        except Exception as exc:
            self._run_status.setText(f"Could not read this run: {exc}")
            return
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

    # --- actions ------------------------------------------------------------

    def _start_run(self) -> None:
        dataset = self._dataset.currentData() or {}
        dataset_dir = str(dataset.get("path") or "")
        dataset_id = str(dataset.get("id") or "")
        if not dataset_dir or not dataset_id:
            QMessageBox.warning(
                self,
                "No dataset",
                "Pick a Music dataset — make one on the Datasets page first.",
            )
            return
        name = self._name.text().strip() or "training run"
        preset = str(self._preset.currentData() or "24g")
        try:
            check = self._client.train_preflight(preset)
        except Exception as exc:
            QMessageBox.warning(self, "Could not check requirements", str(exc))
            return
        self.preflight()
        if not check.get("ok"):
            QMessageBox.warning(
                self,
                "Not ready — nothing was started",
                str(check.get("detail") or " ".join(check.get("problems") or [])),
            )
            return
        try:
            report = self._client.validate_dataset(dataset_id)
        except Exception as exc:
            QMessageBox.warning(self, "Dataset check failed", str(exc))
            return
        if not report.get("ok"):
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
            f"Train “{name}” for {self._steps.value()} steps on {clips} clips "
            f"({self._preset.currentText()})?\n\n"
            "The run is its own process: closing Studio will not stop it. "
            "Cancel any time — it resumes from the last checkpoint.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            run = self._client.start_train_run(
                {
                    "name": name,
                    "dataset_dir": str(dataset_dir),
                    "preset": preset,
                    "steps": int(self._steps.value()),
                    "rank": int(self._rank.value()),
                    "validation": {
                        "prompt": self._validation_prompt.text().strip(),
                        "duration": int(self._validation_duration.value()),
                    },
                }
            )
        except Exception as exc:
            QMessageBox.warning(self, "Training did not start", str(exc))
            return
        self._form_status.setText(
            f"Started “{run.get('name')}” (pid {run.get('pid')}). "
            "It survives Studio closing — this page reattaches."
        )
        self._selected_run = str(run.get("id"))
        self.refresh()

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
        QMessageBox.information(
            self,
            "Adapter installed",
            f"“{row.get('name')}” is in the LoRA folder now. Pick it in the "
            "LoRA dropdown on Generate Music — around 0.8 strength is a good "
            "first audition.",
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
