from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from minimax_studio.ui.state import StudioState
from minimax_studio.worker_client import WorkerClient

MODES = [
    ("t2va", "Text"),
    ("i2va", "First frame"),
    ("l2va", "Last frame"),
    ("fl2va", "First + last"),
    ("ref2va", "References"),
]


class VideoPage(QWidget):
    def __init__(self, client: WorkerClient, state: StudioState) -> None:
        super().__init__()
        self._client = client
        self._state = state
        self._job_id: str | None = None
        self._mode = "t2va"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        title = QLabel("Generate Video")
        title.setObjectName("pageTitle")
        brand = QLabel("MiniMax H3")
        brand.setObjectName("brand")
        sub = QLabel(
            "One form, no wires. Pick a mode and drop images, video, or audio. "
            "Local generate uses official diffusers in-process, or Comfy-Org INT8 "
            "via a running ComfyUI."
        )
        sub.setObjectName("pageSubtitle")
        sub.setWordWrap(True)
        enhance = QPushButton("Enhance prompt")
        enhance.clicked.connect(self._enhance)
        self._enhance_btn = enhance
        ir = QPushButton("Context-IR (API)")
        ir.clicked.connect(self._context_ir)
        self._ir_btn = ir
        header = QHBoxLayout()
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(enhance)
        header.addWidget(ir)
        layout.addLayout(header)
        layout.addWidget(brand)
        layout.addWidget(sub)

        mode_row = QHBoxLayout()
        self._mode_group = QButtonGroup(self)
        for value, label in MODES:
            button = QRadioButton(label)
            if value == "t2va":
                button.setChecked(True)
            self._mode_group.addButton(button)
            button.toggled.connect(lambda checked, v=value: checked and self._set_mode(v))
            mode_row.addWidget(button)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText(_placeholder("t2va"))
        layout.addWidget(self.prompt, 1)

        self._assets = QVBoxLayout()
        self.first_path = _AssetRow("First frame", "image")
        self.last_path = _AssetRow("Last frame", "image")
        self.refs_path = _AssetRow("References (images / video / audio)", "any")
        self._assets.addWidget(self.first_path)
        self._assets.addWidget(self.last_path)
        self._assets.addWidget(self.refs_path)
        layout.addLayout(self._assets)

        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("API resolution"))
        self.resolution = QComboBox()
        self.resolution.addItems(["768P", "2K"])
        res_row.addWidget(self.resolution)
        res_row.addWidget(QLabel("Ratio"))
        self.ratio = QComboBox()
        self.ratio.addItems(["16:9", "9:16", "1:1", "4:3", "21:9"])
        res_row.addWidget(self.ratio)
        self._ref_size_label = QLabel("Ref size")
        self.ref_size = QComboBox()
        self.ref_size.addItems(["match", "max"])
        self.ref_size.setToolTip(
            "match scales references to the output size (faster). "
            "max keeps up to a 2048px short edge (stronger identity)."
        )
        res_row.addWidget(self._ref_size_label)
        res_row.addWidget(self.ref_size)
        res_row.addStretch(1)
        layout.addLayout(res_row)
        self._set_mode("t2va")

        run_row = QHBoxLayout()
        self.generate = QPushButton("Generate")
        self.generate.setObjectName("primary")
        self.generate.clicked.connect(self._generate)
        save = QPushButton("Save preset")
        save.clicked.connect(self._save_preset)
        self._cancel = QPushButton("Cancel")
        self._cancel.setEnabled(False)
        self._cancel.clicked.connect(self._cancel_job)
        run_row.addWidget(self.generate)
        run_row.addWidget(self._cancel)
        run_row.addWidget(save)
        self._status = QLabel("")
        self._status.setObjectName("pageSubtitle")
        run_row.addWidget(self._status, 1)
        layout.addLayout(run_row)
        self._bar = QProgressBar()
        self._bar.hide()
        layout.addWidget(self._bar)
        state.restore_video.connect(self.apply_restore)

    def _set_mode(self, mode: str) -> None:
        self._mode = mode
        self.first_path.setVisible(mode in {"i2va", "fl2va"})
        self.last_path.setVisible(mode in {"l2va", "fl2va"})
        self.refs_path.setVisible(mode == "ref2va")
        self._ref_size_label.setVisible(mode == "ref2va")
        self.ref_size.setVisible(mode == "ref2va")
        self.prompt.setPlaceholderText(_placeholder(mode))

    def poll(self) -> None:
        if not self._job_id:
            return
        try:
            job = self._client.get_job(self._job_id)
        except Exception as exc:
            self._status.setText(str(exc))
            self._job_id = None
            self.generate.setEnabled(True)
            self._cancel.setEnabled(False)
            return
        self._bar.show()
        self._bar.setValue(int(float(job.get("progress") or 0) * 100))
        self._status.setText(str(job.get("error") or job.get("message") or job.get("status")))
        if job.get("status") in {"done", "error", "cancelled"}:
            self._job_id = None
            self.generate.setEnabled(True)
            self._cancel.setEnabled(False)

    def _generate(self) -> None:
        prompt = self.prompt.toPlainText().strip()
        if not prompt:
            QMessageBox.information(self, "Prompt needed", "Describe the shot first.")
            return
        assets = []
        if self.first_path.isVisible() and self.first_path.path:
            assets.append({"role": "first_frame", "path": self.first_path.path})
        if self.last_path.isVisible() and self.last_path.path:
            assets.append({"role": "last_frame", "path": self.last_path.path})
        if self.refs_path.isVisible():
            for path in self.refs_path.paths:
                assets.append({"role": "reference", "path": path})
        payload = {
            "kind": "h3",
            "backend": self._state.backend,
            "mode": self._mode,
            "prompt": prompt,
            "duration_s": min(self._state.duration, 15),
            "seed": self._state.seed,
            "steps": self._state.steps,
            "assets": assets,
            "speed": self._state.speed,
            "attention": self._state.attention,
            "resolution": self.resolution.currentText(),
            "ratio": self.ratio.currentText(),
            "ref_image_size": self.ref_size.currentText(),
            "loras": (
                [{"id": self._state.lora_id, "strength": self._state.lora_strength}]
                if self._state.lora_id
                else []
            ),
        }
        try:
            job = self._client.start_job(payload)
        except Exception as exc:
            QMessageBox.warning(self, "Generate failed", str(exc))
            return
        self._job_id = str(job["id"])
        self.generate.setEnabled(False)
        self._cancel.setEnabled(True)
        self._bar.show()
        self._status.setText("Queued")

    def _cancel_job(self) -> None:
        if not self._job_id:
            return
        try:
            self._client.cancel_job(self._job_id)
        except Exception as exc:
            self._status.setText(str(exc))

    def apply_restore(self, entry: dict) -> None:
        self.prompt.setPlainText(str(entry.get("prompt") or ""))
        mode = str(entry.get("mode") or "t2va")
        for button in self._mode_group.buttons():
            if button.text() == dict(MODES).get(mode, ""):
                button.setChecked(True)
        self._set_mode(mode)
        if entry.get("resolution"):
            self.resolution.setCurrentText(str(entry["resolution"]))
        if entry.get("ratio"):
            self.ratio.setCurrentText(str(entry["ratio"]))
        if entry.get("duration_s"):
            self._state.set_duration(int(entry["duration_s"]))
        if entry.get("seed") is not None:
            self._state.set_seed(int(entry["seed"]))
        if entry.get("steps"):
            self._state.set_steps(int(entry["steps"]))
        if entry.get("backend"):
            self._state.set_backend(str(entry["backend"]))
        if entry.get("speed"):
            self._state.set_speed(str(entry["speed"]))
        if entry.get("attention"):
            self._state.set_attention(str(entry["attention"]))
        if entry.get("ref_image_size"):
            self.ref_size.setCurrentText(str(entry["ref_image_size"]))
        loras = entry.get("loras") or []
        lora_id = entry.get("lora_id") or (loras[0].get("id") if loras else "")
        strength = entry.get("lora_strength")
        if strength is None and loras:
            strength = loras[0].get("strength", 1.0)
        if lora_id:
            self._state.set_lora(str(lora_id), float(strength or 1.0))
        assets = entry.get("assets") or []
        first = next((a.get("path") for a in assets if a.get("role") == "first_frame"), "")
        last = next((a.get("path") for a in assets if a.get("role") == "last_frame"), "")
        refs = [a.get("path") for a in assets if a.get("role") == "reference" and a.get("path")]
        if first:
            self.first_path.set_paths(str(first))
        if last:
            self.last_path.set_paths(str(last))
        if refs:
            self.refs_path.set_paths([str(p) for p in refs])

    def _save_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "Save preset", "Name")
        if not ok or not name.strip():
            return
        try:
            self._client.save_preset(
                {
                    "name": name.strip(),
                    "kind": "h3",
                    "backend": self._state.backend,
                    "mode": self._mode,
                    "prompt": self.prompt.toPlainText(),
                    "duration_s": self._state.duration,
                    "seed": self._state.seed,
                    "steps": self._state.steps,
                    "resolution": self.resolution.currentText(),
                    "ratio": self.ratio.currentText(),
                    "speed": self._state.speed,
                    "attention": self._state.attention,
                    "ref_image_size": self.ref_size.currentText(),
                    "backend": self._state.backend,
                    "lora_id": self._state.lora_id,
                    "lora_strength": self._state.lora_strength,
                    "loras": (
                        [{"id": self._state.lora_id, "strength": self._state.lora_strength}]
                        if self._state.lora_id
                        else []
                    ),
                    "assets": [
                        item
                        for item in [
                            {"role": "first_frame", "path": self.first_path.path}
                            if self.first_path.path
                            else None,
                            {"role": "last_frame", "path": self.last_path.path}
                            if self.last_path.path
                            else None,
                        ]
                        if item
                    ]
                    + [{"role": "reference", "path": p} for p in self.refs_path.paths],
                }
            )
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return
        self._status.setText(f"Saved preset “{name.strip()}”")

    def _enhance(self) -> None:
        from minimax_studio.ui.enhance import start_enhance

        seed = self.prompt.toPlainText().strip()
        if not seed:
            QMessageBox.information(self, "Prompt needed", "Write a short idea first.")
            return
        self._enhance_btn.setEnabled(False)
        self._status.setText("Enhancing with local LLM…")

        def done(text: str) -> None:
            self._enhance_btn.setEnabled(True)
            if text:
                self.prompt.setPlainText(text)
            self._status.setText("Prompt enhanced")

        def fail(err: str) -> None:
            self._enhance_btn.setEnabled(True)
            self._status.setText(err)

        start_enhance(self, self._client, "h3", seed, "", done, fail)

    def _context_ir(self) -> None:
        from PySide6.QtCore import QThread, QObject, Signal

        seed = self.prompt.toPlainText().strip()
        if not seed:
            QMessageBox.information(self, "Prompt needed", "Write a short idea first.")
            return
        assets = []
        if self.first_path.isVisible() and self.first_path.path:
            assets.append({"role": "first_frame", "path": self.first_path.path})
        if self.last_path.isVisible() and self.last_path.path:
            assets.append({"role": "last_frame", "path": self.last_path.path})
        if self.refs_path.isVisible():
            for path in self.refs_path.paths:
                assets.append({"role": "reference", "path": path})
        self._ir_btn.setEnabled(False)
        self._status.setText("Running MiniMax H3 Context-IR…")

        class Worker(QObject):
            finished = Signal(str)
            failed = Signal(str)

            def run(inner_self) -> None:
                try:
                    payload = self._client.context_ir(
                        prompt=seed,
                        mode=self._mode,
                        duration_s=min(self._state.duration, 15),
                        ratio=self.ratio.currentText(),
                        assets=assets,
                    )
                    inner_self.finished.emit(str(payload.get("text") or ""))
                except Exception as exc:
                    inner_self.failed.emit(str(exc))

        thread = QThread(self)
        worker = Worker()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        def done(text: str) -> None:
            self._ir_btn.setEnabled(True)
            if text:
                self.prompt.setPlainText(text)
            self._status.setText("Context-IR prompt ready")

        def fail(err: str) -> None:
            self._ir_btn.setEnabled(True)
            self._status.setText(err)

        worker.finished.connect(done)
        worker.failed.connect(fail)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.start()
        self._ir_thread = thread


def _placeholder(mode: str) -> str:
    if mode == "ref2va":
        return (
            "Name references in order: <Picture 1>, <Video 1>, <Audio 1>. "
            "Say which file drives identity, style, motion, or voice."
        )
    return "Cinematic medium shot. Describe shots, camera, dialogue, SFX, and music."


class _AssetRow(QWidget):
    def __init__(self, label: str, kind: str) -> None:
        super().__init__()
        self._kind = kind
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel(label))
        self._edit = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row.addWidget(self._edit, 1)
        row.addWidget(browse)

    @property
    def path(self) -> str:
        text = self._edit.text().strip()
        if self._kind == "image":
            return text
        return text.split(";")[0] if text else ""

    @property
    def paths(self) -> list[str]:
        return [part.strip() for part in self._edit.text().split(";") if part.strip()]

    def set_paths(self, paths: list[str] | str) -> None:
        if isinstance(paths, str):
            self._edit.setText(paths)
        else:
            self._edit.setText(";".join(paths))

    def _browse(self) -> None:
        if self._kind == "image":
            chosen, _ = QFileDialog.getOpenFileName(
                self, "Choose image", "", "Images (*.png *.jpg *.jpeg *.webp)"
            )
            if chosen:
                self._edit.setText(str(Path(chosen)))
            return
        chosen, _ = QFileDialog.getOpenFileNames(
            self,
            "Choose references",
            "",
            "Media (*.png *.jpg *.jpeg *.webp *.mp4 *.mov *.wav *.mp3)",
        )
        if chosen:
            self._edit.setText(";".join(str(Path(item)) for item in chosen))
