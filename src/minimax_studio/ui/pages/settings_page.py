from __future__ import annotations

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from minimax_studio.config import AppConfig, save_config
from minimax_studio.worker_client import WorkerClient


class SettingsPage(QWidget):
    def __init__(self, client: WorkerClient, config: AppConfig) -> None:
        super().__init__()
        self._client = client
        self._config = config
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        title = QLabel("Settings")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        form = QFormLayout()
        self.output = QLineEdit(config.output_dir or "")
        self.models = QLineEdit(config.models_dir or "")
        self.comfy_models = QLineEdit(config.comfy_models_dir or "")
        self.comfy_models.setPlaceholderText("optional — auto-detects ComfyUI/models")
        self.hf = QLineEdit(config.hf_token or "")
        self.hf.setEchoMode(QLineEdit.EchoMode.Password)
        self.api = QLineEdit(config.minimax_api_key or "")
        self.api.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_base = QLineEdit(config.minimax_api_base)
        self.comfy_url = QLineEdit(config.comfy_url or "http://127.0.0.1:8188")
        self.cuda_device = QComboBox()
        self.cuda_device.addItem("GPU 0", 0)
        if config.cuda_device and config.cuda_device != 0:
            self.cuda_device.addItem(f"GPU {config.cuda_device}", config.cuda_device)
            self.cuda_device.setCurrentIndex(self.cuda_device.count() - 1)
        self.llm_base = QLineEdit(config.llm_base_url)
        self.llm_model = QLineEdit(config.llm_model)
        self.llm_key = QLineEdit(config.llm_api_key or "")
        self.llm_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.llm_key.setPlaceholderText("optional — falls back to ~/.config/llama-api.key")
        self._minimax_mark = _PingMark()
        self._llm_mark = _PingMark()
        self._comfy_mark = _PingMark()
        form.addRow("Output folder", _browse_row(self.output, self))
        form.addRow("Models folder", _browse_row(self.models, self))
        form.addRow("ComfyUI models", _browse_row(self.comfy_models, self))
        form.addRow("Hugging Face token", self.hf)
        form.addRow("MiniMax API key", _field_with_mark(self.api, self._minimax_mark))
        form.addRow("MiniMax API base", self.api_base)
        form.addRow("ComfyUI URL", _field_with_mark(self.comfy_url, self._comfy_mark))
        form.addRow("Studio CUDA device", self.cuda_device)
        form.addRow("Local LLM URL", _field_with_mark(self.llm_base, self._llm_mark))
        form.addRow("Local LLM model", self.llm_model)
        form.addRow("Local LLM key", self.llm_key)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        save = QPushButton("Save")
        save.setObjectName("primary")
        save.clicked.connect(self._save)
        check = QPushButton("Check connections")
        check.clicked.connect(self._refresh_pings)
        buttons.addWidget(save)
        buttons.addWidget(check)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self._save_status = QLabel("")
        self._save_status.setObjectName("pageSubtitle")
        self._save_status.setWordWrap(True)
        layout.addWidget(self._save_status)
        note = QLabel(
            "Tokens stay in the local config file. They are sent to the worker "
            "process on this machine only. MiniMax / LLM / Comfy status is shown "
            "inline — ✓ reachable, ✗ not. Studio CUDA device is for in-process "
            "diffusers only. ComfyUI uses the GPU it was launched with "
            "(--default-device)."
        )
        note.setObjectName("pageSubtitle")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)

    def showEvent(self, event) -> None:  # noqa: ANN001
        super().showEvent(event)
        QTimer.singleShot(0, self._fill_cuda_devices)
        QTimer.singleShot(0, self._refresh_pings)

    def _fill_cuda_devices(self) -> None:
        current = int(self.cuda_device.currentData() or self._config.cuda_device or 0)
        try:
            probe = self._client.probe()
        except Exception:
            return
        gpus = probe.get("gpus") or []
        self.cuda_device.blockSignals(True)
        self.cuda_device.clear()
        if not gpus:
            self.cuda_device.addItem("GPU 0", 0)
        for index, item in enumerate(gpus):
            self.cuda_device.addItem(
                f"{index}: {item.get('name')} ({item.get('vram_gb')} GB)",
                index,
            )
        idx = self.cuda_device.findData(current)
        self.cuda_device.setCurrentIndex(max(0, idx))
        self.cuda_device.blockSignals(False)

    def _save(self) -> None:
        payload = {
            "output_dir": self.output.text().strip() or None,
            "models_dir": self.models.text().strip() or None,
            "comfy_models_dir": self.comfy_models.text().strip(),
            "hf_token": self.hf.text().strip(),
            "minimax_api_key": self.api.text().strip(),
            "minimax_api_base": self.api_base.text().strip() or "https://api.minimax.io",
            "comfy_url": self.comfy_url.text().strip() or "http://127.0.0.1:8188",
            "cuda_device": int(self.cuda_device.currentData() or 0),
            "llm_base_url": self.llm_base.text().strip() or "http://127.0.0.1:8080/v1",
            "llm_model": self.llm_model.text().strip() or "qwen3.8-27b-q4kxl",
            "llm_api_key": self.llm_key.text().strip(),
        }
        try:
            saved = self._client.put_settings(payload)
            updated = AppConfig.model_validate(saved)
            save_config(updated)
            self._config.output_dir = updated.output_dir
            self._config.models_dir = updated.models_dir
            self._config.comfy_models_dir = updated.comfy_models_dir
            self._config.hf_token = updated.hf_token
            self._config.minimax_api_key = updated.minimax_api_key
            self._config.minimax_api_base = updated.minimax_api_base
            self._config.comfy_url = updated.comfy_url
            self._config.cuda_device = updated.cuda_device
            self._config.llm_base_url = updated.llm_base_url
            self._config.llm_model = updated.llm_model
            self._config.llm_api_key = updated.llm_api_key
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return
        self._save_status.setText("Saved. Checking connections…")
        self._refresh_pings()

    def _refresh_pings(self) -> None:
        class Worker(QObject):
            finished = Signal(dict)
            failed = Signal(str)

            def run(inner_self) -> None:
                try:
                    inner_self.finished.emit(self._client.ping())
                except Exception as exc:
                    inner_self.failed.emit(str(exc))

        thread = QThread(self)
        worker = Worker()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        def done(ping: dict) -> None:
            self._apply_ping(self._minimax_mark, ping.get("minimax"))
            self._apply_ping(self._llm_mark, ping.get("llm"))
            self._apply_ping(self._comfy_mark, ping.get("comfy"))
            if self._save_status.text().startswith("Saved"):
                self._save_status.setText("Saved.")

        def fail(err: str) -> None:
            self._minimax_mark.set_state(False, err)
            self._llm_mark.set_state(False, err)
            self._comfy_mark.set_state(False, err)
            if self._save_status.text().startswith("Saved"):
                self._save_status.setText(err)

        worker.finished.connect(done)
        worker.failed.connect(fail)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.start()
        self._ping_thread = thread

    @staticmethod
    def _apply_ping(mark: "_PingMark", result: object) -> None:
        if not isinstance(result, dict):
            mark.set_state(False, "no result")
            return
        mark.set_state(bool(result.get("ok")), str(result.get("detail") or ""))


class _PingMark(QLabel):
    def __init__(self) -> None:
        super().__init__("—")
        self.setObjectName("idleMark")
        self.setFixedWidth(22)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_state(self, ok: bool, detail: str) -> None:
        self.setText("✓" if ok else "✗")
        self.setObjectName("okMark" if ok else "failMark")
        self.setToolTip(detail)
        style = self.style()
        if style is not None:
            style.unpolish(self)
            style.polish(self)


def _field_with_mark(edit: QLineEdit, mark: QLabel) -> QWidget:
    wrap = QWidget()
    row = QHBoxLayout(wrap)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(edit, 1)
    row.addWidget(mark)
    return wrap


def _browse_row(edit: QLineEdit, parent: QWidget) -> QWidget:
    wrap = QWidget()
    row = QHBoxLayout(wrap)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(edit, 1)
    button = QPushButton("Browse…")

    def choose() -> None:
        chosen = QFileDialog.getExistingDirectory(parent, "Choose folder")
        if chosen:
            edit.setText(chosen)

    button.clicked.connect(choose)
    row.addWidget(button)
    return wrap
