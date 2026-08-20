from __future__ import annotations

from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
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
        self.hf = QLineEdit(config.hf_token or "")
        self.hf.setEchoMode(QLineEdit.EchoMode.Password)
        self.api = QLineEdit(config.minimax_api_key or "")
        self.api.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_base = QLineEdit(config.minimax_api_base)
        self.llm_base = QLineEdit(config.llm_base_url)
        self.llm_model = QLineEdit(config.llm_model)
        self.llm_key = QLineEdit(config.llm_api_key or "")
        self.llm_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.llm_key.setPlaceholderText("optional — falls back to ~/.config/llama-api.key")
        form.addRow("Output folder", _browse_row(self.output, self))
        form.addRow("Models folder", _browse_row(self.models, self))
        form.addRow("Hugging Face token", self.hf)
        form.addRow("MiniMax API key", self.api)
        form.addRow("MiniMax API base", self.api_base)
        form.addRow("Local LLM URL", self.llm_base)
        form.addRow("Local LLM model", self.llm_model)
        form.addRow("Local LLM key", self.llm_key)
        layout.addLayout(form)
        save = QPushButton("Save")
        save.setObjectName("primary")
        save.clicked.connect(self._save)
        layout.addWidget(save)
        note = QLabel(
            "Tokens stay in the local config file. They are sent to the worker "
            "process on this machine only."
        )
        note.setObjectName("pageSubtitle")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)

    def _save(self) -> None:
        payload = {
            "output_dir": self.output.text().strip() or None,
            "models_dir": self.models.text().strip() or None,
            "hf_token": self.hf.text().strip() or None,
            "minimax_api_key": self.api.text().strip() or None,
            "minimax_api_base": self.api_base.text().strip() or "https://api.minimax.io",
            "llm_base_url": self.llm_base.text().strip() or "http://127.0.0.1:8080/v1",
            "llm_model": self.llm_model.text().strip() or "qwen3.8-27b-q4kxl",
            "llm_api_key": self.llm_key.text().strip() or None,
        }
        try:
            saved = self._client.put_settings(payload)
            updated = AppConfig.model_validate(saved)
            save_config(updated)
            self._config.output_dir = updated.output_dir
            self._config.models_dir = updated.models_dir
            self._config.hf_token = updated.hf_token
            self._config.minimax_api_key = updated.minimax_api_key
            self._config.minimax_api_base = updated.minimax_api_base
            self._config.llm_base_url = updated.llm_base_url
            self._config.llm_model = updated.llm_model
            self._config.llm_api_key = updated.llm_api_key
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return
        QMessageBox.information(self, "Saved", "Settings written for the app and worker.")


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
