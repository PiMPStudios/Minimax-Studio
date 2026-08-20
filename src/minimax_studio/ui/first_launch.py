from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)


class FirstLaunchDialog(QDialog):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MiniMax Studio")
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        title = QLabel("Choose an output folder")
        title.setObjectName("pageTitle")
        body = QLabel(
            "Generations, history, and downloaded models live here. "
            "Weights are never bundled with the app. If you already have Comfy-Org "
            "MiniMax packs, Studio will find them — you do not need to download again."
        )
        body.setObjectName("pageSubtitle")
        body.setWordWrap(True)
        self._path = QLineEdit()
        self._path.setPlaceholderText("Select a folder with plenty of free disk…")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(title)
        layout.addWidget(body)
        layout.addWidget(self._path)
        layout.addWidget(browse)
        layout.addWidget(buttons)

    @property
    def output_dir(self) -> str:
        return self._path.text().strip()

    def _browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Output folder")
        if chosen:
            self._path.setText(chosen)

    def _accept(self) -> None:
        path = Path(self.output_dir)
        if not self.output_dir:
            return
        path.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            self.accept()
