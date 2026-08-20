from __future__ import annotations

from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from minimax_studio.ui.state import StudioState
from minimax_studio.worker_client import WorkerClient

TAGS = [
    "[Intro]",
    "[Verse]",
    "[Pre-Chorus]",
    "[Chorus]",
    "[Bridge]",
    "[Instrumental]",
    "[Outro]",
]


class MusicPage(QWidget):
    def __init__(self, client: WorkerClient, state: StudioState) -> None:
        super().__init__()
        self._client = client
        self._state = state
        self._job_id: str | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        title = QLabel("Generate Music")
        title.setObjectName("pageTitle")
        brand = QLabel("MiniMax-Music3")
        brand.setObjectName("brand")
        sub = QLabel(
            "Caption describes the song. Lyrics use section tags on their own lines. "
            "Duration, seed, and steps live in the inspector."
        )
        sub.setObjectName("pageSubtitle")
        sub.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(brand)
        layout.addWidget(sub)

        layout.addWidget(QLabel("Global metadata"))
        self.global_box = QPlainTextEdit()
        self.global_box.setPlaceholderText(
            "Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate."
        )
        self.global_box.setFixedHeight(72)
        layout.addWidget(self.global_box)

        layout.addWidget(QLabel("Vocal details"))
        self.vocal_box = QPlainTextEdit()
        self.vocal_box.setPlaceholderText(
            "Soft female lead, close and breathy, light stacked harmonies."
        )
        self.vocal_box.setFixedHeight(64)
        layout.addWidget(self.vocal_box)

        layout.addWidget(QLabel("Arrangement"))
        self.arrange_box = QPlainTextEdit()
        self.arrange_box.setPlaceholderText(
            "Fingerpicked guitar and soft piano; brushed drums enter in the chorus."
        )
        self.arrange_box.setFixedHeight(64)
        layout.addWidget(self.arrange_box)

        layout.addWidget(QLabel("Lyrics"))
        tag_row = QHBoxLayout()
        for tag in TAGS:
            button = QPushButton(tag)
            button.clicked.connect(lambda _, t=tag: self._insert_tag(t))
            tag_row.addWidget(button)
        tag_row.addStretch(1)
        layout.addLayout(tag_row)
        self.lyrics = QPlainTextEdit()
        self.lyrics.setPlaceholderText("[Verse]\nMorning light filtering through the pine\n[Chorus]\nSoftly the world begins to breathe")
        layout.addWidget(self.lyrics, 1)

        run_row = QHBoxLayout()
        self.generate = QPushButton("Generate")
        self.generate.setObjectName("primary")
        self.generate.clicked.connect(self._generate)
        save = QPushButton("Save preset")
        save.clicked.connect(self._save_preset)
        self._status = QLabel("")
        self._status.setObjectName("pageSubtitle")
        run_row.addWidget(self.generate)
        run_row.addWidget(save)
        run_row.addWidget(self._status, 1)
        layout.addLayout(run_row)
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.hide()
        layout.addWidget(self._bar)
        state.restore_music.connect(self.apply_restore)

    def caption(self) -> str:
        parts = [
            self.global_box.toPlainText().strip(),
            self.vocal_box.toPlainText().strip(),
            self.arrange_box.toPlainText().strip(),
        ]
        return " ".join(part for part in parts if part)

    def apply_restore(self, entry: dict) -> None:
        prompt = str(entry.get("prompt") or "")
        self.global_box.setPlainText(prompt)
        self.vocal_box.clear()
        self.arrange_box.clear()
        self.lyrics.setPlainText(str(entry.get("lyrics") or ""))
        if entry.get("duration_s"):
            self._state.set_duration(int(entry["duration_s"]))
        if entry.get("seed") is not None:
            self._state.set_seed(int(entry["seed"]))
        if entry.get("steps"):
            self._state.set_steps(int(entry["steps"]))

    def poll(self) -> None:
        if not self._job_id:
            return
        try:
            job = self._client.get_job(self._job_id)
        except Exception as exc:
            self._status.setText(str(exc))
            self._job_id = None
            self.generate.setEnabled(True)
            return
        status = job.get("status")
        progress = float(job.get("progress") or 0)
        self._bar.show()
        self._bar.setValue(int(progress * 100))
        self._status.setText(str(job.get("message") or status))
        if status == "done":
            self._job_id = None
            self.generate.setEnabled(True)
            self._status.setText(f"Saved {job.get('output_path')}")
            self._state.open_history.emit()
        elif status == "error":
            self._job_id = None
            self.generate.setEnabled(True)
            self._status.setText(str(job.get("error") or "Failed"))

    def _insert_tag(self, tag: str) -> None:
        cursor = self.lyrics.textCursor()
        cursor.insertText(tag + "\n")
        self.lyrics.setTextCursor(cursor)
        self.lyrics.setFocus()

    def _generate(self) -> None:
        prompt = self.caption()
        lyrics = self.lyrics.toPlainText().strip()
        if not prompt:
            QMessageBox.information(self, "Caption needed", "Describe the song first.")
            return
        payload = {
            "kind": "music",
            "backend": self._state.backend,
            "mode": "ttm",
            "prompt": prompt,
            "lyrics": lyrics,
            "duration_s": self._state.duration,
            "seed": self._state.seed,
            "steps": self._state.steps,
        }
        try:
            job = self._client.start_job(payload)
        except Exception as exc:
            QMessageBox.warning(self, "Generate failed", str(exc))
            return
        self._job_id = str(job["id"])
        self.generate.setEnabled(False)
        self._bar.show()
        self._bar.setValue(0)
        self._status.setText("Queued")

    def _save_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "Save preset", "Name")
        if not ok or not name.strip():
            return
        try:
            self._client.save_preset(
                {
                    "name": name.strip(),
                    "kind": "music",
                    "backend": self._state.backend,
                    "mode": "ttm",
                    "prompt": self.caption(),
                    "lyrics": self.lyrics.toPlainText(),
                    "duration_s": self._state.duration,
                    "seed": self._state.seed,
                    "steps": self._state.steps,
                }
            )
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return
        self._status.setText(f"Saved preset “{name.strip()}”")
