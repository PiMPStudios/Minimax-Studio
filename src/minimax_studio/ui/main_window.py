from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QStatusBar,
    QWidget,
)

from minimax_studio import __version__
from minimax_studio.config import AppConfig
from minimax_studio.ui.pages import (
    HelpPage,
    HistoryPage,
    ModelsPage,
    MusicPage,
    PresetsPage,
    SettingsPage,
    VideoPage,
)
from minimax_studio.ui.state import StudioState
from minimax_studio.worker_client import WorkerClient

NAV_ITEMS = [
    ("Create", None),
    ("video", "Generate Video"),
    ("music", "Generate Music"),
    ("Library", None),
    ("history", "History"),
    ("presets", "Presets"),
    ("Setup", None),
    ("models", "Models"),
    ("settings", "Settings"),
    ("help", "Help"),
]


class MainWindow(QMainWindow):
    def __init__(self, client: WorkerClient, config: AppConfig) -> None:
        super().__init__()
        self._client = client
        self._config = config
        self._state = StudioState()
        self.setWindowTitle(f"MiniMax Studio {__version__}")
        self.resize(1320, 840)

        self._music = MusicPage(client, self._state)
        self._video = VideoPage(client, self._state)
        self._history = HistoryPage(client, self._state)
        self._models = ModelsPage(client)
        self._settings = SettingsPage(client, config)
        self._presets = PresetsPage(client, self._state)
        self._help = HelpPage()

        self._stack = QStackedWidget()
        self._pages = {
            "video": self._video,
            "music": self._music,
            "history": self._history,
            "presets": self._presets,
            "models": self._models,
            "settings": self._settings,
            "help": self._help,
        }
        self._page_index: dict[str, int] = {}
        for key, page in self._pages.items():
            self._page_index[key] = self._stack.addWidget(page)

        self._nav = QListWidget()
        self._nav.setFixedWidth(220)
        self._nav.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._nav_keys: list[str | None] = []
        for key, label in NAV_ITEMS:
            if label is None:
                item = QListWidgetItem(key)
                item.setFlags(Qt.ItemFlag.NoItemFlags)
                item.setForeground(Qt.GlobalColor.gray)
                self._nav.addItem(item)
                self._nav_keys.append(None)
            else:
                self._nav.addItem(QListWidgetItem(label))
                self._nav_keys.append(key)
        self._nav.currentRowChanged.connect(self._on_nav)

        splitter = QWidget()
        row = QHBoxLayout(splitter)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self._nav)
        row.addWidget(self._stack, 1)
        self.setCentralWidget(splitter)

        self._backend = QComboBox()
        self._backend.addItems(["Auto", "Local", "Comfy", "API"])
        self._backend.currentTextChanged.connect(lambda text: self._state.set_backend(text))
        self._speed = QComboBox()
        self._speed.addItems(["Quality", "Fast"])
        self._speed.currentTextChanged.connect(lambda text: self._state.set_speed(text))
        self._attention = QComboBox()
        self._attention.addItems(["Default", "Sage"])
        self._attention.currentTextChanged.connect(lambda text: self._state.set_attention(text))
        self._duration = QSpinBox()
        self._duration.setRange(1, 300)
        self._duration.setValue(self._state.duration)
        self._duration.setSuffix(" s")
        self._duration.valueChanged.connect(self._on_duration)
        self._seed = QSpinBox()
        self._seed.setRange(-1, 2_147_483_647)
        self._seed.setSpecialValueText("Random")
        self._seed.setValue(self._state.seed)
        self._seed.valueChanged.connect(self._state.set_seed)
        self._steps = QSpinBox()
        self._steps.setRange(1, 100)
        self._steps.setValue(self._state.steps)
        self._steps.valueChanged.connect(self._state.set_steps)
        self._gpu_label = QLabel("Probing…")
        self._gpu_label.setWordWrap(True)
        self._lora = QComboBox()
        self._lora.addItem("None", "")
        self._lora.currentIndexChanged.connect(self._lora_changed)
        self._lora_strength = QDoubleSpinBox()
        self._lora_strength.setRange(0.0, 2.0)
        self._lora_strength.setSingleStep(0.05)
        self._lora_strength.setValue(1.0)
        self._lora_strength.valueChanged.connect(self._lora_changed)
        import_lora = QPushButton("Import LoRA…")
        import_lora.clicked.connect(self._import_lora)
        self._state.changed.connect(self._sync_inspector)
        self._state.open_history.connect(lambda: self.show_page("history"))
        self._state.restore_music.connect(lambda _: self.show_page("music"))
        self._state.restore_video.connect(lambda _: self.show_page("video"))

        inspector = QWidget()
        form = QFormLayout(inspector)
        form.addRow("Backend", self._backend)
        form.addRow("Speed", self._speed)
        form.addRow("Attention", self._attention)
        form.addRow("Duration", self._duration)
        self._duration_hint = QLabel("")
        self._duration_hint.setObjectName("pageSubtitle")
        self._duration_hint.setWordWrap(True)
        form.addRow("", self._duration_hint)
        form.addRow("Seed", self._seed)
        form.addRow("Steps", self._steps)
        form.addRow("LoRA", self._lora)
        form.addRow("LoRA strength", self._lora_strength)
        form.addRow(import_lora)
        form.addRow("Hardware", self._gpu_label)
        brand = QLabel("MiniMax H3  ·  MiniMax-Music3")
        brand.setObjectName("brand")
        form.addRow(brand)

        dock = QDockWidget("Inspector")
        dock.setObjectName("inspectorDock")
        dock.setWidget(inspector)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        toggle = dock.toggleViewAction()
        toggle.setText("Show Inspector")
        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(toggle)

        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(quit_action)

        status = QStatusBar()
        self._status_worker = QLabel("Worker: connecting…")
        status.addWidget(self._status_worker)
        self.setStatusBar(status)

        first_page_row = next(i for i, key in enumerate(self._nav_keys) if key == "music")
        self._nav.setCurrentRow(first_page_row)
        self._apply_duration_mode("music")
        self.refresh_probe()
        self.refresh_loras()
        self._history.refresh()

        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def show_page(self, key: str) -> None:
        row = next(i for i, item in enumerate(self._nav_keys) if item == key)
        self._nav.setCurrentRow(row)
        if key == "history":
            self._history.refresh()

    def _on_nav(self, row: int) -> None:
        if row < 0 or row >= len(self._nav_keys):
            return
        key = self._nav_keys[row]
        if key is None:
            for i in range(row + 1, len(self._nav_keys)):
                if self._nav_keys[i] is not None:
                    self._nav.setCurrentRow(i)
                    return
            return
        self._stack.setCurrentIndex(self._page_index[key])
        if key == "history":
            self._history.refresh()
        if key == "models":
            self._models.refresh()
        if key == "presets":
            self._presets.refresh()
        self._apply_duration_mode(key)

    def _sync_inspector(self) -> None:
        mapping = {"auto": 0, "local": 1, "cuda": 1, "comfy": 2, "api": 3}
        self._backend.blockSignals(True)
        self._speed.blockSignals(True)
        self._attention.blockSignals(True)
        self._duration.blockSignals(True)
        self._seed.blockSignals(True)
        self._steps.blockSignals(True)
        self._backend.setCurrentIndex(mapping.get(self._state.backend, 0))
        self._speed.setCurrentIndex(0 if self._state.speed != "fast" else 1)
        self._attention.setCurrentIndex(1 if self._state.attention == "sage" else 0)
        self._duration.setValue(self._state.duration)
        self._seed.setValue(self._state.seed)
        self._steps.setValue(self._state.steps)
        self._lora.blockSignals(True)
        self._lora_strength.blockSignals(True)
        if self._state.lora_id and self._lora.findData(self._state.lora_id) < 0:
            from pathlib import Path

            self._lora.addItem(Path(self._state.lora_id).name, self._state.lora_id)
        idx = self._lora.findData(self._state.lora_id)
        self._lora.setCurrentIndex(max(0, idx))
        self._lora_strength.setValue(self._state.lora_strength)
        self._lora.blockSignals(False)
        self._lora_strength.blockSignals(False)
        self._backend.blockSignals(False)
        self._speed.blockSignals(False)
        self._attention.blockSignals(False)
        self._duration.blockSignals(False)
        self._seed.blockSignals(False)
        self._steps.blockSignals(False)

    def refresh_loras(self) -> None:
        current = self._lora.currentData()
        self._lora.blockSignals(True)
        self._lora.clear()
        self._lora.addItem("None", "")
        try:
            for item in self._client.list_loras():
                self._lora.addItem(item["name"], item["path"])
        except Exception:
            pass
        index = self._lora.findData(current)
        self._lora.setCurrentIndex(max(0, index))
        self._lora.blockSignals(False)
        self._lora_changed()

    def _lora_changed(self) -> None:
        self._state.set_lora(str(self._lora.currentData() or ""), self._lora_strength.value())

    def _import_lora(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Import LoRA", "", "LoRA (*.safetensors)"
        )
        if not chosen:
            return
        try:
            self._client.import_lora(chosen)
        except Exception:
            return
        self.refresh_loras()

    def _on_duration(self, value: int) -> None:
        self._state.set_duration(value)
        self._refresh_duration_hint()

    def _apply_duration_mode(self, key: str | None) -> None:
        self._duration.blockSignals(True)
        if key == "video":
            self._duration.setRange(5, 15)
            if self._state.duration < 5:
                self._state.set_duration(5)
            elif self._state.duration > 15:
                self._state.set_duration(15)
        else:
            self._duration.setRange(1, 300)
        self._duration.setValue(self._state.duration)
        self._duration.blockSignals(False)
        self._refresh_duration_hint(key)

    def _refresh_duration_hint(self, key: str | None = None) -> None:
        if key is None:
            row = self._nav.currentRow()
            key = self._nav_keys[row] if 0 <= row < len(self._nav_keys) else None
        if key == "video":
            from minimax_studio.h3_timing import format_h3_duration

            self._duration_hint.setText(
                "H3 snaps to " + format_h3_duration(self._state.duration)
            )
        else:
            self._duration_hint.setText("")

    def refresh_probe(self) -> None:
        try:
            health = self._client.health()
            probe = self._client.probe()
        except Exception as exc:
            self._status_worker.setText(f"Worker: down ({exc})")
            self._gpu_label.setText("Worker unreachable")
            return
        self._status_worker.setText(f"Worker: ok  v{health.get('version', '?')}")
        self._gpu_label.setText(_format_probe(probe))

    def _tick(self) -> None:
        self._music.poll()
        self._video.poll()
        row = self._nav.currentRow()
        if 0 <= row < len(self._nav_keys) and self._nav_keys[row] == "models":
            self._models.refresh()


def _format_probe(probe: dict[str, Any]) -> str:
    parts = [f"{probe.get('os')} {probe.get('machine')}"]
    if probe.get("cuda"):
        parts.append(f"{probe.get('cuda_name')} ({probe.get('vram_gb')} GB)")
    elif probe.get("apple_silicon"):
        parts.append("Apple Silicon")
    else:
        parts.append("no CUDA")
    ram = probe.get("ram_gb")
    if ram:
        parts.append(f"{ram} GB RAM")
    extras = []
    if probe.get("sageattention"):
        extras.append("SageAttention")
    if probe.get("ffmpeg"):
        extras.append("ffmpeg")
    if extras:
        parts.append(", ".join(extras))
    titles = probe.get("packs_ready_titles") or []
    if titles:
        n = len(titles)
        from_comfy = probe.get("packs_from_comfy") or 0
        extra = f", {from_comfy} from Comfy" if from_comfy else ""
        parts.append(f"{n} pack{'s' if n != 1 else ''} ready{extra}")
    return "\n".join(parts)
