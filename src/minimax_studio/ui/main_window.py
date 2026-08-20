from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QSpinBox,
    QStackedWidget,
    QStatusBar,
    QWidget,
)

from minimax_studio.ui.pages import build_pages
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
]


class MainWindow(QMainWindow):
    def __init__(self, client: WorkerClient) -> None:
        super().__init__()
        self._client = client
        self.setWindowTitle("MiniMax Studio")
        self.resize(1280, 800)

        self._stack = QStackedWidget()
        self._pages = build_pages()
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
        self._backend.addItems(["Auto", "Local", "API"])
        self._duration = QSpinBox()
        self._duration.setRange(1, 300)
        self._duration.setValue(30)
        self._duration.setSuffix(" s")
        self._seed = QSpinBox()
        self._seed.setRange(-1, 2_147_483_647)
        self._seed.setSpecialValueText("Random")
        self._seed.setValue(-1)
        self._steps = QSpinBox()
        self._steps.setRange(1, 100)
        self._steps.setValue(30)
        self._gpu_label = QLabel("Probing…")
        self._gpu_label.setWordWrap(True)

        inspector = QWidget()
        form = QFormLayout(inspector)
        form.addRow("Backend", self._backend)
        form.addRow("Duration", self._duration)
        form.addRow("Seed", self._seed)
        form.addRow("Steps", self._steps)
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
        self.refresh_probe()

    def _on_nav(self, row: int) -> None:
        if row < 0 or row >= len(self._nav_keys):
            return
        key = self._nav_keys[row]
        if key is None:
            # Snap back to the last real page.
            for i in range(row + 1, len(self._nav_keys)):
                if self._nav_keys[i] is not None:
                    self._nav.setCurrentRow(i)
                    return
            return
        self._stack.setCurrentIndex(self._page_index[key])

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
    return "\n".join(parts)
