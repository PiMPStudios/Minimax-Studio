from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtGui import QAction, QKeySequence
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
        self._route_thread: QThread | None = None
        self._route_ticks = 0
        self._models_ticks = 0
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
        self._backend.currentTextChanged.connect(lambda _t: self._refresh_route())
        self._speed = QComboBox()
        self._speed.addItems(["Quality", "Fast"])
        self._speed.setToolTip(
            "H3 Fast needs the Turbo LoRA on Models (8 steps, 4 for Ref2VA). "
            "Music ignores this."
        )
        self._speed.currentTextChanged.connect(self._on_speed)
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
        self._cfg = QDoubleSpinBox()
        self._cfg.setRange(0.0, 20.0)
        self._cfg.setSingleStep(0.1)
        self._cfg.setValue(self._state.cfg)
        self._cfg.setToolTip("Music CFG / guidance. H3 local sampling does not use this.")
        self._cfg.valueChanged.connect(self._state.set_cfg)
        self._gpu_pick = QComboBox()
        self._gpu_pick.addItem("GPU 0", 0)
        self._gpu_pick.currentIndexChanged.connect(self._gpu_changed)
        self._gpu_label = QLabel("Probing…")
        self._gpu_label.setWordWrap(True)
        self._lora = QComboBox()
        self._lora.addItem("None", "")
        self._lora.setToolTip(
            "Community MiniMax adapters are still rare. Import a .safetensors to apply."
        )
        self._lora.currentIndexChanged.connect(self._lora_changed)
        self._lora_strength = QDoubleSpinBox()
        self._lora_strength.setRange(0.0, 2.0)
        self._lora_strength.setSingleStep(0.05)
        self._lora_strength.setValue(1.0)
        self._lora_strength.valueChanged.connect(self._lora_changed)
        self._lora2 = QComboBox()
        self._lora2.addItem("None", "")
        self._lora2.setToolTip("Optional second adapter, stacked after LoRA 1 (and after Turbo when Fast).")
        self._lora2.currentIndexChanged.connect(self._lora_changed)
        self._lora2_strength = QDoubleSpinBox()
        self._lora2_strength.setRange(0.0, 2.0)
        self._lora2_strength.setSingleStep(0.05)
        self._lora2_strength.setValue(1.0)
        self._lora2_strength.valueChanged.connect(self._lora_changed)
        import_lora = QPushButton("Import LoRA…")
        import_lora.clicked.connect(self._import_lora)
        self._state.changed.connect(self._sync_inspector)
        self._state.open_history.connect(lambda: self.show_page("history"))
        self._state.restore_music.connect(lambda _: self.show_page("music"))
        self._state.restore_video.connect(lambda _: self.show_page("video"))

        inspector = QWidget()
        form = QFormLayout(inspector)
        form.addRow("Backend", self._backend)
        self._route = QLabel("Will use: …")
        self._route.setObjectName("pageSubtitle")
        self._route.setWordWrap(True)
        form.addRow("", self._route)
        form.addRow("Speed", self._speed)
        form.addRow("Attention", self._attention)
        form.addRow("Duration", self._duration)
        self._duration_hint = QLabel("")
        self._duration_hint.setObjectName("pageSubtitle")
        self._duration_hint.setWordWrap(True)
        form.addRow("", self._duration_hint)
        form.addRow("Seed", self._seed)
        form.addRow("Steps", self._steps)
        form.addRow("CFG", self._cfg)
        form.addRow("LoRA", self._lora)
        form.addRow("LoRA strength", self._lora_strength)
        form.addRow("LoRA 2", self._lora2)
        form.addRow("LoRA 2 strength", self._lora2_strength)
        form.addRow(import_lora)
        form.addRow("CUDA GPU", self._gpu_pick)
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

        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        open_out = QAction("Open Output Folder", self)
        open_out.triggered.connect(lambda: self._open_folder(self._config.output_dir))
        open_models = QAction("Open Models Folder", self)
        open_models.triggered.connect(lambda: self._open_folder(self._config.models_dir))
        open_log = QAction("Open Comfy Log", self)
        open_log.triggered.connect(self._open_comfy_log)
        open_comfy = QAction("Open ComfyUI in Browser", self)
        open_comfy.triggered.connect(self._open_comfy_browser)
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(open_out)
        file_menu.addAction(open_models)
        file_menu.addAction(open_log)
        file_menu.addAction(open_comfy)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

        toggle = dock.toggleViewAction()
        toggle.setText("Show Inspector")
        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(toggle)
        welcome_again = QAction("Setup checklist…", self)
        welcome_again.triggered.connect(self._show_welcome)
        view_menu.addAction(welcome_again)
        self._add_go_menu()

        status = QStatusBar()
        self._status_worker = QLabel("Worker: connecting…")
        self._status_job = QLabel("Job: idle")
        self._status_cancel = QPushButton("Cancel")
        self._status_cancel.setEnabled(False)
        self._status_cancel.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._status_cancel.setFlat(True)
        self._status_cancel.clicked.connect(self._cancel_active_job)
        self._status_dl = QLabel("")
        self._active_job_id: str | None = None
        status.addWidget(self._status_worker)
        status.addPermanentWidget(self._status_dl)
        status.addPermanentWidget(self._status_job)
        status.addPermanentWidget(self._status_cancel)
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
        QTimer.singleShot(0, self._refresh_route)
        QTimer.singleShot(0, self._refresh_job_status)

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
        self._refresh_route()

    def _sync_inspector(self) -> None:
        mapping = {"auto": 0, "local": 1, "cuda": 1, "comfy": 2, "api": 3}
        self._backend.blockSignals(True)
        self._speed.blockSignals(True)
        self._attention.blockSignals(True)
        self._duration.blockSignals(True)
        self._seed.blockSignals(True)
        self._steps.blockSignals(True)
        self._cfg.blockSignals(True)
        self._backend.setCurrentIndex(mapping.get(self._state.backend, 0))
        self._speed.setCurrentIndex(0 if self._state.speed != "fast" else 1)
        self._attention.setCurrentIndex(1 if self._state.attention == "sage" else 0)
        self._duration.setValue(self._state.duration)
        self._seed.setValue(self._state.seed)
        self._steps.setValue(self._state.steps)
        self._cfg.setValue(self._state.cfg)
        self._lora.blockSignals(True)
        self._lora_strength.blockSignals(True)
        self._lora2.blockSignals(True)
        self._lora2_strength.blockSignals(True)
        from pathlib import Path

        if self._state.lora_id and self._lora.findData(self._state.lora_id) < 0:
            self._lora.addItem(Path(self._state.lora_id).name, self._state.lora_id)
        idx = self._lora.findData(self._state.lora_id)
        self._lora.setCurrentIndex(max(0, idx))
        self._lora_strength.setValue(self._state.lora_strength)
        if self._state.lora2_id and self._lora2.findData(self._state.lora2_id) < 0:
            self._lora2.addItem(Path(self._state.lora2_id).name, self._state.lora2_id)
        idx2 = self._lora2.findData(self._state.lora2_id)
        self._lora2.setCurrentIndex(max(0, idx2))
        self._lora2_strength.setValue(self._state.lora2_strength)
        self._lora.blockSignals(False)
        self._lora_strength.blockSignals(False)
        self._lora2.blockSignals(False)
        self._lora2_strength.blockSignals(False)
        self._backend.blockSignals(False)
        self._speed.blockSignals(False)
        self._attention.blockSignals(False)
        self._duration.blockSignals(False)
        self._seed.blockSignals(False)
        self._steps.blockSignals(False)
        self._cfg.blockSignals(False)

    def refresh_loras(self) -> None:
        current = self._lora.currentData()
        current2 = self._lora2.currentData()
        self._lora.blockSignals(True)
        self._lora2.blockSignals(True)
        self._lora.clear()
        self._lora2.clear()
        self._lora.addItem("None", "")
        self._lora2.addItem("None", "")
        try:
            for item in self._client.list_loras():
                self._lora.addItem(item["name"], item["path"])
                self._lora2.addItem(item["name"], item["path"])
        except Exception:
            pass
        self._lora.setCurrentIndex(max(0, self._lora.findData(current)))
        self._lora2.setCurrentIndex(max(0, self._lora2.findData(current2)))
        self._lora.blockSignals(False)
        self._lora2.blockSignals(False)
        self._lora_changed()

    def _lora_changed(self) -> None:
        self._state.set_lora(str(self._lora.currentData() or ""), self._lora_strength.value())
        self._state.set_lora2(str(self._lora2.currentData() or ""), self._lora2_strength.value())

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

    def _on_speed(self, text: str) -> None:
        self._state.set_speed(text)
        row = self._nav.currentRow()
        key = self._nav_keys[row] if 0 <= row < len(self._nav_keys) else None
        if text.lower() == "fast" and key == "video" and self._state.steps >= 16:
            self._state.set_steps(8)
        self._refresh_route()

    def _on_duration(self, value: int) -> None:
        self._state.set_duration(value)
        self._refresh_duration_hint()

    def _apply_duration_mode(self, key: str | None) -> None:
        self._speed.setEnabled(key == "video")
        if key == "video":
            self._speed.setToolTip(
                "H3 Fast needs the Turbo LoRA on Models (8 steps, 4 for Ref2VA)."
            )
        else:
            self._speed.setToolTip("Fast is H3-only (Turbo LoRA). Music ignores Speed.")
        self._duration.blockSignals(True)
        if key == "video":
            self._duration.setRange(5, 15)
            if self._state.duration < 5 or self._state.duration > 15:
                self._state.set_duration(8)
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
        self._fill_gpu_pick(probe.get("gpus") or [])

    def _fill_gpu_pick(self, gpus: list) -> None:
        current = int(self._config.cuda_device or 0)
        self._gpu_pick.blockSignals(True)
        self._gpu_pick.clear()
        if not gpus:
            self._gpu_pick.addItem("GPU 0", 0)
        for index, item in enumerate(gpus):
            label = f"{index}: {item.get('name')} ({item.get('vram_gb')} GB)"
            self._gpu_pick.addItem(label, index)
        idx = self._gpu_pick.findData(current)
        self._gpu_pick.setCurrentIndex(max(0, idx))
        self._gpu_pick.blockSignals(False)

    def _gpu_changed(self) -> None:
        device = int(self._gpu_pick.currentData() or 0)
        self._config.cuda_device = device
        try:
            saved = self._client.put_settings({"cuda_device": device})
            from minimax_studio.config import AppConfig, save_config

            updated = AppConfig.model_validate(saved)
            save_config(updated)
        except Exception:
            pass

    def _tick(self) -> None:
        # One list_jobs per tick shared by both pages and the status bar —
        # this runs every 500 ms, so extra round-trips here add up.
        try:
            jobs = self._client.list_jobs()
        except Exception:
            jobs = []
        self._music.poll(jobs)
        self._video.poll(jobs)
        self._refresh_job_status(jobs)
        self._refresh_download_status()
        row = self._nav.currentRow()
        if 0 <= row < len(self._nav_keys) and self._nav_keys[row] == "models":
            self._models_ticks += 1
            if self._models_ticks >= 4:  # /packs walks model trees — 2 s is enough
                self._models_ticks = 0
                self._models.refresh()
        else:
            self._models_ticks = 0
        self._route_ticks += 1
        if self._route_ticks >= 8:
            self._route_ticks = 0
            self._refresh_route()

    def _refresh_job_status(self, jobs: list[dict] | None = None) -> None:
        if jobs is None:
            try:
                jobs = self._client.list_jobs()
            except Exception:
                jobs = []
        running = next(
            (
                item
                for item in jobs
                if item.get("status") in {"running", "cancelling"}
            ),
            None,
        )
        queued = [item for item in jobs if item.get("status") == "queued"]
        queued.sort(key=lambda item: item.get("created_at") or 0)
        active = running or (queued[0] if queued else None)
        n_queued = len(queued)
        if not active:
            self._active_job_id = None
            self._status_job.setText("Job: idle")
            self._status_cancel.setEnabled(False)
        else:
            self._active_job_id = str(active.get("id") or "")
            pct = int(float(active.get("progress") or 0) * 100)
            kind = active.get("kind") or "job"
            status = active.get("status") or ""
            message = active.get("message") or ""
            extra = f"  ·  {n_queued} queued" if n_queued else ""
            self._status_job.setText(
                f"Job: {kind} {status} {pct}% — {message}{extra}"
            )
            self._status_cancel.setEnabled(
                status != "cancelling" and bool(self._active_job_id)
            )
        self._refresh_download_status()

    def _refresh_download_status(self) -> None:
        try:
            downloads = self._client.list_downloads()
        except Exception:
            downloads = []
        active_dl = next(
            (
                item
                for item in downloads
                if item.get("status") in {"queued", "running", "cancelling"}
            ),
            None,
        )
        if not active_dl:
            self._status_dl.setText("")
            return
        total = float(active_dl.get("total_bytes") or 0)
        done = float(active_dl.get("bytes") or 0)
        pct = int(min(99, (done / total) * 100)) if total else 0
        pack = active_dl.get("pack_id") or "download"
        dl_msg = active_dl.get("message") or active_dl.get("status") or ""
        self._status_dl.setText(f"DL: {pack} {pct}% — {dl_msg}")

    def _cancel_active_job(self) -> None:
        job_id = self._active_job_id
        if not job_id:
            return
        try:
            self._client.cancel_job(job_id)
        except Exception:
            return
        self._refresh_job_status()

    def _open_folder(self, path: str | None) -> None:
        from PySide6.QtWidgets import QMessageBox

        from minimax_studio.ui.reveal import reveal_path

        if not path:
            QMessageBox.information(self, "Folder", "Set this folder in Settings first.")
            return
        try:
            reveal_path(path)
        except Exception as exc:
            QMessageBox.warning(self, "Open folder failed", str(exc))

    def _open_comfy_log(self) -> None:
        from pathlib import Path

        from PySide6.QtWidgets import QMessageBox

        from minimax_studio.ui.reveal import reveal_path

        root = self._config.output_dir
        path = Path(root) / "comfy-studio.log" if root else None
        if path is None or not path.is_file():
            QMessageBox.information(
                self,
                "Comfy log",
                "No comfy-studio.log yet. Start ComfyUI from Studio to create it.",
            )
            return
        try:
            reveal_path(path)
        except Exception as exc:
            QMessageBox.warning(self, "Open log failed", str(exc))

    def _open_comfy_browser(self) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        url = self._config.comfy_url or "http://127.0.0.1:8188"
        QDesktopServices.openUrl(QUrl(url))

    def _add_go_menu(self) -> None:
        go_menu = self.menuBar().addMenu("&Go")
        pages = [
            ("video", "Generate Video", "Ctrl+1"),
            ("music", "Generate Music", "Ctrl+2"),
            ("history", "History", "Ctrl+3"),
            ("presets", "Presets", "Ctrl+4"),
            ("models", "Models", "Ctrl+5"),
            ("settings", "Settings", "Ctrl+6"),
            ("help", "Help", "Ctrl+7"),
        ]
        for key, label, shortcut in pages:
            action = QAction(label, self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(lambda _checked=False, k=key: self.show_page(k))
            go_menu.addAction(action)
        go_menu.addSeparator()
        generate = QAction("Generate", self)
        generate.setShortcut(QKeySequence("Ctrl+Return"))
        generate.triggered.connect(self._run_generate)
        go_menu.addAction(generate)
        start_comfy = QAction("Start ComfyUI", self)
        start_comfy.triggered.connect(self._start_comfy)
        go_menu.addAction(start_comfy)

    def _run_generate(self) -> None:
        row = self._nav.currentRow()
        key = self._nav_keys[row] if 0 <= row < len(self._nav_keys) else None
        if key == "music":
            self._music._generate()
        elif key == "video":
            self._video._generate()

    def _start_comfy(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        from minimax_studio.ui.comfy_watch import watch_comfy_start

        def done(ok: bool, info: dict) -> None:
            self.refresh_probe()
            if not ok:
                QMessageBox.warning(
                    self,
                    "Start ComfyUI failed",
                    str(
                        info.get("detail")
                        or info.get("log_tail")
                        or "ComfyUI did not come up."
                    ),
                )

        watch_comfy_start(self, self._client, self._status_worker.setText, done)

    def _show_welcome(self) -> None:
        from minimax_studio.ui.welcome import WelcomeDialog

        WelcomeDialog(self._client).exec()

    def _refresh_route(self) -> None:
        """Preflight off the UI thread — it can hit Comfy/packs and take
        seconds; the window must not freeze while it thinks."""
        row = self._nav.currentRow()
        key = self._nav_keys[row] if 0 <= row < len(self._nav_keys) else None
        kind = "h3" if key == "video" else "music"
        mode = getattr(self._video, "_mode", "t2va") if kind == "h3" else "ttm"
        if self._route_thread is not None:
            try:
                if self._route_thread.isRunning():
                    return  # a check is already in flight
            except RuntimeError:  # C++ object already deleted
                self._route_thread = None

        from PySide6.QtCore import QObject, QThread, Signal

        from minimax_studio.ui.enhance import on_main

        class Worker(QObject):
            finished = Signal(dict)
            failed = Signal(str)

            def run(inner_self) -> None:
                try:
                    inner_self.finished.emit(
                        self._client.preflight(
                            kind, self._state.backend, mode, self._state.speed
                        )
                    )
                except Exception as exc:
                    inner_self.failed.emit(str(exc))

        thread = QThread(self)
        worker = Worker()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        def done(check: dict) -> None:
            if check.get("ok"):
                self._route.setText(
                    str(check.get("detail") or f"Will use {check.get('backend')}")
                )
            else:
                self._route.setText(
                    "Not ready: " + str(check.get("detail") or "no backend")
                )
            thread.quit()

        def fail(_err: str) -> None:
            self._route.setText("Will use: worker unreachable")
            thread.quit()

        worker.finished.connect(on_main(done))
        worker.failed.connect(on_main(fail))
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)

        def _reset() -> None:
            self._route_thread = None

        thread.finished.connect(_reset)
        thread.finished.connect(thread.deleteLater)
        # Signal connections hold only a weak reference to the worker —
        # without this it is garbage-collected before `started` is delivered
        # and the thread never runs.
        thread._worker = worker
        thread.start()
        self._route_thread = thread


def _format_probe(probe: dict[str, Any]) -> str:
    parts = [f"{probe.get('os')} {probe.get('machine')}"]
    gpus = probe.get("gpus") or []
    if gpus:
        parts.append(
            " + ".join(f"{item.get('name')} ({item.get('vram_gb')} GB)" for item in gpus)
        )
        if not probe.get("torch_available"):
            parts.append("PyTorch not in Studio venv — Comfy INT8 still works")
    elif probe.get("cuda"):
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
