from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from minimax_studio.config import AppConfig
from minimax_studio.ui.main_window import MainWindow
from minimax_studio.ui.theme import apply_theme


class FakeWorker:
    def health(self) -> dict[str, object]:
        return {"ok": True, "version": "0.1.0"}

    def probe(self) -> dict[str, object]:
        return {
            "os": "Linux",
            "machine": "x86_64",
            "cuda": False,
            "apple_silicon": False,
            "ram_gb": 32.0,
        }

    def list_history(self) -> list:
        return []

    def list_packs(self) -> list:
        return []

    def list_downloads(self) -> list:
        return []

    def get_job(self, _job_id: str) -> dict:
        return {"status": "done", "progress": 1, "message": "Done"}

    def list_presets(self) -> list:
        return []

    def list_loras(self) -> list:
        return []


def test_main_window_builds(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    config = AppConfig(output_dir=str(tmp_path), models_dir=str(tmp_path / "models"))
    window = MainWindow(FakeWorker(), config)  # type: ignore[arg-type]
    from minimax_studio import __version__

    assert window.windowTitle() == f"MiniMax Studio {__version__}"
    assert window._stack.count() == 7
