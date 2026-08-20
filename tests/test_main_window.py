from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

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


def test_main_window_builds() -> None:
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    window = MainWindow(FakeWorker())  # type: ignore[arg-type]
    assert window.windowTitle() == "MiniMax Studio"
    assert window._stack.count() == 6
