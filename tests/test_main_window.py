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

    def start_download(self, pack_id: str) -> dict:
        return {"id": "x", "pack_id": pack_id, "status": "queued"}

    def cancel_download(self, job_id: str) -> dict:
        return {"id": job_id, "status": "cancelling"}

    def list_downloads(self) -> list:
        return []

    def get_job(self, _job_id: str) -> dict:
        return {"status": "done", "progress": 1, "message": "Done"}

    def list_jobs(self) -> list:
        return []

    def cancel_job(self, job_id: str) -> dict:
        return {"id": job_id, "status": "cancelling"}

    def list_presets(self) -> list:
        return []

    def list_loras(self) -> list:
        return []

    def preflight(self, kind: str, backend: str = "auto", mode: str = "t2va") -> dict:
        return {
            "ok": False,
            "detail": "ComfyUI is not running",
            "backend": None,
            "kind": kind,
        }

    def put_settings(self, payload: dict) -> dict:
        return payload

    def ping(self) -> dict:
        return {
            "minimax": {"ok": False, "detail": "no key"},
            "llm": {"ok": True, "detail": "HTTP 200"},
            "comfy": {"ok": False, "detail": "connection refused"},
        }

    def start_comfy(self) -> dict:
        return {"ok": True, "already": True, "detail": "already running"}

    def comfy_status(self) -> dict:
        return {"root": None, "running": False}


def test_welcome_dialog_builds(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    from minimax_studio.ui.welcome import WelcomeDialog

    dialog = WelcomeDialog(FakeWorker())  # type: ignore[arg-type]
    assert dialog.windowTitle() == "MiniMax Studio"


def test_main_window_builds(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    config = AppConfig(output_dir=str(tmp_path), models_dir=str(tmp_path / "models"))
    window = MainWindow(FakeWorker(), config)  # type: ignore[arg-type]
    from minimax_studio import __version__

    assert window.windowTitle() == f"MiniMax Studio {__version__}"
    assert window._stack.count() == 7
    menus = [action.text() for action in window.menuBar().actions()]
    assert "&File" in menus
    assert "&Go" in menus


def test_history_filter(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    apply_theme(app)

    class HistoryClient(FakeWorker):
        def list_history(self) -> list:
            return [
                {"id": "a", "kind": "h3", "prompt": "cat video", "output_path": ""},
                {"id": "b", "kind": "music", "prompt": "folk song", "output_path": ""},
            ]

    from minimax_studio.ui.pages.history_page import HistoryPage
    from minimax_studio.ui.state import StudioState

    page = HistoryPage(HistoryClient(), StudioState())  # type: ignore[arg-type]
    page.refresh()
    assert page._list.count() == 2
    page._kind.setCurrentText("Music")
    assert page._list.count() == 1
    assert page._visible[0]["kind"] == "music"
    page._kind.setCurrentText("All")
    page._search.setText("cat")
    assert page._list.count() == 1
    assert page._visible[0]["id"] == "a"
    page._search.setText("")
    page.refresh()
    assert page._list.currentRow() == 0


def test_presets_filter(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    apply_theme(app)

    class PresetClient(FakeWorker):
        def list_presets(self) -> list:
            return [
                {"id": "1", "name": "cinematic night", "kind": "h3"},
                {"id": "2", "name": "folk acoustic", "kind": "music"},
            ]

    from minimax_studio.ui.pages.presets_page import PresetsPage
    from minimax_studio.ui.state import StudioState

    page = PresetsPage(PresetClient(), StudioState())  # type: ignore[arg-type]
    page.refresh()
    assert page._list.count() == 2
    page._kind.setCurrentText("Video")
    assert page._list.count() == 1
    assert page._visible[0]["kind"] == "h3"
    page._kind.setCurrentText("All")
    page._search.setText("folk")
    assert page._list.count() == 1
    assert page._visible[0]["id"] == "2"
