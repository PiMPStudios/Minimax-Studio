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

    def preflight(
        self, kind: str, backend: str = "auto", mode: str = "t2va", speed: str = "quality"
    ) -> dict:
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
    assert "&View" in menus
    view = next(action for action in window.menuBar().actions() if action.text() == "&View")
    view_texts = [child.text() for child in view.menu().actions()]
    assert any("Setup checklist" in text for text in view_texts)


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


def test_history_detail_includes_seed() -> None:
    from minimax_studio.ui.pages.history_page import _history_detail

    text = _history_detail(
        {
            "kind": "h3",
            "backend": "comfy",
            "mode": "t2va",
            "duration_s": 8,
            "seed": 7,
            "steps": 30,
            "prompt": "a fox",
            "output_path": "/tmp/x.mp4",
        }
    )
    assert "seed 7" in text
    assert "8s" in text
    assert "a fox" in text


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


def _drain_window(app, window) -> None:
    """Stop the tick timer and join any route-preflight thread the window
    started, so pytest teardown never sees a running QThread."""
    window._timer.stop()
    app.processEvents()
    if window._route_thread is not None:
        window._route_thread.wait(3000)


def test_tick_shares_one_jobs_snapshot_and_avoids_get_job(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    apply_theme(app)

    class CountingClient(FakeWorker):
        def __init__(self) -> None:
            self.list_jobs_calls = 0
            self.get_job_calls = 0

        def list_jobs(self) -> list:
            self.list_jobs_calls += 1
            return [
                {
                    "id": "live1",
                    "kind": "music",
                    "status": "running",
                    "progress": 0.5,
                    "message": "Sampling",
                    "created_at": 1,
                }
            ]

        def get_job(self, job_id: str) -> dict:
            self.get_job_calls += 1
            return {"status": "running", "progress": 0.5, "message": "Sampling"}

    client = CountingClient()
    config = AppConfig(output_dir=str(tmp_path), models_dir=str(tmp_path / "models"))
    window = MainWindow(client, config)  # type: ignore[arg-type]
    window._music._job_id = "live1"

    client.list_jobs_calls = 0
    client.get_job_calls = 0
    window._tick()

    # status bar + music queue + video queue + music live job all come from
    # ONE list_jobs call, and no per-page get_job fires.
    assert client.list_jobs_calls == 1
    assert client.get_job_calls == 0
    assert "music running 50%" in window._status_job.text()
    assert window._music._bar.value() == 50
    assert "Sampling" in window._music._status.text()
    _drain_window(app, window)


def test_tick_throttles_models_refresh(tmp_path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    apply_theme(app)

    client = FakeWorker()
    config = AppConfig(output_dir=str(tmp_path), models_dir=str(tmp_path / "models"))
    window = MainWindow(client, config)  # type: ignore[arg-type]
    refreshes = {"n": 0}
    monkeypatch.setattr(
        window._models,
        "refresh",
        lambda: refreshes.__setitem__("n", refreshes["n"] + 1),
    )
    window.show_page("models")
    refreshes["n"] = 0  # ignore the refresh the page switch itself does
    for _ in range(7):
        window._tick()
    # Ticks fire refresh only every 4th tick (Models walks model trees).
    assert refreshes["n"] == 1
    _drain_window(app, window)
