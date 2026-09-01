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
        self,
        kind: str,
        backend: str = "auto",
        mode: str = "t2va",
        speed: str = "quality",
        resolution: str = "768P",
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

    # Build pages (S2). Empty by default; the tests that care subclass these.
    def list_datasets(self) -> list:
        return []

    def get_dataset(self, dataset_id: str) -> dict:
        return {"id": dataset_id, "name": dataset_id, "kind": "music", "path": ""}

    def validate_dataset(self, dataset_id: str) -> dict:
        return {"ok": True, "checked": 0, "rows": []}

    def train_preflight(self, preset: str = "24g") -> dict:
        return {
            "ok": False,
            "detail": "stub: SimpleTuner is not installed",
            "problems": ["stub: SimpleTuner is not installed"],
            "warnings": [],
            "presets": {},
        }

    def list_train_runs(self) -> list:
        return []

    def list_adapters(self) -> list:
        return []

    def get_train_run(self, run_id: str, tail: int = 60) -> dict:
        return {"id": run_id, "status": "completed", "progress": {}, "log_tail": []}


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
    assert window._stack.count() == 10
    assert [
        key for key in window._nav_keys if key in {"datasets", "train", "adapters"}
    ] == ["datasets", "train", "adapters"], "Build pages belong in the sidebar"
    assert window._train.isEnabled()
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


def test_history_trim_dialog_defaults_and_restore_on_child(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    apply_theme(app)

    from minimax_studio.ui.pages.history_page import (
        HistoryPage,
        TrimDialog,
        _history_detail,
    )
    from minimax_studio.ui.state import StudioState

    dialog = TrimDialog(8.0)
    assert dialog.points() == (0.0, 8.0)
    dialog.close()

    child = {
        "id": "childchild12",
        "kind": "music",
        "prompt": "folk song",
        "output_path": str(tmp_path / "cut.wav"),
        "trimmed_from": "parentparent",
        "duration_s": 2.5,
    }
    assert "trimmed from parentparent" in _history_detail(child)

    class HistoryClient(FakeWorker):
        def list_history(self) -> list:
            return [child]

    state = StudioState()
    caught: list = []
    state.restore_music.connect(lambda entry: caught.append(entry))
    page = HistoryPage(HistoryClient(), state)  # type: ignore[arg-type]
    page.refresh()
    assert "trim" in page._list.item(0).text()
    page._restore_current()
    assert caught and caught[0]["trimmed_from"] == "parentparent"


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
    """Stop the tick timer and join any background thread the window
    started, so pytest teardown never sees a running QThread."""
    window._timer.stop()
    app.processEvents()
    for attr in ("_route_thread", "_jobs_thread", "_train_status_thread"):
        thread = getattr(window, attr, None)
        if thread is not None:
            try:
                thread.wait(3000)
            except RuntimeError:
                pass
    app.processEvents()
    app.processEvents()


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
    _drain_window(app, window)
    window._music._job_id = "live1"

    client.list_jobs_calls = 0
    client.get_job_calls = 0
    window._tick()
    _drain_window(app, window)

    # status bar + music queue + video queue + music live job all come from
    # ONE list_jobs call, and no per-page get_job fires.
    assert client.list_jobs_calls == 1
    assert client.get_job_calls == 0
    assert "music running 50%" in window._status_job.text()
    assert window._music._bar.value() == 50
    assert "Sampling" in window._music._status.text()


def test_tick_throttles_models_refresh(tmp_path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    apply_theme(app)

    client = FakeWorker()
    config = AppConfig(output_dir=str(tmp_path), models_dir=str(tmp_path / "models"))
    window = MainWindow(client, config)  # type: ignore[arg-type]
    refreshes = {"n": 0}
    monkeypatch.setattr(
        window._models,
        "poll",
        lambda: refreshes.__setitem__("n", refreshes["n"] + 1),
    )
    window.show_page("models")
    refreshes["n"] = 0  # ignore the refresh the page switch itself does
    for _ in range(7):
        window._tick()
    # Ticks fire refresh only every 4th tick (Models walks model trees).
    assert refreshes["n"] == 1
    _drain_window(app, window)


def test_inspector_now_line_and_music_api_hints(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    config = AppConfig(output_dir=str(tmp_path), models_dir=str(tmp_path / "models"))
    window = MainWindow(FakeWorker(), config)  # type: ignore[arg-type]

    assert "Nothing running" in window._now_label.text()
    window._refresh_job_status(
        [
            {
                "id": "1",
                "kind": "music",
                "status": "running",
                "progress": 0.4,
                "backend": "api",
                "message": "Calling API",
            }
        ]
    )
    text = window._now_label.text()
    assert "Music" in text and "api" in text and "GPU stays idle" in text
    window._refresh_job_status([])
    assert "Nothing running" in window._now_label.text()

    # Music via API: the controls it ignores must say so; other backends restore.
    window._apply_api_param_hints({"kind": "music", "backend": "api"})
    assert "ignores" in window._seed.toolTip()
    assert "ignores" in window._cfg.toolTip()
    window._apply_api_param_hints({"kind": "h3", "backend": "cuda"})
    assert window._seed.toolTip() == ""
    assert window._cfg.toolTip() == window._default_param_tips[window._cfg]


def test_status_bar_names_a_live_training_run(tmp_path) -> None:
    """Training is detached: it has no job in the queue, so the status bar is
    where it stays visible while you are on another page."""
    app = QApplication.instance() or QApplication([])
    apply_theme(app)

    class Training(FakeWorker):
        def __init__(self) -> None:
            self.runs = [
                {"id": "r1", "name": "Overnight", "status": "running", "steps": 1000},
                {"id": "r2", "name": "Retake", "status": "completed", "steps": 500},
            ]

        def list_train_runs(self) -> list:
            return list(self.runs)

    client = Training()
    config = AppConfig(output_dir=str(tmp_path), models_dir=str(tmp_path / "models"))
    window = MainWindow(client, config)  # type: ignore[arg-type]
    # Constructor already kicked this off-thread via _on_nav; wait, then ask
    # again so a skipped start_background (thread still running) cannot hide
    # a hung fetch.
    _drain_window(app, window)
    window._refresh_train_status()
    _drain_window(app, window)
    assert "Training: Overnight" in window._status_train.text()
    assert "running" in window._status_train.text()
    assert "Ctrl+Shift+T" in window._status_train.text(), "say where to look"

    client.runs[0]["status"] = "lost"
    window._refresh_train_status()
    _drain_window(app, window)
    assert "Training: Overnight" in window._status_train.text()
    assert "lost" in window._status_train.text()

    window._client = FakeWorker()  # run finished / worker unreachable
    window._refresh_train_status()
    _drain_window(app, window)
    assert window._status_train.text() == ""


def test_dataset_page_hands_its_dataset_to_the_train_page(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    apply_theme(app)

    class WithDataset(FakeWorker):
        def list_datasets(self) -> list:
            return [
                {
                    "id": "summer",
                    "name": "Summer",
                    "kind": "music",
                    "clip_count": 3,
                    "path": "/data/datasets/summer",
                    "last_validation": {"ok": True, "checked": 3},
                }
            ]

    config = AppConfig(output_dir=str(tmp_path), models_dir=str(tmp_path / "models"))
    window = MainWindow(WithDataset(), config)  # type: ignore[arg-type]
    window.show_page("datasets")
    window._datasets._train_btn.click()
    assert window._stack.currentWidget() is window._train
    assert window._train._dataset.currentData()["id"] == "summer"
    _drain_window(app, window)


def test_installing_an_adapter_updates_the_registry_and_the_picker(tmp_path, monkeypatch) -> None:
    """Install adapter is the handoff point: the LoRA dropdown and the
    Adapters page must both know about the file without a restart."""
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    config = AppConfig(output_dir=str(tmp_path), models_dir=str(tmp_path / "models"))
    window = MainWindow(FakeWorker(), config)  # type: ignore[arg-type]
    refreshes = {"pickers": 0, "registry": 0}
    monkeypatch.setattr(
        window, "refresh_loras", lambda: refreshes.__setitem__("pickers", refreshes["pickers"] + 1)
    )
    monkeypatch.setattr(
        window._adapters,
        "refresh",
        lambda: refreshes.__setitem__("registry", refreshes["registry"] + 1),
    )
    window._train.adapter_installed.emit()
    assert refreshes == {"pickers": 1, "registry": 1}
    _drain_window(app, window)


def test_music_poll_empty_snapshot_does_not_call_get_job(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    apply_theme(app)

    class Client(FakeWorker):
        def __init__(self) -> None:
            self.gets = 0

        def get_job(self, _job_id: str) -> dict:
            self.gets += 1
            raise RuntimeError("should not round-trip")

    from minimax_studio.ui.pages.music_page import MusicPage
    from minimax_studio.ui.state import StudioState

    client = Client()
    page = MusicPage(client, StudioState())  # type: ignore[arg-type]
    page._job_id = "abc123"
    page.poll([])
    assert client.gets == 0
    assert page._job_id == "abc123"
