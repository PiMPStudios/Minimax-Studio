"""PLAN-V2 S2: the Build pages, off-GPU and off-SimpleTuner.

What these pages promise is that nothing burns VRAM on a bad input and nothing
is pretended about a run we do not own. So the tests are mostly about the
refusals, and about the payloads the worker is handed when everything is fine.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from typing import Any

import pytest

# A modal in a test is a hung CI runner: every box answers itself. Shared with
# the Storage dialog tests.
from dialogs import Dialogs as _Dialogs
from PySide6.QtWidgets import QMessageBox

from minimax_studio.ui.pages import adapters_page as adapters_module
from minimax_studio.ui.pages import datasets_page as datasets_module
from minimax_studio.ui.pages import train_page as train_module
from minimax_studio.ui.pages.datasets_page import DatasetsPage
from minimax_studio.ui.pages.train_page import TrainPage

DATASET = {
    "id": "summer",
    "name": "Summer",
    "kind": "music",
    "clip_count": 3,
    "notes": "keep",
    "path": "/data/datasets/summer",
    "last_validation": {"ok": True, "checked": 3, "with_problems": 0},
}
BROKEN = {
    "id": "scraps",
    "name": "Scraps",
    "kind": "music",
    "clip_count": 2,
    "notes": "",
    "path": "/data/datasets/scraps",
    "last_validation": {"ok": False, "checked": 2, "with_problems": 2},
}
CLIP_REPORT = {
    "ok": False,
    "checked": 2,
    "at": 1.0,
    "rows": [
        {"file": "good.wav", "ok": True, "problems": []},
        {
            "file": "one.wav",
            "ok": False,
            "problems": ["missing caption one.txt", "4.0s is under the 3s floor"],
        },
        {"file": "stray.txt", "ok": False, "problems": ["caption with no matching audio file"]},
    ],
}
CLEAN_REPORT = {"ok": True, "checked": 3, "at": 1.0, "rows": []}
PRESETS = {
    "24g": {
        "title": "24 GB — conservative LoRA",
        "vram_floor_gb": 24,
        "lora_rank": 16,
        "note": "int8 everywhere",
        "family": "music",
        "ram_torch": False,
        "resolution": 512,
    },
    "48g": {
        "title": "48 GB — room to breathe",
        "vram_floor_gb": 48,
        "lora_rank": 32,
        "note": "bf16 transformer",
        "family": "music",
        "ram_torch": False,
        "resolution": 512,
    },
    "h3-24g": {
        "title": "24 GB — H3 LoRA with RamTorch",
        "vram_floor_gb": 24,
        "lora_rank": 16,
        "note": "RamTorch CPU-offload",
        "family": "h3",
        "ram_torch": True,
        "resolution": 480,
    },
    "h3-80g": {
        "title": "80 GB — H3 LoRA, full precision transformer",
        "vram_floor_gb": 80,
        "lora_rank": 32,
        "note": "bf16 at 1080p",
        "family": "h3",
        "ram_torch": False,
        "resolution": 1080,
    },
}

VIDEO_DATASET = {
    "id": "movies",
    "name": "Movies",
    "kind": "video",
    "clip_count": 2,
    "notes": "stills + one push-in",
    "path": "/data/datasets/movies",
    "h3_target_mode": "video",
    "last_validation": {"ok": True, "checked": 2, "with_problems": 0},
}
VIDEO_REPORT = {
    "ok": True,
    "checked": 2,
    "at": 1.0,
    "kind": "video",
    "stills": 1,
    "clips": 1,
    "target_mode": "video",
    "av_ready": False,
    "warnings": [],
    "rows": [
        {
            "file": "cover.png",
            "entry_kind": "still",
            "ok": True,
            "problems": [],
            "width": 1280,
            "height": 720,
            "seconds": None,
            "has_audio": False,
        },
        {
            "file": "push.mp4",
            "entry_kind": "clip",
            "ok": True,
            "problems": [],
            "width": 1920,
            "height": 1080,
            "seconds": 4.2,
            "has_audio": False,
        },
    ],
}


class FakeBuildWorker:
    """The Build half of the worker API, with call recording."""

    def __init__(self, **overrides: Any) -> None:
        self.datasets: dict[str, dict[str, Any]] = {"summer": dict(DATASET)}
        self.reports: dict[str, dict[str, Any]] = {"summer": CLEAN_REPORT}
        self.preflight_ok = True
        self.preflight_problems: list[str] = []
        self.runs: list[dict[str, Any]] = []
        self.detail: dict[str, Any] = {}
        self.started: list[dict[str, Any]] = []
        self.imported: list[tuple[str, str]] = []
        self.validated: list[str] = []
        self.cancelled: list[str] = []
        self.installed: list[str] = []
        self.deleted: list[str] = []
        self.created: list[tuple[str, str, str]] = []
        self.loras: list[dict[str, str]] = []
        self.history: list[dict[str, Any]] = []
        self.modes: list[tuple[str, str]] = []
        self.preflights: list[tuple[str, str | None]] = []
        for key, value in overrides.items():
            setattr(self, key, value)

    # datasets

    def list_datasets(self) -> list[dict[str, Any]]:
        return [
            row
            for row in sorted(self.datasets.values(), key=lambda r: str(r["name"]))
        ]

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        row = dict(self.datasets[dataset_id])
        row["validation"] = self.reports.get(dataset_id)
        row["entries"] = []
        return row

    def create_dataset(self, name: str, kind: str = "music", notes: str = "") -> dict:
        self.created.append((name, kind, notes))
        row = {**DATASET, "id": name.lower(), "name": name, "kind": kind, "path": f"/d/{name}"}
        self.datasets[row["id"]] = row
        return row

    def delete_dataset(self, dataset_id: str) -> dict:
        self.deleted.append(dataset_id)
        self.datasets.pop(dataset_id, None)
        return {"ok": True}

    def import_dataset_folder(self, dataset_id: str, folder: str) -> dict:
        self.imported.append((dataset_id, folder))
        return {"copied": ["one.wav", "two.wav"], "captions": 1}

    def add_dataset_from_history(self, dataset_id: str, history_id: str) -> dict:
        return {"added": f"{history_id}.wav"}

    def validate_dataset(self, dataset_id: str) -> dict:
        self.validated.append(dataset_id)
        return self.reports.setdefault(dataset_id, CLEAN_REPORT)

    def list_history(self) -> list[dict[str, Any]]:
        return self.history

    # training

    def set_dataset_target_mode(self, dataset_id: str, mode: str) -> dict[str, Any]:
        self.modes.append((dataset_id, mode))
        row = self.datasets[dataset_id]
        if mode == "av" and not row.get("av_capable", True):
            raise RuntimeError(
                "av mode needs an audio stream in every clip, and 1 of 2 have "
                "none (first: push.mp4). Use Video mode, or re-export the "
                "clips with audio."
            )
        row["h3_target_mode"] = mode
        return dict(row)

    def train_preflight(
        self, preset: str = "24g", dataset_dir: str | None = None
    ) -> dict:
        self.preflights.append((preset, dataset_dir))
        problems = [] if self.preflight_ok else list(self.preflight_problems) or [
            "SimpleTuner is not installed — run: pip install 'minimax-studio[train]'."
        ]
        row = PRESETS.get(preset) or PRESETS["24g"]
        return {
            "ok": not problems,
            "preset": preset,
            "family": row["family"],
            "vram_floor_gb": row["vram_floor_gb"],
            "free_vram_gb": 23.4,
            "presets": PRESETS,
            "problems": problems,
            "warnings": [],
            "dataset_dir": dataset_dir,
            "detail": "Ready to train." if not problems else " ".join(problems),
        }

    def list_train_runs(self) -> list[dict[str, Any]]:
        return self.runs

    def get_train_run(self, run_id: str, tail: int = 60) -> dict:
        return {
            "id": run_id,
            "name": "Summer LoRA",
            "status": "running",
            "steps": 1000,
            "started_at": self.detail.get("started_at", 0),
            "path": f"/runs/{run_id}",
            "progress": {"step": 250, "total_steps": 1000, "loss": 0.4123, "checkpoints": ["checkpoints/a.safetensors"]},
            "log_tail": ["step 249", "loss 0.41", "saving checkpoint"],
        }

    def start_train_run(self, payload: dict[str, Any]) -> dict:
        self.started.append(payload)
        return {"id": "run-1", "name": payload["name"], "pid": 4242, "status": "running"}

    def cancel_train_run(self, run_id: str) -> dict:
        self.cancelled.append(run_id)
        return {"id": run_id, "status": "running", "cancel_requested": True}

    def install_train_adapter(self, run_id: str, path: str | None = None) -> dict:
        self.installed.append(run_id)
        return {"id": "a.safetensors", "name": "a", "path": "/models/loras/a.safetensors"}




@pytest.fixture(autouse=True)
def no_modals(monkeypatch):
    """Answer every box, No by default. Tests that want the Yes path install
    their own _Dialogs on top of this one."""
    return _Dialogs(
        monkeypatch, {"question": QMessageBox.StandardButton.No}, datasets_module, train_module
    )


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication

    instance = QApplication.instance() or QApplication([])
    from minimax_studio.ui.theme import apply_theme

    apply_theme(instance)
    return instance


# --- Datasets page ----------------------------------------------------------


def test_lists_datasets_with_validation_state(app) -> None:
    page = DatasetsPage(FakeBuildWorker(datasets={"summer": dict(DATASET), "scraps": dict(BROKEN)}))
    texts = [page._list.item(i).text() for i in range(page._list.count())]
    assert texts == ["Scraps\n2 clips · 2 problems", "Summer\n3 clips · ready"]


def test_broken_clips_come_first_and_name_the_problem(app) -> None:
    page = DatasetsPage(
        FakeBuildWorker(
            datasets={"scraps": dict(BROKEN)}, reports={"scraps": dict(CLIP_REPORT)}
        )
    )
    page.select("scraps")
    assert page._tree.topLevelItemCount() == 3
    first = page._tree.topLevelItem(0)
    assert first.text(0) == "one.wav"
    assert "missing caption one.txt" in first.text(1)
    assert page._tree.topLevelItem(2).text(1) == "✓ ready"
    assert "2 of 2 entries have problems" in page._status.text()
    assert "missing caption" in page._status.text()
    assert "Training will refuse" in page._status.text()


def test_never_validated_says_so_without_crying_ready(app) -> None:
    unchecked = {**DATASET, "last_validation": None}
    page = DatasetsPage(
        FakeBuildWorker(datasets={"summer": unchecked}, reports={"summer": None})
    )
    assert "never validated" in page._status.text().lower()
    assert "never validated" in page._detail_label.text().lower()


def test_a_dataset_name_cannot_inject_markup_into_its_label(app) -> None:
    weird = {**DATASET, "name": "R&B <b>live</b>", "notes": "cost <5 & loud"}
    page = DatasetsPage(FakeBuildWorker(datasets={"summer": weird}))
    text = page._detail_label.text()
    assert "<b>R&amp;B &lt;b&gt;live&lt;/b&gt;</b>" in text
    assert text.count("<b>") == 1, "the only bold in that label is ours"


def test_both_trainers_are_offered_from_the_same_button(app) -> None:
    """S4 arrived: a video dataset trains today, so the button that used to
    apologise has to work — and the av control appears only where it means
    something."""
    clips = {"movies": dict(VIDEO_DATASET)}
    page = DatasetsPage(FakeBuildWorker(datasets=clips))
    page.select("movies")
    assert page._train_btn.isEnabled()
    assert page._train_btn.toolTip() == ""
    # isHidden(), not isVisible(): the page itself is never shown in a test.
    assert not page._mode_btn.isHidden()

    page = DatasetsPage(FakeBuildWorker())  # the default fake is a music dataset
    assert page._train_btn.isEnabled()
    assert page._mode_btn.isHidden()


def test_train_button_hands_the_dataset_to_the_train_page(app) -> None:
    page = DatasetsPage(FakeBuildWorker())
    got: list[str] = []
    page.train_requested.connect(got.append)
    page._train_btn.click()
    assert got == ["summer"]


def test_import_copies_then_validates_without_being_asked(app, monkeypatch) -> None:
    worker = FakeBuildWorker()
    monkeypatch.setattr(
        datasets_module.QFileDialog,
        "getExistingDirectory",
        staticmethod(lambda *args, **kwargs: "/home/me/tapes"),
    )
    page = DatasetsPage(worker)
    page._import_btn.click()
    assert worker.imported == [("summer", "/home/me/tapes")]
    assert worker.validated == ["summer"]
    assert "Copied 2 clips" in page._status.text()
    assert "1 brought a caption" in page._status.text()


def test_history_picker_offers_only_media_this_dataset_takes(app, monkeypatch) -> None:
    worker = FakeBuildWorker(
        history=[
            {"id": "h1", "kind": "music", "prompt": "folk", "output_path": "/o/h1.wav"},
            {"id": "h2", "kind": "h3", "prompt": "fox", "output_path": "/o/h2.mp4"},
        ]
    )
    captured: dict[str, list] = {}

    class Picker:
        def __init__(self, entries, kind, parent=None) -> None:
            captured["entries"] = [entry["id"] for entry in entries]
            captured["kind"] = kind

        def exec(self) -> int:
            return 1  # Accepted

        def selected_entry(self) -> dict:
            return {"id": "h1"}

    monkeypatch.setattr(datasets_module, "_HistoryPicker", Picker)
    page = DatasetsPage(worker)
    page._from_history_btn.click()
    assert captured == {"entries": ["h1"], "kind": "music"}
    assert "Added h1.wav from History" in page._status.text()


def test_delete_is_confirmed_and_says_what_survives(app, monkeypatch) -> None:
    dialogs = _Dialogs(
        monkeypatch,
        {"question": QMessageBox.StandardButton.Yes},
        datasets_module,
    )
    worker = FakeBuildWorker()
    page = DatasetsPage(worker)
    page._delete_btn.click()
    assert worker.deleted == ["summer"]
    kind, title, body = dialogs.last()
    assert kind == "question"
    assert "original files and History are untouched" in body
    assert "/data/datasets/summer" in body


def test_new_dataset_dialog_passes_kind_and_notes(app, monkeypatch) -> None:
    worker = FakeBuildWorker()
    page = DatasetsPage(worker)

    class Dialog:
        def __init__(self, parent=None) -> None:
            pass

        def exec(self) -> int:
            return 1

        def dataset_name(self) -> str:
            return "Outtakes"

        def dataset_kind(self) -> str:
            return "music"

        def dataset_notes(self) -> str:
            return "b-sides"

    monkeypatch.setattr(datasets_module, "_NewDatasetDialog", Dialog)
    page._new_btn.click()
    assert worker.created == [("Outtakes", "music", "b-sides")]


# --- Train page -------------------------------------------------------------


def test_presets_come_from_the_worker_not_a_second_copy(app) -> None:
    page = TrainPage(FakeBuildWorker())
    assert [page._preset.itemData(i) for i in range(page._preset.count())] == [
        "24g",
        "48g",
    ], "lowest VRAM floor first"
    assert page._rank.value() == 16
    page._preset.setCurrentIndex(1)
    assert page._rank.value() == 32, "rank follows the preset, but stays editable"


def test_preflight_problems_are_quoted_verbatim(app) -> None:
    worker = FakeBuildWorker(
        preflight_ok=False,
        preflight_problems=["'24 GB — conservative LoRA' needs 24 GB of free VRAM; 17 GB is free right now."],
    )
    page = TrainPage(worker)
    text = page._preflight_label.text()
    assert "Not yet" in text
    assert "17 GB is free right now" in text


def test_both_kinds_are_offered_and_labelled(app) -> None:
    """S4 changed this promise: a video dataset is trainable, so the picker
    names which trainer reads it instead of hiding it away."""
    worker = FakeBuildWorker(
        datasets={"summer": dict(DATASET), "movies": dict(VIDEO_DATASET)},
        reports={"movies": dict(VIDEO_REPORT)},
    )
    page = TrainPage(worker)
    labels = [page._dataset.itemText(i) for i in range(page._dataset.count())]
    assert any(
        text.startswith("Summer — 3 clips · ready") and "[H3]" not in text
        for text in labels
    ), labels
    assert any(
        text.startswith("Movies — 2 stills/clips · ready") and text.endswith("[H3]")
        for text in labels
    ), labels


def test_the_picker_tells_2_problems_apart_from_never_checked(app) -> None:
    page = TrainPage(FakeBuildWorker(datasets={"scraps": dict(BROKEN)}))
    assert "2 problems" in page._dataset.currentText()
    unchecked = {**DATASET, "last_validation": None}
    page = TrainPage(FakeBuildWorker(datasets={"summer": unchecked}))
    assert "not checked" in page._dataset.currentText()


def test_no_dataset_explains_what_to_do_instead(app) -> None:
    page = TrainPage(FakeBuildWorker(datasets={}))
    assert not page._dataset.isEnabled()
    assert not page._start_btn.isEnabled()
    assert "Datasets page" in page._form_status.text()


def test_start_refuses_while_preflight_is_failing(app, monkeypatch) -> None:
    dialogs = _Dialogs(monkeypatch, {}, train_module)
    worker = FakeBuildWorker(preflight_ok=False)
    page = TrainPage(worker)
    page._start_btn.click()
    assert worker.started == []
    assert dialogs.kinds()[-1] == "warning"
    assert "nothing was started" in dialogs.last()[1]


def test_start_refuses_a_dataset_that_does_not_validate(app, monkeypatch) -> None:
    dialogs = _Dialogs(monkeypatch, {}, train_module)
    worker = FakeBuildWorker(datasets={"scraps": dict(BROKEN)})
    worker.reports["scraps"] = dict(CLIP_REPORT)
    page = TrainPage(worker)
    assert page._dataset.count() == 1
    page._start_btn.click()
    assert worker.started == []
    assert "not ready" in dialogs.last()[1].lower()
    assert "one.wav" in dialogs.last()[2]
    assert "missing caption one.txt" in dialogs.last()[2]


def test_start_sends_the_contract_the_worker_reads(app, monkeypatch) -> None:
    _Dialogs(
        monkeypatch,
        {"question": QMessageBox.StandardButton.Yes},
        train_module,
    )
    worker = FakeBuildWorker()
    page = TrainPage(worker)
    page._name.setText("Summer LoRA")
    page._steps.setValue(400)
    page._validation_prompt.setText("jangly indie")
    page._start_btn.click()
    assert worker.started == [
        {
            "name": "Summer LoRA",
            "dataset_dir": "/data/datasets/summer",
            "preset": "24g",
            "steps": 400,
            "rank": 16,
            "validation": {"prompt": "jangly indie", "duration": 15},
        }
    ]
    assert "pid 4242" in page._form_status.text()
    assert "survives studio closing" in page._form_status.text().lower()


def test_cancel_is_asked_for_before_signalling_the_process_group(app, monkeypatch) -> None:
    _Dialogs(
        monkeypatch,
        {"question": QMessageBox.StandardButton.Yes},
        train_module,
    )
    worker = FakeBuildWorker(runs=[{"id": "r1", "name": "Summer LoRA", "status": "running", "steps": 1000, "path": "/runs/r1"}])
    page = TrainPage(worker)
    page._cancel_btn.click()
    assert worker.cancelled == ["r1"]


def test_installed_adapter_goes_straight_to_the_lora_picker(app, no_modals) -> None:
    worker = FakeBuildWorker(
        runs=[{"id": "r1", "name": "Summer LoRA", "status": "completed", "steps": 1000, "path": "/runs/r1"}]
    )
    page = TrainPage(worker)
    fired: list[bool] = []
    page.adapter_installed.connect(lambda: fired.append(True))
    assert page._install_btn.isEnabled(), "the fake run reports a checkpoint"
    page._install_btn.click()
    assert worker.installed == ["r1"]
    assert fired == [True]
    kind, title, body = no_modals.last()
    assert kind == "information"
    assert title == "Adapter installed"
    assert "LoRA dropdown" in body


def test_a_run_we_did_not_start_is_still_lively(app) -> None:
    """Detached means detached: after a Studio restart the run is read back
    from its folder, with progress, and stays cancellable."""
    worker = FakeBuildWorker(
        runs=[
            {
                "id": "r1",
                "name": "Overnight",
                "status": "running",
                "steps": 1000,
                "path": "/runs/r1",
            }
        ],
        detail={"started_at": 1.0},
    )
    page = TrainPage(worker)
    row = page._runs_tree.topLevelItem(0)
    assert row.text(0) == "Overnight"
    assert row.text(1) == "running"
    status = page._run_status.text()
    assert "step 250 of 1000" in status
    assert "loss 0.4123" in status
    assert "1 checkpoint" in status
    assert "\n".join(page._log.toPlainText().splitlines()[-1:]) == "saving checkpoint"
    assert page._cancel_btn.isEnabled()


def test_a_finished_run_offers_install_not_cancel(app) -> None:
    class Done(FakeBuildWorker):
        def get_train_run(self, run_id: str, tail: int = 60) -> dict:
            return {
                "id": run_id,
                "name": "Overnight",
                "status": "completed",
                "steps": 10,
                "started_at": 1.0,
                "path": "/runs/r1",
                "exit_code": 0,
                "progress": {"step": 10, "total_steps": 10, "loss": 0.2, "checkpoints": []},
                "log_tail": ["done"],
            }

    page = TrainPage(Done(runs=[{"id": "r1", "name": "Overnight", "status": "completed", "steps": 10, "path": "/runs/r1"}]))
    assert not page._cancel_btn.isEnabled()
    assert not page._install_btn.isEnabled(), "no checkpoint, nothing to install"
    assert "completed" in page._run_status.text()


def test_build_pages_name_their_buttons_consistently() -> None:
    """`self._audition = QPushButton(...)` silently *replaces* the method of the
    same name, and the next line — ``.clicked.connect(self._audition)`` — hands
    a widget to a signal: TypeError at construction, or worse, a dead button.
    This bit three pages, so buttons now wear ``_btn`` and the rule is checked
    rather than remembered.
    """
    import inspect
    import re

    for module in (datasets_module, train_module, adapters_module):
        for match in re.finditer(r"self\.(_[a-z0-9_]+)\s*=\s*QPushButton", inspect.getsource(module)):
            name = match.group(1)
            assert name.endswith("_btn"), f"{module.__name__}: button attribute {name} needs the _btn suffix"


# --- H3 (PLAN-V2 S4, off-GPU half) ------------------------------------------


def _video_page(monkeypatch, **worker_kwargs):
    """A DatasetsPage holding one validated H3 dataset."""
    worker = FakeBuildWorker(
        datasets={"movies": dict(VIDEO_DATASET)},
        reports={"movies": dict(VIDEO_REPORT)},
        **worker_kwargs,
    )
    page = DatasetsPage(worker)
    page.select("movies")
    return page, worker


def test_the_detail_says_which_mode_and_what_was_measured(app) -> None:
    page, _worker = _video_page(None)
    detail = page._detail_label.text()
    assert "video dataset" in detail
    assert "target mode <b>video</b>" in detail
    # Measured facts, not just filenames: a still and a clip are different work.
    rows = [page._tree.topLevelItem(i).text(0) for i in range(page._tree.topLevelItemCount())]
    assert any("cover.png" in text and "1280×720" in text and "still" in text for text in rows)
    assert any("push.mp4" in text and "4.2s" in text for text in rows)
    status = page._status.text()
    assert "(1 still, 1 clip)" in status and "Ready to train" in status


def test_av_mode_is_confirmed_then_applied(app, monkeypatch) -> None:
    page, worker = _video_page(monkeypatch)
    dialogs = _Dialogs(
        monkeypatch, {"question": QMessageBox.StandardButton.Yes}, datasets_module
    )
    page._mode_btn.click()
    body = dialogs.last()[2]
    assert "extra VRAM and disk" in body and "every clip" in body
    assert worker.modes == [("movies", "av")]
    assert "Target mode is <b>av</b>" in page._status.text()


def test_saying_no_to_av_mode_changes_nothing(app, monkeypatch) -> None:
    page, worker = _video_page(monkeypatch)
    page._mode_btn.click()  # the autouse fixture answers No
    assert worker.modes == []


def test_av_refusal_from_the_worker_is_read_out_not_swallowed(app, monkeypatch) -> None:
    page, worker = _video_page(monkeypatch)
    worker.datasets["movies"]["av_capable"] = False
    _Dialogs(monkeypatch, {"question": QMessageBox.StandardButton.Yes}, datasets_module)
    page._mode_btn.click()
    assert worker.modes == [("movies", "av")]  # asked; the worker said no
    assert "push.mp4" in page._status.text()
    assert "av mode needs an audio stream" in page._status.text()
    assert worker.datasets["movies"]["h3_target_mode"] == "video"  # untouched


def test_giving_up_av_mode_needs_no_confirmation(app, monkeypatch) -> None:
    page, worker = _video_page(monkeypatch)
    worker.datasets["movies"]["h3_target_mode"] = "av"
    page.refresh()
    page.select("movies")
    assert page._mode_btn.text().startswith("Back to video-only")
    dialogs = _Dialogs(monkeypatch, {}, datasets_module)
    page._mode_btn.click()
    assert dialogs.kinds() == []  # the cheap direction is not a question
    assert worker.modes == [("movies", "video")]


def _select_dataset(page, dataset_id: str) -> None:
    for index in range(page._dataset.count()):
        if (page._dataset.itemData(index) or {}).get("id") == dataset_id:
            page._dataset.setCurrentIndex(index)
            return
    raise AssertionError(f"no {dataset_id} in the picker")


def test_the_train_page_switches_tiers_with_the_dataset(app) -> None:
    worker = FakeBuildWorker(
        datasets={"summer": dict(DATASET), "movies": dict(VIDEO_DATASET)},
        reports={"movies": dict(VIDEO_REPORT)},
    )
    page = TrainPage(worker)
    _select_dataset(page, "summer")
    assert [page._preset.itemText(i) for i in range(page._preset.count())] == [
        "24 GB — conservative LoRA · rank 16",
        "48 GB — room to breathe · rank 32",
    ]
    assert page._dataset_label.text() == "Music dataset"
    assert not page._validation_duration.isHidden()

    _select_dataset(page, "movies")
    assert [page._preset.itemData(i) for i in range(page._preset.count())] == [
        "h3-24g",
        "h3-80g",
    ]
    assert page._dataset_label.text().startswith("H3 dataset")
    # An H3 run has no audio length to promise, so the box is not there.
    assert page._validation_duration.isHidden()
    assert worker.preflights[-1][0] == "h3-24g"
    assert worker.preflights[-1][1] == "/data/datasets/movies"
    # What is guaranteed, precisely: the first check of the page's life cannot
    # know the tier table yet (only the worker has it), so it asks about the
    # preset on screen and re-asks once the table moves the selection. What must
    # never happen is a *stale verdict staying on screen*, and once the table is
    # known, a family switch asks about the right pair the first time.
    asked_about_movies = [preset for preset, path in worker.preflights if path == "/data/datasets/movies"]
    assert asked_about_movies[-1] == "h3-24g"
    assert asked_about_movies.count("h3-24g") == asked_about_movies.count("24g") + 1

    before = len(worker.preflights)
    _select_dataset(page, "summer")
    _select_dataset(page, "movies")
    assert worker.preflights[before:] == [
        ("24g", "/data/datasets/summer"),
        ("h3-24g", "/data/datasets/movies"),
    ]

    _select_dataset(page, "summer")
    assert [page._preset.itemData(i) for i in range(page._preset.count())] == [
        "24g",
        "48g",
    ]


def test_the_h3_tiers_cannot_be_started_on_a_music_dataset(app, monkeypatch) -> None:
    """The picker filters, and the payload still goes through the worker's own
    refusal — belt and braces, because the wrong pair wastes hours."""
    page = TrainPage(FakeBuildWorker())
    assert page._preset.currentData() == "24g"
    assert not any("h3" in str(page._preset.itemData(i)) for i in range(page._preset.count()))
    assert not page._validation_duration.isHidden()


def test_an_h3_run_is_started_without_an_audio_length(app, monkeypatch) -> None:
    worker = FakeBuildWorker(
        datasets={"movies": dict(VIDEO_DATASET)},
        reports={"movies": dict(VIDEO_REPORT)},
    )
    page = TrainPage(worker)
    _Dialogs(monkeypatch, {"question": QMessageBox.StandardButton.Yes}, train_module)
    _select_dataset(page, "movies")
    page._name.setText("push-in lora")
    page._start_run()
    assert worker.started, [call[0] for call in worker.preflights]
    payload = worker.started[-1]
    assert payload["preset"] == "h3-24g"
    assert payload["dataset_dir"] == "/data/datasets/movies"
    assert "duration" not in payload["validation"]
