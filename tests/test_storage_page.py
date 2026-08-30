"""PLAN-V2 S5 UI: the Storage dialog, Resume, Import — off-GPU, off-modal.

The promise being tested is a narrow one: **the number comes before the
deletion.** Every destructive action here has to say, in its confirmation, how
much it frees and what it keeps, and the buttons must not exist at all while a
trainer is live. Plus the two honest-refusal paths — a worker that says "still
training" and an import folder that is not a run.
"""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

# A modal in a test is a hung CI runner.
from dialogs import Dialogs
from PySide6.QtWidgets import QMessageBox
from test_build_pages import FakeBuildWorker

from minimax_studio.ui.pages import storage_dialog as storage_module
from minimax_studio.ui.pages import train_page as train_module
from minimax_studio.ui.pages.storage_dialog import StorageDialog, human_bytes

_GB = 1024**3


def _checkpoints() -> list[dict[str, Any]]:
    # Newest first, and the *second* one is the installed adapter: that is the
    # case a naive "keep the newest N" policy gets wrong.
    return [
        {
            "path": "checkpoints/step-800/lora.safetensors",
            "abs": "/runs/run-1/checkpoints/step-800/lora.safetensors",
            "bytes": 1 * _GB,
            "written_at": 300.0,
            "installed": False,
        },
        {
            "path": "checkpoints/step-400/lora.safetensors",
            "abs": "/runs/run-1/checkpoints/step-400/lora.safetensors",
            "bytes": 1 * _GB,
            "written_at": 200.0,
            "installed": True,
        },
        {
            "path": "checkpoints/step-200/lora.safetensors",
            "abs": "/runs/run-1/checkpoints/step-200/lora.safetensors",
            "bytes": 1 * _GB,
            "written_at": 100.0,
            "installed": False,
        },
    ]


def _report(status: str = "completed") -> dict[str, Any]:
    return {
        "id": "run-1",
        "name": "Summer LoRA",
        "status": status,
        "path": "/runs/run-1",
        "cache_bytes": 3 * _GB,
        "checkpoint_bytes": 3 * _GB,
        "bytes": 6 * _GB,
        "free_gb": 41.2,
        "checkpoints": _checkpoints(),
    }


class FakeStorageWorker(FakeBuildWorker):
    """The S5 half of the worker API, with call recording."""

    def __init__(self, **overrides: Any) -> None:
        super().__init__(**overrides)
        self.storage_report: dict[str, Any] = _report()
        self.pruned: list[tuple[str, int]] = []
        self.cleared: list[str] = []
        self.exported: list[tuple[str, str, bool]] = []
        self.resumed: list[tuple[str, str | None]] = []
        self.imported_runs: list[str] = []
        self.freed: list[str] = []
        self.storage_error: str | None = None

    def get_train_run(self, run_id: str, tail: int = 60) -> dict:
        return {
            "id": run_id,
            "name": "Summer LoRA",
            "status": self.detail.get("status", "completed"),
            "steps": 1000,
            "started_at": 0,
            "path": f"/runs/{run_id}",
            "progress": {
                "step": 800,
                "total_steps": 1000,
                "loss": 0.38,
                "checkpoints": self.detail.get("checkpoints", ["checkpoints/step-800/lora.safetensors"]),
            },
            "log_tail": ["step 799", "loss 0.38"],
            "resume_count": 1,
        }

    def train_run_storage(self, run_id: str) -> dict:
        if self.storage_error:
            raise RuntimeError(self.storage_error)
        return self.storage_report

    def prune_train_checkpoints(self, run_id: str, keep: int = 3, dry_run: bool = False) -> dict:
        self.pruned.append((run_id, keep, dry_run))
        rows = self.storage_report["checkpoints"]
        doomed = [row for row in rows[keep:] if not row["installed"]]
        freed = sum(int(row["bytes"]) for row in doomed)
        kept = [row["path"] for row in rows if row not in doomed]
        if dry_run:
            return {"kept": kept, "removed": [row["path"] for row in doomed], "freed_bytes": freed, "dry_run": True}
        self.storage_report = {**self.storage_report, "checkpoints": [row for row in rows if row not in doomed]}
        return {"kept": kept, "removed": [row["path"] for row in doomed], "freed_bytes": freed, "dry_run": False}

    def clear_train_cache(self, run_id: str) -> dict:
        self.cleared.append(run_id)
        freed = self.storage_report["cache_bytes"] if self.cleared.count(run_id) == 1 else 0
        return {"cleared": freed > 0, "freed_bytes": freed}

    def export_train_run(self, run_id: str, dest: str, include_cache: bool = False) -> dict:
        self.exported.append((run_id, dest, include_cache))
        return {"path": f"{dest}/{run_id}", "files": 5, "bytes": 3 * _GB, "id": run_id}

    def delete_train_run(self, run_id: str) -> dict:
        self.freed.append(run_id)
        self.runs = [row for row in self.runs if row.get("id") != run_id]
        return {"id": run_id, "freed_bytes": 6 * _GB}

    def resume_train_run(self, run_id: str, checkpoint: str | None = None) -> dict:
        self.resumed.append((run_id, checkpoint))
        return {
            "id": run_id,
            "name": "Summer LoRA",
            "status": "running",
            "pid": 909,
            "resume_count": 2,
            "resumed_from": "/runs/run-1/checkpoints/step-800/lora.safetensors",
        }

    def import_train_run(self, folder: str) -> dict:
        self.imported_runs.append(folder)
        row = {"id": "run-9", "name": "Imported LoRA", "status": "completed", "path": "/runs/run-9"}
        self.runs = [row] + [r for r in self.runs if r.get("id") != "run-9"]
        return row


@pytest.fixture(autouse=True)
def no_modals(monkeypatch):
    """No box opens unanswered in this file — and No is the default answer."""
    return Dialogs(
        monkeypatch,
        {"question": QMessageBox.StandardButton.No},
        storage_module,
        train_module,
    )


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication

    instance = QApplication.instance() or QApplication([])
    from minimax_studio.ui.theme import apply_theme

    apply_theme(instance)
    return instance


@pytest.fixture
def worker() -> FakeStorageWorker:
    return FakeStorageWorker()


def _dialog(app, worker, status: str = "completed") -> StorageDialog:
    worker.storage_report = _report(status)
    return StorageDialog(worker, {"id": "run-1", "name": "Summer LoRA", "path": "/runs/run-1"})


# --- the numbers -------------------------------------------------------------


def test_human_bytes_uses_one_honest_unit() -> None:
    assert human_bytes(0) == "0 B"
    assert human_bytes(1536) == "1.5 KB"
    assert human_bytes(5 * _GB) == "5.0 GB"
    assert human_bytes("nonsense") == "—"


def test_the_dialog_leads_with_the_sizes(app, worker) -> None:
    dialog = _dialog(app, worker)
    totals = dialog._totals.text()
    assert "6.0 GB" in totals  # the whole run
    assert "3.0 GB" in totals  # and both halves of it
    assert "41.2 GB free" in totals
    assert dialog._path_label.text() == "/runs/run-1"


def test_installed_checkpoints_are_marked_as_kept(app, worker) -> None:
    dialog = _dialog(app, worker)
    rows = [
        [dialog._tree.topLevelItem(i).text(col) for col in range(4)]
        for i in range(dialog._tree.topLevelItemCount())
    ]
    assert [row[0] for row in rows] == [
        "checkpoints/step-800/lora.safetensors",
        "checkpoints/step-400/lora.safetensors",
        "checkpoints/step-200/lora.safetensors",
    ]
    assert [row[3] for row in rows] == ["", "yes", ""]
    assert rows[0][1] == "1.0 GB"


def test_a_run_that_never_saved_says_so(app, worker) -> None:
    worker.storage_report = {**_report(), "checkpoints": [], "checkpoint_bytes": 0, "bytes": 3 * _GB}
    dialog = StorageDialog(worker, {"id": "run-1", "name": "Summer LoRA", "path": "/runs/run-1"})
    assert "no checkpoints written yet" in dialog._totals.text()


def test_a_cant_measure_run_says_why_instead_of_shrugging(app, worker) -> None:
    worker.storage_error = "No training run 'run-1'."
    dialog = StorageDialog(worker, {"id": "run-1", "name": "Summer LoRA"})
    assert "Could not measure this run" in dialog._totals.text()
    assert "No training run" in dialog._totals.text()
    assert not dialog._prune_btn.isEnabled()


# --- pruning -----------------------------------------------------------------


def test_prune_names_the_count_and_the_gigabytes_before_deleting(app, worker, monkeypatch) -> None:
    dialogs = Dialogs(
        monkeypatch, {"question": QMessageBox.StandardButton.Yes}, storage_module
    )
    dialog = _dialog(app, worker)
    dialog._keep.setValue(1)
    dialog._prune()
    kind, title, body = dialogs.last()
    assert kind == "question" and title == "Prune checkpoints"
    assert "1 older checkpoint" in body and "1.0 GB" in body
    # and what survives: the newest, plus the one that became an adapter
    assert "newest 1" in body and "1 installed as adapters" in body
    assert "optimiser state" in body  # a step folder goes whole, not just the weights
    assert worker.pruned == [("run-1", 1, True), ("run-1", 1, False)]
    assert "Freed 1.0 GB" in dialog._result.text()


def test_prune_does_nothing_and_says_so_when_everything_is_already_kept(app, worker) -> None:
    dialog = _dialog(app, worker)  # keep=3, three checkpoints
    dialog._prune()
    assert worker.pruned == [("run-1", 3, True)]  # planned, and stopped there
    assert "Nothing to prune" in dialog._result.text()


def test_prune_says_no_is_no(app, worker) -> None:
    dialog = _dialog(app, worker)
    dialog._keep.setValue(1)
    dialog._prune()  # autouse fixture answers No
    assert worker.pruned == [("run-1", 1, True)]  # nothing but the measurement


def test_a_live_run_has_no_delete_buttons_at_all(app, worker) -> None:
    """Not "they warn" — they are disabled. A live trainer holds these files
    open, and on Windows the delete would half-finish."""
    dialog = _dialog(app, worker, status="running")
    for button in (dialog._prune_btn, dialog._cache_btn, dialog._delete_btn):
        assert not button.isEnabled()
        assert "Training is live" in button.toolTip()
    assert "Training is live" in dialog._result.text()
    assert dialog._export_btn.isEnabled()  # reading is always safe


def test_clear_caches_names_the_size_and_the_consequence(app, worker, monkeypatch) -> None:
    dialogs = Dialogs(
        monkeypatch, {"question": QMessageBox.StandardButton.Yes}, storage_module
    )
    dialog = _dialog(app, worker)
    dialog._clear_caches()
    assert "3.0 GB" in dialogs.last()[2] and "derived data" in dialogs.last()[2].lower()
    assert worker.cleared == ["run-1"]
    assert "Freed 3.0 GB of caches" in dialog._result.text()


def test_clearing_a_cache_that_is_already_gone_is_a_fact_not_an_error(app, worker, monkeypatch) -> None:
    Dialogs(monkeypatch, {"question": QMessageBox.StandardButton.Yes}, storage_module)
    worker.storage_report = {**_report(), "cache_bytes": 0, "bytes": 3 * _GB}
    dialog = StorageDialog(worker, {"id": "run-1", "name": "Summer LoRA", "path": "/runs/run-1"})
    worker.clear_train_cache("run-1")  # the second call reports nothing freed
    dialog._clear_caches()
    assert "No cache folder to clear" in dialog._result.text()


def test_delete_warns_that_installed_adapters_are_copies(app, worker, monkeypatch) -> None:
    dialogs = Dialogs(
        monkeypatch, {"question": QMessageBox.StandardButton.Yes}, storage_module
    )
    dialog = _dialog(app, worker)
    fired = []
    dialog.run_deleted.connect(lambda: fired.append(True))
    dialog._delete_run()
    assert "6.0 GB" in dialogs.last()[2]
    assert "they are copies" in dialogs.last()[2]
    assert fired == [True]
    assert worker.freed == ["run-1"]
    assert not dialog._delete_btn.isEnabled()  # there is no folder left to delete


# --- the resume picker -------------------------------------------------------


def test_resume_defaults_to_the_newest_checkpoint(app, worker, monkeypatch) -> None:
    dialogs = Dialogs(
        monkeypatch, {"question": QMessageBox.StandardButton.Yes}, storage_module
    )
    dialog = _dialog(app, worker)
    dialog._resume_selected()
    assert "checkpoints/step-800/lora.safetensors" in dialogs.last()[2]
    assert worker.resumed == [("run-1", "/runs/run-1/checkpoints/step-800/lora.safetensors")]
    assert "pid 909" in dialog._result.text()


def test_resume_uses_the_highlighted_checkpoint(app, worker, monkeypatch) -> None:
    Dialogs(monkeypatch, {"question": QMessageBox.StandardButton.Yes}, storage_module)
    dialog = _dialog(app, worker)
    dialog._tree.setCurrentItem(dialog._tree.topLevelItem(1))  # step-400, the installed one
    dialog._resume_selected()
    assert worker.resumed == [("run-1", "/runs/run-1/checkpoints/step-400/lora.safetensors")]


def test_resume_is_closed_to_a_live_run(app, worker) -> None:
    dialog = _dialog(app, worker, status="running")
    assert not dialog._resume_btn.isEnabled()
    assert "Training is live" in dialog._resume_btn.toolTip()


def test_resume_without_a_checkpoint_says_what_is_missing(app, worker, monkeypatch) -> None:
    Dialogs(monkeypatch, {"question": QMessageBox.StandardButton.Yes}, storage_module)
    worker.storage_report = {**_report(), "checkpoints": []}
    dialog = StorageDialog(worker, {"id": "run-1", "name": "Summer LoRA", "path": "/runs/run-1"})
    dialog._resume_selected()
    assert "not written a checkpoint" in dialog._result.text()
    assert worker.resumed == []


def test_resume_refusal_is_shown_not_buried(app, worker, monkeypatch) -> None:
    Dialogs(monkeypatch, {"question": QMessageBox.StandardButton.Yes}, storage_module)

    def refuse(run_id, checkpoint=None):
        raise RuntimeError("SimpleTuner is not installed — run: pip install 'minimax-studio[train]'.")

    worker.resume_train_run = refuse  # type: ignore[method-assign]
    dialog = _dialog(app, worker)
    dialog._resume_selected()
    assert "SimpleTuner is not installed" in dialog._result.text()


# --- moving a run ------------------------------------------------------------


def test_export_reports_where_it_went_and_that_caches_stayed(app, worker, monkeypatch) -> None:
    Dialogs(
        monkeypatch,
        {"question": QMessageBox.StandardButton.Yes, "folder": "/mnt/backup"},
        storage_module,
    )
    dialog = _dialog(app, worker)
    dialog._export()
    assert worker.exported == [("run-1", "/mnt/backup", False)]
    assert "/mnt/backup/run-1" in dialog._result.text()
    assert "caches left behind" in dialog._result.text()


def test_export_can_be_asked_for_the_cache_too(app, worker, monkeypatch) -> None:
    Dialogs(monkeypatch, {"folder": "/mnt/backup"}, storage_module)
    dialog = _dialog(app, worker)
    dialog._include_cache.setChecked(True)
    dialog._export()
    assert worker.exported == [("run-1", "/mnt/backup", True)]
    assert "caches left behind" not in dialog._result.text()


def test_a_cancelled_export_is_not_an_error(app, worker, monkeypatch) -> None:
    Dialogs(monkeypatch, {"folder": ""}, storage_module)
    dialog = _dialog(app, worker)
    dialog._export()
    assert worker.exported == []
    assert dialog._result.text() == ""


def test_a_worker_refusal_is_printed_verbally_not_swallowed(app, worker, monkeypatch) -> None:
    """409 carries a sentence written for a person; the dialog shows it."""
    Dialogs(monkeypatch, {"question": QMessageBox.StandardButton.Yes}, storage_module)

    def refuse(*args, **kwargs):
        raise RuntimeError(
            "Run “Summer LoRA” is still training (pid 4112) — not pruning files "
            "under it while it lives."
        )

    worker.prune_train_checkpoints = refuse  # type: ignore[method-assign]
    dialog = _dialog(app, worker)
    dialog._keep.setValue(1)
    dialog._prune()
    assert "still training (pid 4112)" in dialog._result.text()


# --- the Train page's side ---------------------------------------------------


def _page(app, worker) -> "train_module.TrainPage":
    worker.runs = [{"id": "run-1", "name": "Summer LoRA", "status": "completed", "path": "/runs/run-1"}]
    page = train_module.TrainPage(worker)
    page._runs_tree.setCurrentItem(page._runs_tree.topLevelItem(0))
    return page


def _run_ids(page) -> list[str]:
    from PySide6.QtCore import Qt

    tree = page._runs_tree
    return [
        str(tree.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole))
        for i in range(tree.topLevelItemCount())
    ]


def test_resume_is_offered_only_to_a_run_that_stopped(app, worker) -> None:
    worker.detail = {"status": "completed"}
    page = _page(app, worker)
    assert page._resume_btn.isEnabled()
    worker.detail = {"status": "running"}
    page.refresh()
    assert not page._resume_btn.isEnabled()  # two trainers, one folder: corruption
    worker.detail = {"status": "failed", "checkpoints": []}
    page.refresh()
    assert not page._resume_btn.isEnabled()  # and there is nothing to resume from


def test_resume_says_which_checkpoint_it_will_use(app, worker, monkeypatch) -> None:
    dialogs = Dialogs(
        monkeypatch, {"question": QMessageBox.StandardButton.Yes}, train_module
    )
    page = _page(app, worker)
    page._resume_run()
    # The worker resumes `latest` (mtime), not the lexicographic last path, so
    # the box must not name a specific file — Storage… is that picker.
    body = dialogs.last()[2]
    assert "newest checkpoint" in body
    assert "Storage" in body
    assert "step-800" not in body
    assert worker.resumed == [("run-1", None)]
    assert "resume #2" in page._form_status.text()
    assert "pid 909" in page._form_status.text()


def test_resume_failure_reaches_the_user(app, worker, monkeypatch) -> None:
    dialogs = Dialogs(
        monkeypatch, {"question": QMessageBox.StandardButton.Yes}, train_module
    )

    def refuse(run_id, checkpoint=None):
        raise RuntimeError("SimpleTuner is not installed — run: pip install 'minimax-studio[train]'.")

    worker.resume_train_run = refuse  # type: ignore[method-assign]
    page = _page(app, worker)
    page._resume_run()
    assert "not installed" in dialogs.bodies()


def test_storage_button_opens_the_dialog_for_the_selected_run(app, worker, monkeypatch) -> None:
    opened: list[tuple[str, str]] = []

    class StubSignal:
        def connect(self, slot):
            self.slot = slot

        def emit(self):
            self.slot()

    class NoopDialog:
        def __init__(self, client, run, parent=None) -> None:
            self.run_deleted = StubSignal()
            opened.append((str(run.get("id")), type(parent).__name__))

        def exec(self) -> int:
            worker.delete_train_run("run-1")  # pretend the folder got deleted
            self.run_deleted.emit()
            return 0

    monkeypatch.setattr(train_module, "StorageDialog", NoopDialog)
    page = _page(app, worker)
    page._open_storage()
    assert opened == [("run-1", "TrainPage")]
    # The delete reached the page: the ghost is gone from the list.
    assert page._selected_run is None


def test_import_picks_up_a_folder_and_selects_the_run(app, worker, monkeypatch) -> None:
    Dialogs(monkeypatch, {"folder": "/mnt/backup/run-9"}, train_module)
    page = _page(app, worker)
    page._import_run()
    assert worker.imported_runs == ["/mnt/backup/run-9"]
    assert page._selected_run == "run-9"
    assert "Imported “Imported LoRA”" in page._form_status.text()
    assert _run_ids(page) == ["run-9", "run-1"]


def test_import_refusal_names_the_folder_problem(app, worker, monkeypatch) -> None:
    dialogs = Dialogs(monkeypatch, {"folder": "/mnt/backup/checkpoints"}, train_module)

    def refuse(folder):
        raise RuntimeError(
            f"{folder} is not a Studio training run — no state.json inside."
        )

    worker.import_train_run = refuse  # type: ignore[method-assign]
    page = _page(app, worker)
    page._import_run()
    assert dialogs.kinds()[-1] == "warning"
    assert "no state.json" in dialogs.bodies()


def test_a_deleted_run_leaves_the_list(app, worker) -> None:
    """The dialog deletes the folder; the page must stop listing a ghost."""
    page = _page(app, worker)
    assert _run_ids(page) == ["run-1"]
    worker.delete_train_run("run-1")  # what the dialog's button ends up doing
    page._run_was_deleted()
    assert page._selected_run is None
    assert page._runs_tree.topLevelItemCount() == 0
