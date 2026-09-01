"""PLAN-V2 S3 UI: the Adapters page — provenance you can read, one-click audition."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from typing import Any

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from minimax_studio.ui.pages import adapters_page as adapters_module
from minimax_studio.ui.pages.adapters_page import AdaptersPage

TRAINED = {
    "id": "summer-lora.safetensors",
    "file": "summer-lora.safetensors",
    "name": "summer-lora",
    "kind": "music",
    "source": "trained",
    "on_disk": True,
    "can_audition": True,
    "dataset_exists": True,
    "audition_prompt": "moody folk with a warm chorus",
    "created_at": 1787000000,
    "path": "/models/loras/summer-lora.safetensors",
    "trainer": "simpletuner 4.8.0",
    "base_pack": "music3-cuda",
    "preset": "24g",
    "steps": 900,
    "rank": 16,
    "run_name": "Summer",
    "dataset": {
        "path": "/data/datasets/summer",
        "clip_count": 3,
        "manifest_hash": "a1b2c3d4e5f6",
        "exists": True,
    },
}
IMPORTED = {
    "id": "borrowed.safetensors",
    "file": "borrowed.safetensors",
    "name": "borrowed",
    "kind": "music",
    "source": "imported",
    "on_disk": True,
    "can_audition": True,
    "dataset_exists": False,
    "audition_prompt": "",
    "created_at": 1786999000,
    "path": "/models/loras/borrowed.safetensors",
    "dataset": {},
}
H3 = {
    "id": "h3-lora.safetensors",
    "file": "h3-lora.safetensors",
    "name": "h3-lora",
    "kind": "h3",
    "source": "trained",
    "on_disk": True,
    "can_audition": True,
    "dataset_exists": True,
    "audition_prompt": "a golden retriever puppy",
    "created_at": 1787000100,
    "path": "/models/loras/h3-lora.safetensors",
    "trainer": "simpletuner 4.8.0",
    "base_pack": "h3-diffusers-fl2va",
    "preset": "h3-24g",
    "steps": 50,
    "rank": 16,
    "run_name": "S0 metal h3 50",
    "dataset": {
        "path": "/data/datasets/s0-metal-h3",
        "clip_count": 3,
        "manifest_hash": "cb3be809ea3f",
        "exists": True,
    },
}
GONE = {
    "id": "ghost.safetensors",
    "file": "ghost.safetensors",
    "name": "ghost",
    "kind": "music",
    "source": "trained",
    "on_disk": False,
    "can_audition": False,
    "dataset_exists": False,
    "audition_prompt": "",
    "created_at": 1786998000,
    "path": "/models/loras/ghost.safetensors",
    "dataset": {"path": "/data/datasets/old", "clip_count": 9, "manifest_hash": "ff00ff"},
}


class FakeAdapterWorker:
    def __init__(self, **overrides: Any) -> None:
        self.rows: list[dict[str, Any]] = [dict(TRAINED), dict(IMPORTED), dict(GONE)]
        self.audition_error: str | None = None
        self.auditioned: list[tuple[str, str, float]] = []
        self.forgotten: list[str] = []
        self.imported: list[str] = []
        self.import_kinds: list[str | None] = []
        self.catalog: list[dict[str, Any]] = []
        self.started: list[str] = []
        self.deleted_packs: list[str] = []
        for key, value in overrides.items():
            setattr(self, key, value)

    def list_adapters(self) -> list[dict[str, Any]]:
        return self.rows

    def list_adapter_catalog(self) -> list[dict[str, Any]]:
        return self.catalog

    def list_downloads(self) -> list[dict[str, Any]]:
        return []

    def start_download(self, pack_id: str, force: bool = False) -> dict[str, Any]:
        self.started.append(pack_id)
        return {"id": "dl1", "pack_id": pack_id, "status": "queued"}

    def delete_pack(self, pack_id: str, delete_shared: bool = False) -> dict[str, Any]:
        self.deleted_packs.append(pack_id)
        return {"ok": True, "removed": True}

    def audition_adapter(
        self,
        adapter_id: str,
        prompt: str = "",
        duration_s: float | None = None,
        backend: str = "auto",
    ) -> dict[str, Any]:
        if self.audition_error:
            raise RuntimeError(self.audition_error)
        self.auditioned.append((adapter_id, prompt, float(duration_s or 0)))
        row = next((item for item in self.rows if item["id"] == adapter_id), {})
        kind = str(row.get("kind") or "music")
        return {
            "job_id": "job7",
            "adapter": adapter_id,
            "kind": kind,
            "prompt": prompt or "moody folk with a warm chorus",
            "strength": 0.8,
            "duration_s": float(duration_s or (5 if kind == "h3" else 30)),
        }

    def forget_adapter(self, adapter_id: str) -> dict[str, Any]:
        self.forgotten.append(adapter_id)
        self.rows = [row for row in self.rows if row["id"] != adapter_id]
        return {"ok": True}

    def import_lora(self, path: str, kind: str | None = None) -> dict[str, Any]:
        self.imported.append(path)
        self.import_kinds.append(kind)
        return {"id": "new.safetensors", "name": "new", "path": "/models/loras/new.safetensors"}


@pytest.fixture(autouse=True)
def no_modals(monkeypatch):
    """A modal in a test is a hung CI runner."""

    class Shim:
        StandardButton = QMessageBox.StandardButton

        def __init__(self, dialogs: list) -> None:
            self.dialogs = dialogs

        def warning(self, *_args):
            self.dialogs.append(("warning", _args[2] if len(_args) > 2 else ""))

        def information(self, *_args):
            self.dialogs.append(("information", _args[2] if len(_args) > 2 else ""))

        def question(self, *_args):
            self.dialogs.append(("question", _args[2] if len(_args) > 2 else ""))
            return QMessageBox.StandardButton.Yes

    dialogs: list = []
    monkeypatch.setattr(adapters_module, "QMessageBox", Shim(dialogs))
    return dialogs


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication

    from minimax_studio.ui.theme import apply_theme

    instance = QApplication.instance() or QApplication([])
    apply_theme(instance)
    return instance


def _select(page: AdaptersPage, adapter_id: str) -> None:
    for index in range(page._tree.topLevelItemCount()):
        item = page._tree.topLevelItem(index)
        if item.data(0, Qt.ItemDataRole.UserRole) == adapter_id:
            page._tree.setCurrentItem(item)
            return
    raise AssertionError(f"{adapter_id} is not in the list")


def test_adapter_list_survives_a_failed_refresh(app) -> None:
    worker = FakeAdapterWorker()
    page = AdaptersPage(worker)
    assert page._tree.topLevelItemCount() == 3

    def boom() -> list:
        raise RuntimeError("worker down")

    worker.list_adapters = boom  # type: ignore[method-assign]
    page.refresh()
    assert page._tree.topLevelItemCount() == 3
    assert "keeping the last list" in page._status.text()


def test_every_loadable_lora_is_listed_with_its_origin(app) -> None:
    page = AdaptersPage(FakeAdapterWorker())
    rows = [
        (
            page._tree.topLevelItem(i).text(0),
            page._tree.topLevelItem(i).text(1),
            page._tree.topLevelItem(i).text(3),
        )
        for i in range(page._tree.topLevelItemCount())
    ]
    assert ("summer-lora", "trained here", "3 clips · a1b2c3d4e5f6") in rows
    assert ("borrowed", "imported", "—") in rows
    assert any(name == "ghost" for name, _, _ in rows), "a deleted file still reads"


def test_detail_reads_like_a_provenance_label_not_a_dump(app) -> None:
    page = AdaptersPage(FakeAdapterWorker())
    _select(page, TRAINED["id"])
    detail = page._detail.text()
    assert "simpletuner 4.8.0" in detail
    assert "900" in detail and "16" in detail
    assert "manifest a1b2c3d4e5f6" in detail
    assert "/data/datasets/summer" in detail
    # Empty provenance is left out, not shown as "None".
    _select(page, IMPORTED["id"])
    assert "None" not in page._detail.text()
    assert "Trainer" not in page._detail.text()


def test_a_missing_file_is_said_in_red_not_shrugged_off(app) -> None:
    page = AdaptersPage(FakeAdapterWorker())
    _select(page, GONE["id"])
    assert "file is gone" in page._detail.text()
    assert not page._audition_btn.isEnabled()
    assert "not on disk" in page._audition_btn.toolTip()


def test_blank_prompt_advertises_the_caption_it_will_reuse(app) -> None:
    page = AdaptersPage(FakeAdapterWorker())
    _select(page, TRAINED["id"])
    assert "moody folk with a warm chorus" in page._prompt.placeholderText()
    _select(page, IMPORTED["id"])
    assert "type an audition prompt" in page._prompt.placeholderText().lower()


def test_an_h3_adapter_auditions_as_a_short_clip(app) -> None:
    worker = FakeAdapterWorker()
    worker.rows.append(dict(H3))
    page = AdaptersPage(worker)
    _select(page, H3["id"])
    assert page._audition_btn.isEnabled()
    assert page._duration.value() == 5
    assert page._duration.maximum() == 15
    page._audition_btn.click()
    assert worker.auditioned == [(H3["id"], "", 5.0)]
    assert "5 s clip" in page._status.text()
    assert "job job7" in page._status.text()


def test_audition_queues_the_selected_adapter(app) -> None:
    worker = FakeAdapterWorker()
    page = AdaptersPage(worker)
    _select(page, TRAINED["id"])
    page._duration.setValue(20)
    page._audition_btn.click()
    assert worker.auditioned == [(TRAINED["id"], "", 20.0)]
    status = page._status.text()
    assert "job job7" in status
    assert "0.8 strength" in status
    assert "History badged as an audition" in status
    assert "Restore to Generate" in status


def test_a_typed_prompt_is_sent_verbatim_and_the_box_is_cleared(app) -> None:
    worker = FakeAdapterWorker()
    page = AdaptersPage(worker)
    _select(page, IMPORTED["id"])
    page._prompt.setText("jangly indie, female vocal")
    page._audition_btn.click()
    assert worker.auditioned[0][1] == "jangly indie, female vocal"
    assert page._prompt.text() == ""


def test_a_refused_audition_is_shown_not_swallowed(app, no_modals) -> None:
    worker = FakeAdapterWorker(audition_error="Nothing to audition “borrowed”: it was imported by hand.")
    page = AdaptersPage(worker)
    _select(page, IMPORTED["id"])
    page._audition_btn.click()
    assert no_modals[-1][0] == "warning"
    assert "Nothing to audition" in no_modals[-1][1]
    assert "Audition queued" not in page._status.text()


def test_filters_separate_the_three_origins(app) -> None:
    page = AdaptersPage(FakeAdapterWorker())
    assert page._tree.topLevelItemCount() == 3
    page._mine_only.setChecked(True)
    assert [
        page._tree.topLevelItem(i).text(0) for i in range(page._tree.topLevelItemCount())
    ] == ["summer-lora", "ghost"]
    page._mine_only.setChecked(False)
    page._missing_only.setChecked(True)
    assert [
        page._tree.topLevelItem(i).text(0) for i in range(page._tree.topLevelItemCount())
    ] == ["ghost"]


def test_forget_asks_and_says_what_it_leaves_alone(app, no_modals) -> None:
    worker = FakeAdapterWorker()
    page = AdaptersPage(worker)
    _select(page, IMPORTED["id"])
    page._forget_btn.click()
    assert worker.forgotten == [IMPORTED["id"]]
    assert ".safetensors stays on disk" in no_modals[-1][1]


def test_import_bring_its_own_provenance_label(app, monkeypatch) -> None:
    worker = FakeAdapterWorker()
    monkeypatch.setattr(
        adapters_module.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *args, **kwargs: ("/home/me/cool.safetensors", "")),
    )
    monkeypatch.setattr(adapters_module, "ask_lora_family", lambda *a, **k: "music")
    page = AdaptersPage(worker)
    page._import_btn.click()
    assert worker.imported == ["/home/me/cool.safetensors"]
    assert worker.import_kinds == ["music"]
    assert "listed as imported" in page._status.text()


def test_catalog_lists_rows_and_download_asks_territory(app, no_modals) -> None:
    worker = FakeAdapterWorker(
        catalog=[
            {
                "id": "h3-realism-people",
                "title": "H3 Realism People (fal)",
                "family": "h3",
                "approx_gb": 0.13,
                "ready": False,
                "summary": "r34l1sm",
                "territory_notice": "US/EU/UK/KR",
                "license_name": "MiniMax H3 Community License",
            }
        ]
    )
    page = AdaptersPage(worker)
    assert page._catalog.topLevelItemCount() == 1
    assert "Realism People" in page._catalog.topLevelItem(0).text(0)
    page._catalog.setCurrentItem(page._catalog.topLevelItem(0))
    page._download_catalog()
    assert worker.started == ["h3-realism-people"]
    assert any(item[0] == "question" and "US/EU" in item[1] for item in no_modals)
