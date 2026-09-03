"""Adapters — provenance and the audition loop (PLAN-V2 S3).

Every ``.safetensors`` the app can load is listed here, whether Studio trained
it, you imported it, the Catalog downloaded it, or it was simply already in a
LoRA folder — ``trained`` / ``imported`` / ``catalog`` / ``untracked`` say
which, and the untracked ones are the honest part: the picker has always
loaded them, so pretending not to know would be a lie by omission.

The button that closes the loop is **Audition**: Music gets a 30-second song,
H3 a short still-pair (or text-to-video if the dataset has no stills), both at
0.8 strength with the caption the adapter was actually trained on, queued as an
ordinary job and badged in History.
"""

from __future__ import annotations

import html
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from minimax_studio.ui.ready import ask_lora_family
from minimax_studio.ui.reveal import reveal_path
from minimax_studio.worker_client import WorkerClient

_OK = QBrush(QColor("#32d74b"))
_WARN = QBrush(QColor("#ff9f0a"))
_BAD = QBrush(QColor("#ff453a"))
_DIM = QBrush(QColor("#8e8e93"))

SOURCE_LABEL = {
    "trained": "trained here",
    "imported": "imported",
    "catalog": "from catalog",
    "untracked": "found on disk",
}


def _when(timestamp: Any) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(timestamp)))
    except (TypeError, ValueError):
        return "—"


def _detail_text(row: dict[str, Any]) -> str:
    """Everything a filename hides. Blank fields are left out rather than
    shown as “None”, which is how inventories start lying."""
    lines = [f"<b>{html.escape(str(row.get('name')))}</b>"]
    bits = [SOURCE_LABEL.get(str(row.get("source")), str(row.get("source") or "?"))]
    if row.get("created_at"):
        bits.append(_when(row.get("created_at")))
    if not row.get("on_disk"):
        bits.append("<span style='color:#ff453a'>file is gone</span>")
    lines.append("  ·  ".join(bits))

    dataset = row.get("dataset") or {}
    if dataset.get("path"):
        lines.append("")
        lines.append(
            f"<b>Trained on</b> {html.escape(str(Path(str(dataset.get('path'))).name))}"
            f" — {dataset.get('clip_count', 0)} clips"
            + (
                f" · manifest {dataset.get('manifest_hash')}"
                if dataset.get("manifest_hash")
                else ""
            )
        )
        if not row.get("dataset_exists"):
            lines.append(
                "<span style='color:#ff9f0a'>That dataset folder is gone — the "
                "provenance still reads, but an audition needs a prompt typed "
                "in.</span>"
            )
        else:
            lines.append(
                f"<span style='color:#8e8e93'>{html.escape(str(dataset.get('path')))}</span>"
            )
    for label, key in (
        ("Run", "run_name"),
        ("Trainer", "trainer"),
        ("Base pack", "base_pack"),
        ("Preset", "preset"),
        ("Rank", "rank"),
        ("Steps", "steps"),
    ):
        value = row.get(key)
        if value not in (None, ""):
            lines.append(f"<b>{label}:</b> {html.escape(str(value))}")
    if row.get("path"):
        lines.append(
            f"<span style='color:#8e8e93'>{html.escape(str(row.get('path')))}</span>"
        )
    return "<br>".join(lines)


class AdaptersPage(QWidget):
    def __init__(self, client: WorkerClient) -> None:
        super().__init__()
        self._client = client
        self._rows: list[dict[str, Any]] = []
        self._visible: list[dict[str, Any]] = []
        self._selected: str | None = None
        self._poll_thread = None

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        title = QLabel("Adapters")
        title.setObjectName("pageTitle")
        subtitle = QLabel(
            "Every LoRA the picker can load, with where it came from. "
            "<b>Audition</b> queues a short render at 0.8 strength using the "
            "caption the adapter was trained on — 30 seconds that tell you "
            "whether the run was worth it."
        )
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)

        cat_title = QLabel("Catalog")
        cat_title.setObjectName("pageSubtitle")
        cat_help = QLabel(
            "Known LoRAs you can download — not a store, not a scrape. "
            "H3 rows use the MiniMax H3 Community License (US/EU/UK/KR need a "
            "separate grant). Music stays Import a file; that catalog is still thin."
        )
        cat_help.setObjectName("pageSubtitle")
        cat_help.setWordWrap(True)
        root.addWidget(cat_title)
        root.addWidget(cat_help)
        self._catalog = QTreeWidget()
        self._catalog.setHeaderLabels(["Adapter", "Family", "Size", "Status"])
        self._catalog.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._catalog.setMaximumHeight(140)
        self._catalog.currentItemChanged.connect(lambda *_: self._catalog_selected())
        root.addWidget(self._catalog)
        cat_row = QHBoxLayout()
        self._cat_download_btn = QPushButton("Download")
        self._cat_download_btn.clicked.connect(self._download_catalog)
        self._cat_cancel_btn = QPushButton("Cancel")
        self._cat_cancel_btn.hide()
        self._cat_cancel_btn.clicked.connect(self._cancel_catalog)
        self._cat_remove_btn = QPushButton("Remove")
        self._cat_remove_btn.clicked.connect(self._remove_catalog)
        cat_row.addWidget(self._cat_download_btn)
        cat_row.addWidget(self._cat_cancel_btn)
        cat_row.addWidget(self._cat_remove_btn)
        cat_row.addStretch(1)
        root.addLayout(cat_row)
        self._catalog_rows: list[dict[str, Any]] = []
        self._catalog_jobs: dict[str, Any] = {}

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Adapter", "Source", "Kind", "Trained on", "When"])
        self._tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._tree.currentItemChanged.connect(lambda *_: self._row_selected())
        root.addWidget(self._tree, 3)

        self._detail = QLabel("Select an adapter.")
        self._detail.setObjectName("pageSubtitle")
        self._detail.setWordWrap(True)
        self._detail.setTextFormat(Qt.TextFormat.RichText)
        self._detail.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        root.addWidget(self._detail)

        row = QHBoxLayout()
        self._prompt = QLineEdit()
        self._prompt.setPlaceholderText("Audition prompt — blank reuses the dataset caption")
        self._duration = QSpinBox()
        self._duration.setRange(5, 120)
        self._duration.setValue(30)
        self._duration.setSuffix(" s")
        self._duration.setToolTip(
            "Auditions are short on purpose — Music 30 s, H3 5–15 s."
        )
        self._audition_btn = QPushButton("Audition")
        self._audition_btn.setObjectName("primary")
        self._audition_btn.clicked.connect(self._audition)
        self._import_btn = QPushButton("Import .safetensors…")
        self._import_btn.clicked.connect(self._import)
        self._reveal_btn = QPushButton("Show in folder")
        self._reveal_btn.clicked.connect(self._reveal)
        self._forget_btn = QPushButton("Forget")
        self._forget_btn.setToolTip(
            "Remove Studio's provenance row for this file. The .safetensors "
            "itself is left alone."
        )
        self._forget_btn.clicked.connect(self._forget)
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self.refresh)
        row.addWidget(self._prompt, 1)
        row.addWidget(self._duration)
        row.addWidget(self._audition_btn)
        row.addWidget(self._import_btn)
        row.addWidget(self._reveal_btn)
        row.addWidget(self._forget_btn)
        row.addWidget(self._refresh_btn)
        root.addLayout(row)

        filters = QHBoxLayout()
        self._mine_only = QCheckBox("Only what Studio trained")
        self._mine_only.toggled.connect(self._rebuild_rows)
        self._missing_only = QCheckBox("Only missing files")
        self._missing_only.setToolTip(
            "Registry rows whose .safetensors was deleted from disk."
        )
        self._missing_only.toggled.connect(self._rebuild_rows)
        filters.addWidget(self._mine_only)
        filters.addWidget(self._missing_only)
        filters.addStretch(1)
        root.addLayout(filters)

        self._status = QLabel("")
        self._status.setObjectName("pageSubtitle")
        self._status.setWordWrap(True)
        root.addWidget(self._status)
        self._set_buttons_enabled(False)
        self._cat_download_btn.setEnabled(False)
        self._cat_remove_btn.setEnabled(False)
        self._cat_cancel_btn.hide()
        self.refresh()

    # --- data ---------------------------------------------------------------

    def refresh(self) -> None:
        try:
            rows = self._client.list_adapters()
        except Exception:
            self._status.setText("Could not list adapters — keeping the last list.")
            return
        self._rows = rows
        self._rebuild_rows()
        self._refresh_catalog()

    def _refresh_catalog(self) -> None:
        list_cat = getattr(self._client, "list_adapter_catalog", None)
        if not callable(list_cat):
            return
        try:
            catalog = list_cat()
            downloads: dict[str, Any] = {}
            list_dl = getattr(self._client, "list_downloads", None)
            if callable(list_dl):
                downloads = {
                    item["pack_id"]: item
                    for item in list_dl()
                    if isinstance(item, dict) and item.get("pack_id")
                }
        except Exception:
            return
        self._sync_catalog(catalog, downloads)

    def _catalog_job(self, pack: dict[str, Any] | None) -> dict[str, Any] | None:
        if not pack:
            return None
        job = self._catalog_jobs.get(str(pack.get("id")))
        if job and job.get("status") in {"queued", "running", "cancelling"}:
            return job
        return None

    def _apply_catalog(
        self, catalog: list[dict[str, Any]], downloads: dict[str, Any]
    ) -> None:
        self._catalog_jobs = downloads
        self._catalog_rows = catalog
        wanted = None
        current = self._catalog.currentItem()
        if current is not None:
            wanted = current.data(0, Qt.ItemDataRole.UserRole)
        self._catalog.blockSignals(True)
        self._catalog.clear()
        for pack in catalog:
            size = float(pack.get("approx_gb") or 0)
            size_s = f"{size * 1024:.0f} MB" if size < 1 else f"{size:.1f} GB"
            job = downloads.get(str(pack.get("id")))
            status = "Ready" if pack.get("ready") else "Not downloaded"
            if job and job.get("status") in {"queued", "running", "cancelling"}:
                status = str(job.get("message") or job.get("status") or "Downloading")
            elif job and job.get("status") == "error":
                status = "Failed"
            item = QTreeWidgetItem(
                [
                    str(pack.get("title") or pack.get("id")),
                    str(pack.get("family") or ""),
                    size_s,
                    status,
                ]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, str(pack.get("id")))
            item.setToolTip(0, str(pack.get("summary") or ""))
            if pack.get("ready"):
                item.setForeground(3, _OK)
            self._catalog.addTopLevelItem(item)
        self._catalog.blockSignals(False)
        restore = next(
            (
                index
                for index in range(self._catalog.topLevelItemCount())
                if self._catalog.topLevelItem(index).data(0, Qt.ItemDataRole.UserRole)
                == wanted
            ),
            None,
        )
        if restore is not None:
            self._catalog.setCurrentItem(self._catalog.topLevelItem(restore))
        elif self._catalog.topLevelItemCount():
            self._catalog.setCurrentItem(self._catalog.topLevelItem(0))
        self._catalog_selected()

    def _catalog_current(self) -> dict[str, Any] | None:
        item = self._catalog.currentItem()
        if item is None:
            return None
        pack_id = item.data(0, Qt.ItemDataRole.UserRole)
        return next((row for row in self._catalog_rows if row.get("id") == pack_id), None)

    def _catalog_selected(self) -> None:
        pack = self._catalog_current()
        stored = self._catalog_jobs.get(str(pack.get("id"))) if pack else None
        job = self._catalog_job(pack)
        busy = job is not None
        ready = bool(pack and pack.get("ready"))
        failed = bool(stored and stored.get("status") == "error")
        self._cat_download_btn.setEnabled(bool(pack) and not ready and not busy)
        self._cat_remove_btn.setEnabled(ready and not busy)
        if busy:
            self._cat_download_btn.setText("Downloading…")
        elif ready:
            self._cat_download_btn.setText("Re-download")
        elif failed:
            self._cat_download_btn.setText("Retry")
        else:
            self._cat_download_btn.setText("Download")
        self._cat_cancel_btn.setVisible(busy)
        cancelling = bool(job and job.get("status") == "cancelling")
        self._cat_cancel_btn.setEnabled(busy and not cancelling)
        self._cat_cancel_btn.setText("Cancelling…" if cancelling else "Cancel")

    def _download_catalog(self) -> None:
        pack = self._catalog_current()
        if not pack or self._catalog_job(pack):
            return
        from minimax_studio.ui.download import confirm_and_download

        job = confirm_and_download(self, self._client, pack, noun="adapter")
        if job is None:
            return
        pack_id = str(pack["id"])
        self._catalog_jobs[pack_id] = job
        self._status.setText(f"Downloading “{pack.get('title')}”…")
        self._catalog_selected()
        self._refresh_catalog()

    def _cancel_catalog(self) -> None:
        pack = self._catalog_current()
        job = self._catalog_job(pack)
        job_id = str((job or {}).get("id") or "")
        if not job_id:
            return
        try:
            self._client.cancel_download(job_id)
        except Exception as exc:
            QMessageBox.warning(self, "Cancel failed", str(exc))
            return
        self._status.setText("Cancel requested — Hugging Face may finish the current file.")
        self._refresh_catalog()

    def _remove_catalog(self) -> None:
        pack = self._catalog_current()
        if not pack:
            return
        answer = QMessageBox.question(
            self,
            "Remove adapter",
            f"Delete the Studio copy of “{pack.get('title')}”? "
            "Other LoRAs in the same folder are left alone.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._client.delete_pack(str(pack["id"]))
        except Exception as exc:
            QMessageBox.warning(self, "Remove failed", str(exc))
            return
        self._status.setText(f"Removed “{pack.get('title')}”.")
        self.refresh()

    def poll(self) -> None:
        """Timer path: list adapters off the GUI thread. Overlapping polls skip."""
        from minimax_studio.ui.enhance import start_background

        def work() -> tuple[list, list, dict]:
            catalog: list = []
            downloads: dict = {}
            rows = self._client.list_adapters()
            list_cat = getattr(self._client, "list_adapter_catalog", None)
            if callable(list_cat):
                catalog = list_cat()
            list_dl = getattr(self._client, "list_downloads", None)
            if callable(list_dl):
                downloads = {
                    item["pack_id"]: item
                    for item in list_dl()
                    if isinstance(item, dict) and item.get("pack_id")
                }
            return rows, catalog, downloads

        def done(payload: object) -> None:
            if not isinstance(payload, tuple) or len(payload) != 3:
                return
            rows, catalog, downloads = payload
            if isinstance(rows, list):
                signature = self._rows_signature(rows)
                previous = self._rows_signature(self._rows)
                self._rows = rows
                if signature != previous:
                    self._rebuild_rows()
            if isinstance(catalog, list):
                self._sync_catalog(
                    catalog, downloads if isinstance(downloads, dict) else {}
                )

        def fail() -> None:
            self._status.setText("Could not list adapters — keeping the last list.")

        start_background(self, work, done, fail, attr="_poll_thread")

    def _rows_signature(self, rows: list[dict[str, Any]]) -> list[tuple]:
        return [
            (
                row.get("id"),
                row.get("source"),
                row.get("kind"),
                row.get("on_disk"),
                row.get("can_audition"),
                row.get("created_at"),
            )
            for row in rows
        ]

    def _catalog_signature(
        self, catalog: list[dict[str, Any]], downloads: dict[str, Any]
    ) -> list[tuple]:
        return [
            (
                row.get("id"),
                row.get("ready"),
                row.get("bytes_on_disk"),
                (downloads.get(str(row.get("id"))) or {}).get("status"),
                (downloads.get(str(row.get("id"))) or {}).get("message"),
            )
            for row in catalog
        ]

    def _sync_catalog(
        self, catalog: list[dict[str, Any]], downloads: dict[str, Any]
    ) -> None:
        """Rebuild the catalog tree only when ready/size/download text changed.

        Poll ticks every ~2 s; clear()+rebuild flickers, drops the click, and
        closes the tooltip. Same skip the installed-adapter list already uses.
        """
        signature = self._catalog_signature(catalog, downloads)
        previous = self._catalog_signature(self._catalog_rows, self._catalog_jobs)
        if signature != previous:
            self._apply_catalog(catalog, downloads)
            return
        self._catalog_jobs = downloads
        self._catalog_selected()

    def _rebuild_rows(self) -> None:
        wanted = self._selected
        self._visible = [
            row
            for row in self._rows
            if (not self._mine_only.isChecked() or row.get("source") == "trained")
            and (not self._missing_only.isChecked() or not row.get("on_disk"))
        ]
        self._tree.blockSignals(True)
        self._tree.clear()
        for row in self._visible:
            dataset = row.get("dataset") or {}
            trained_on = (
                f"{dataset.get('clip_count', 0)} clips"
                if dataset.get("path")
                else "—"
            )
            if dataset.get("manifest_hash"):
                trained_on += f" · {dataset['manifest_hash']}"
            item = QTreeWidgetItem(
                [
                    str(row.get("name")),
                    SOURCE_LABEL.get(str(row.get("source")), str(row.get("source") or "?")),
                    str(row.get("kind") or "?"),
                    trained_on,
                    _when(row.get("created_at")),
                ]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, str(row.get("id")))
            source = str(row.get("source"))
            if not row.get("on_disk"):
                item.setForeground(0, _BAD)
            elif source == "trained":
                item.setForeground(1, _OK)
            elif source == "imported":
                item.setForeground(1, _WARN)
            else:
                item.setForeground(1, _DIM)
            self._tree.addTopLevelItem(item)
        self._tree.blockSignals(False)
        restore = next(
            (
                index
                for index in range(self._tree.topLevelItemCount())
                if self._tree.topLevelItem(index).data(0, Qt.ItemDataRole.UserRole)
                == wanted
            ),
            None,
        )
        if restore is not None:
            self._tree.setCurrentItem(self._tree.topLevelItem(restore))
        elif self._tree.topLevelItemCount():
            self._tree.setCurrentItem(self._tree.topLevelItem(0))
        else:
            self._selected = None
            self._show()

    def _current(self) -> dict[str, Any] | None:
        return next(
            (row for row in self._visible if row.get("id") == self._selected), None
        )

    def _row_selected(self) -> None:
        item = self._tree.currentItem()
        self._selected = item.data(0, Qt.ItemDataRole.UserRole) if item else None
        self._show()

    def _show(self) -> None:
        row = self._current()
        self._set_buttons_enabled(bool(row))
        if not row:
            self._detail.setText(
                "Nothing here yet. Train a LoRA and <b>Install adapter</b>, or "
                "import a .safetensors — either way it shows up in the picker "
                "on Generate Music too."
            )
            self._prompt.setPlaceholderText("Audition prompt")
            return
        self._detail.setText(_detail_text(row))
        caption = str(row.get("audition_prompt") or "")
        self._prompt.setPlaceholderText(
            (
                f"Audition prompt — blank reuses “{caption[:70]}”"
                if caption
                else "No dataset caption behind this one — type an audition prompt"
            )
        )
        kind = str(row.get("kind") or "music")
        if kind == "h3":
            self._duration.setRange(5, 15)
            self._duration.setValue(5)
            self._duration.setToolTip(
                "H3 auditions snap to the 5–15 s frame grid. 5 s is the check."
            )
        else:
            self._duration.setRange(5, 120)
            if self._duration.value() < 15:
                self._duration.setValue(30)
            self._duration.setToolTip(
                "Auditions are short on purpose — you are checking for the voice, "
                "not mastering a track."
            )

    def _set_buttons_enabled(self, have_selection: bool) -> None:
        row = self._current()
        for button in (self._reveal_btn, self._forget_btn):
            button.setEnabled(have_selection)
        self._audition_btn.setEnabled(bool(row and row.get("can_audition")))
        if row and not row.get("can_audition"):
            self._audition_btn.setToolTip(
                "The file is not on disk — reinstall it from its run."
            )
        elif row and row.get("kind") == "h3":
            self._audition_btn.setToolTip(
                "Short H3 generate at 0.8 strength: still-pair when the dataset "
                "has frames, otherwise text-to-video. Lands in History."
            )
        else:
            self._audition_btn.setToolTip(
                "One short generate job at 0.8 strength, badged in History."
            )

    # --- actions ------------------------------------------------------------

    def _audition(self) -> None:
        row = self._current()
        if not row:
            return
        try:
            queued = self._client.audition_adapter(
                str(row["id"]),
                prompt=self._prompt.text().strip(),
                duration_s=float(self._duration.value()),
            )
        except Exception as exc:
            QMessageBox.warning(self, "Audition did not start", str(exc))
            return
        self._prompt.clear()
        seconds = float(queued.get("duration_s") or 0)
        length = f"{seconds:g} s clip" if (queued.get("kind") or row.get("kind")) == "h3" else f"{int(seconds)} s"
        self._status.setText(
            f"Audition queued for “{row.get('name')}” — "
            f"{length} at "
            f"{float(queued.get('strength') or 0):.1f} strength, job "
            f"{queued.get('job_id')}. It lands in History badged as an "
            "audition, and Restore to Generate works on it like any take."
        )

    def _import(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Import a LoRA",
            str(Path.home()),
            "LoRA files (*.safetensors)",
        )
        if not path:
            return
        kind = ask_lora_family(self, path)
        if kind is None:
            return
        try:
            row = self._client.import_lora(path, kind=kind)
        except Exception as exc:
            QMessageBox.warning(self, "Import failed", str(exc))
            return
        self._status.setText(
            f"Imported “{row.get('name')}”. It is listed as imported — Studio "
            "knows it did not train it, and the picker has it."
        )
        self.refresh()

    def _reveal(self) -> None:
        row = self._current()
        path = str((row or {}).get("path") or "")
        if not path or not Path(path).is_file():
            QMessageBox.information(
                self, "Show in folder", "This adapter file is not on disk."
            )
            return
        try:
            reveal_path(path)
        except OSError as exc:
            QMessageBox.warning(self, "Show in folder failed", str(exc))

    def _forget(self) -> None:
        row = self._current()
        if not row:
            return
        answer = QMessageBox.question(
            self,
            "Forget adapter",
            f"Forget Studio’s record of “{row.get('name')}”?\n\n"
            "The .safetensors stays on disk and stays in the LoRA picker — it "
            "just comes back as “found on disk”, with no provenance.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._client.forget_adapter(str(row["id"]))
        except Exception as exc:
            QMessageBox.warning(self, "Forget failed", str(exc))
            return
        self._selected = None
        self.refresh()
