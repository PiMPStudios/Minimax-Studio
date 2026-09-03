"""GUI process boundary: no worker import, no substring-matched disk copy."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GUI_FILES = sorted((ROOT / "src" / "minimax_studio" / "ui").rglob("*.py")) + [
    ROOT / "src" / "minimax_studio" / "app.py",
    ROOT / "src" / "minimax_studio" / "worker_client.py",
]

_WORKER_IMPORT = re.compile(
    r"^\s*(from|import)\s+minimax_studio\.worker\b", re.MULTILINE
)


def test_gui_never_imports_the_worker() -> None:
    offenders = [
        str(path.relative_to(ROOT))
        for path in GUI_FILES
        if _WORKER_IMPORT.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_gui_does_not_substring_match_disk_copy() -> None:
    hits = [
        str(path.relative_to(ROOT))
        for path in GUI_FILES
        if "Not enough free disk" in path.read_text(encoding="utf-8")
    ]
    assert hits == []
    ui = ROOT / "src" / "minimax_studio" / "ui"
    models = (ui / "pages" / "models_page.py").read_text(encoding="utf-8")
    adapters = (ui / "pages" / "adapters_page.py").read_text(encoding="utf-8")
    assert "confirm_and_download" in models
    assert "confirm_and_download" in adapters
