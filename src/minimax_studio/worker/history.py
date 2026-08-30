from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from minimax_studio.worker.fsutil import atomic_write_text
from minimax_studio.worker.runtime import runtime


def history_index_path() -> Path:
    return runtime.config.history_root() / "index.jsonl"


def _require_id(entry_id: str) -> str:
    name = str(entry_id or "")
    if not name or Path(name).name != name or name in {".", ".."}:
        raise KeyError(entry_id)
    return name


def record_entry(entry: dict[str, Any]) -> dict[str, Any]:
    root = runtime.config.history_root()
    root.mkdir(parents=True, exist_ok=True)
    entry_id = _require_id(str(entry["id"]))
    item_dir = root / entry_id
    item_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **entry,
        "id": entry_id,
        "created_at": entry.get("created_at") or time.time(),
        "dir": str(item_dir),
    }
    atomic_write_text(item_dir / "meta.json", json.dumps(payload, indent=2))
    with runtime.lock:
        index = history_index_path()
        if not index.is_file():
            rows = _rows_from_dirs()
            if not any(row.get("id") == entry_id for row in rows):
                rows.append(payload)
            _write_index(index, rows)
        else:
            with index.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
    return payload


def list_history(limit: int = 200) -> list[dict[str, Any]]:
    path = history_index_path()
    rows = _read_index(path) if path.is_file() else []
    if not rows:
        rows = _rows_from_dirs()
        if rows:
            try:
                _write_index(path, rows)
            except OSError:
                pass
    rows.sort(key=lambda item: item.get("created_at") or 0, reverse=True)
    return rows[:limit]


def delete_entry(entry_id: str) -> None:
    import shutil

    entry_id = _require_id(entry_id)
    root = runtime.config.history_root()
    item_dir = root / entry_id
    if item_dir.is_dir():
        shutil.rmtree(item_dir)
    index = history_index_path()
    with runtime.lock:
        if not index.is_file():
            return
        kept = [
            row
            for row in _read_index(index)
            if row.get("id") != entry_id
        ]
        _write_index(index, kept)


def get_entry(entry_id: str) -> dict[str, Any]:
    entry_id = _require_id(entry_id)
    meta = runtime.config.history_root() / entry_id / "meta.json"
    if not meta.is_file():
        raise KeyError(entry_id)
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KeyError(entry_id) from exc
    if not isinstance(payload, dict):
        raise KeyError(entry_id)
    return payload


def _read_index(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _write_index(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_text(
        path, "".join(json.dumps(row) + "\n" for row in rows)
    )


def _rows_from_dirs() -> list[dict[str, Any]]:
    try:
        root = runtime.config.history_root()
    except RuntimeError:
        return []
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for child in root.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        try:
            _require_id(child.name)
        except KeyError:
            continue
        meta = child / "meta.json"
        if not meta.is_file():
            continue
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows
