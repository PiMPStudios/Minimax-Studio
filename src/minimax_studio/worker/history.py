from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from minimax_studio.worker.fsutil import atomic_write_text
from minimax_studio.worker.runtime import runtime
from minimax_studio.worker.trim import VIDEO_EXTS, trim_media


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


def trim_entry(entry_id: str, start_s: float, end_s: float) -> dict[str, Any]:
    """Cut ``[start_s, end_s)`` of a take into a **new** History row.

    The original is never mutated. The child keeps prompt/lyrics/loras/mode
    so Restore to Generate still works. ``trimmed_from`` is the parent id.
    """
    parent = get_entry(entry_id)
    src = Path(str(parent.get("output_path") or ""))
    if not src.is_file():
        raise RuntimeError("This take has no file on disk to trim.")
    root = runtime.config.history_root().resolve()
    new_id = _fresh_id(root)
    name = _trim_filename(src)
    dest = (root / new_id / name).resolve()
    if dest != root and root not in dest.parents:
        raise RuntimeError("refusing to write a trim outside History")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        cut = trim_media(src, dest, start_s, end_s)
    except Exception:
        shutil.rmtree(dest.parent, ignore_errors=True)
        raise
    skip = {"id", "created_at", "dir", "output_path", "duration_s"}
    child = {key: value for key, value in parent.items() if key not in skip}
    child.update(
        {
            "id": new_id,
            "output_path": str(dest),
            "duration_s": cut["duration_s"],
            "trimmed_from": parent["id"],
            "trim_start_s": cut["start_s"],
            "trim_end_s": cut["end_s"],
        }
    )
    return record_entry(child)


def _fresh_id(root: Path) -> str:
    for _ in range(8):
        candidate = uuid.uuid4().hex[:12]
        if not (root / candidate).exists():
            return candidate
    raise RuntimeError("could not allocate a History id")


def _trim_filename(src: Path) -> str:
    name = src.name
    if Path(name).name != name or name in {".", ".."}:
        return "video.mp4" if src.suffix.lower() in VIDEO_EXTS else "audio.wav"
    return name


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
