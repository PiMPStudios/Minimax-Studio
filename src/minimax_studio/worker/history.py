from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from minimax_studio.worker.runtime import runtime


def history_index_path() -> Path:
    return runtime.config.history_root() / "index.jsonl"


def record_entry(entry: dict[str, Any]) -> dict[str, Any]:
    root = runtime.config.history_root()
    root.mkdir(parents=True, exist_ok=True)
    item_dir = root / entry["id"]
    item_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **entry,
        "created_at": entry.get("created_at") or time.time(),
        "dir": str(item_dir),
    }
    (item_dir / "meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with history_index_path().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    return payload


def list_history(limit: int = 200) -> list[dict[str, Any]]:
    path = history_index_path()
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    rows.reverse()
    return rows[:limit]


def get_entry(entry_id: str) -> dict[str, Any]:
    meta = runtime.config.history_root() / entry_id / "meta.json"
    if not meta.is_file():
        raise KeyError(entry_id)
    return json.loads(meta.read_text(encoding="utf-8"))
