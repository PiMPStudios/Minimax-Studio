from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from minimax_studio.worker.runtime import runtime


def _path() -> Path:
    return Path(runtime.config.output_dir or ".") / "presets.json"


def list_presets() -> list[dict[str, Any]]:
    path = _path()
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return []


def save_preset(payload: dict[str, Any]) -> dict[str, Any]:
    items = list_presets()
    item = {
        "id": payload.get("id") or uuid.uuid4().hex[:10],
        "name": payload.get("name") or "Untitled",
        "created_at": time.time(),
        "kind": payload.get("kind", "music"),
        "backend": payload.get("backend", "auto"),
        "mode": payload.get("mode"),
        "prompt": payload.get("prompt", ""),
        "lyrics": payload.get("lyrics", ""),
        "duration_s": payload.get("duration_s", 30),
        "seed": payload.get("seed", -1),
        "steps": payload.get("steps", 30),
        "width": payload.get("width", 960),
        "height": payload.get("height", 544),
        "resolution": payload.get("resolution", "768P"),
        "ratio": payload.get("ratio", "16:9"),
        "speed": payload.get("speed", "quality"),
        "attention": payload.get("attention", "default"),
        "ref_image_size": payload.get("ref_image_size", "match"),
        "assets": payload.get("assets") or [],
        "loras": payload.get("loras") or [],
        "lora_id": payload.get("lora_id") or "",
        "lora_strength": payload.get("lora_strength", 1.0),
    }
    items = [row for row in items if row.get("id") != item["id"]]
    items.append(item)
    dest = _path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(items, indent=2), encoding="utf-8")
    return item


def delete_preset(preset_id: str) -> None:
    items = [row for row in list_presets() if row.get("id") != preset_id]
    _path().write_text(json.dumps(items, indent=2), encoding="utf-8")
