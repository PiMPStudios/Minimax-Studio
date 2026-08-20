from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from minimax_studio.worker.runtime import runtime


def lora_dirs() -> list[Path]:
    root = runtime.config.models_root()
    return [
        root / "loras",
        root / "h3-comfy" / "loras",
        root / "music3-cuda",
    ]


def list_loras() -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for folder in lora_dirs():
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*.safetensors")):
            key = path.name.lower()
            if key in seen:
                continue
            if "turbo" not in path.name.lower() and "lora" not in path.name.lower():
                # still include turbo and anything in the dedicated loras folder
                if folder.name != "loras":
                    continue
            seen.add(key)
            rows.append({"id": path.name, "name": path.stem, "path": str(path)})
    return rows


def import_lora(src: str) -> dict[str, Any]:
    source = Path(src)
    if not source.is_file():
        raise FileNotFoundError(src)
    dest_dir = runtime.config.models_root() / "loras"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / source.name
    shutil.copy2(source, dest)
    return {"id": dest.name, "name": dest.stem, "path": str(dest)}
