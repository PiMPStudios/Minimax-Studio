from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from minimax_studio.worker.runtime import runtime


def lora_dirs() -> list[Path]:
    from minimax_studio.worker.model_paths import search_roots

    root = runtime.config.models_root()
    dirs = [
        root / "loras",
        root / "h3-comfy" / "loras",
        root / "music3-cuda",
    ]
    for extra in search_roots(root, runtime.config.comfy_models_dir):
        dirs.extend(
            [
                extra / "loras",
                extra / "h3-comfy" / "loras",
                extra / "minimax-h3" / "loras",
            ]
        )
    seen: set[str] = set()
    unique: list[Path] = []
    for folder in dirs:
        key = str(folder)
        if key in seen:
            continue
        seen.add(key)
        unique.append(folder)
    return unique


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


def _free_lora_path(dest_dir: Path, name: str) -> Path:
    target = dest_dir / name
    if not target.exists():
        return target
    stem, suffix = Path(name).stem, Path(name).suffix
    index = 2
    while (dest_dir / f"{stem}-{index}{suffix}").exists():
        index += 1
    return dest_dir / f"{stem}-{index}{suffix}"


def import_lora(
    src: str, dest_name: str | None = None, kind: str | None = None
) -> dict[str, Any]:
    source = Path(src)
    if not source.is_file():
        raise FileNotFoundError(src)
    if source.suffix.lower() != ".safetensors":
        raise RuntimeError("Only .safetensors files can be imported as a LoRA.")
    dest_dir = runtime.config.models_root() / "loras"
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = dest_name or source.name
    if Path(name).suffix.lower() != ".safetensors":
        name = f"{Path(name).stem}.safetensors"
    dest = _free_lora_path(dest_dir, Path(name).name)
    shutil.copy2(source, dest)
    row = {"id": dest.name, "name": dest.stem, "path": str(dest)}
    # PLAN-V2 S3: an import is provenance too — "we did not train this" is a
    # fact the picker should say out loud instead of leaving to memory.
    from minimax_studio.worker import adapters

    resolved = kind if kind in {"music", "h3"} else adapters.kind_from_path(source)
    adapters.record_imported({**row, "kind": resolved})
    return row
