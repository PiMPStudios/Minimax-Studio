"""Infer Music vs H3 from a LoRA path. GUI-safe — no worker import."""

from __future__ import annotations

from pathlib import Path

# Folder names that mean this file is an H3 adapter, not Music.
_H3_LORA_FOLDERS = {"h3-comfy", "minimax-h3"}


def kind_from_path(path: str | Path) -> str:
    """Infer adapter family from the folders a .safetensors lives in."""
    parts = {part.lower() for part in Path(path).parts}
    if parts & _H3_LORA_FOLDERS:
        return "h3"
    return "music"
