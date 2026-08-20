from __future__ import annotations

from typing import Any

from minimax_studio.worker.runtime import runtime


def select_cuda_device() -> str:
    """Pin in-process torch to Settings → CUDA device. No-op without CUDA."""
    index = max(0, int(runtime.config.cuda_device or 0))
    try:
        import torch
    except ImportError:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    count = int(torch.cuda.device_count())
    if count <= 0:
        return "cpu"
    if index >= count:
        index = 0
    torch.cuda.set_device(index)
    return f"cuda:{index}"


def selected_vram_gb(hw: dict[str, Any]) -> float:
    gpus = hw.get("gpus") or []
    index = max(0, int(runtime.config.cuda_device or 0))
    if gpus:
        if index >= len(gpus):
            index = 0
        return float(gpus[index].get("vram_gb") or 0)
    return float(hw.get("vram_gb") or 0)
