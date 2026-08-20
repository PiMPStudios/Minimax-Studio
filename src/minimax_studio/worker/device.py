from __future__ import annotations

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
