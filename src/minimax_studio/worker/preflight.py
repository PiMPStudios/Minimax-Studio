from __future__ import annotations

from typing import Any

from minimax_studio.worker.ping import ping_services
from minimax_studio.worker.probe import probe
from minimax_studio.worker.runtime import runtime


def preflight(kind: str, backend: str = "auto", mode: str = "t2va") -> dict[str, Any]:
    hw = probe()
    ping = ping_services()
    result: dict[str, Any] = {
        "kind": kind,
        "mode": mode,
        "requested": backend,
        "ok": False,
        "backend": None,
        "detail": "",
        "comfy": ping.get("comfy"),
        "torch_available": hw.get("torch_available"),
        "cuda": hw.get("cuda"),
        "gpus": hw.get("gpus") or [],
        "cuda_device": runtime.config.cuda_device,
        "comfy_url": runtime.config.comfy_url,
    }
    try:
        if kind == "h3":
            from minimax_studio.worker.backends.h3 import resolve_h3_backend

            resolved = resolve_h3_backend(backend)
        elif kind == "music":
            from minimax_studio.worker.backends.music import resolve_music_backend

            resolved = resolve_music_backend(backend)
        else:
            raise RuntimeError(f"unknown kind: {kind}")
    except RuntimeError as exc:
        result["detail"] = str(exc)
        return result

    result["backend"] = resolved
    if resolved == "cuda" and not hw.get("torch_available"):
        result["detail"] = (
            "Local diffusers needs PyTorch in the Studio venv "
            "(`pip install torch` in .venv). Comfy-Org INT8 still works if you "
            f"start ComfyUI at {runtime.config.comfy_url}."
        )
        return result
    result["ok"] = True
    if resolved == "comfy":
        result["detail"] = (
            f"Will generate via ComfyUI at {runtime.config.comfy_url}. "
            "Comfy uses the GPU it was launched with (--default-device), not Studio’s CUDA device."
        )
    elif resolved == "cuda":
        device = int(runtime.config.cuda_device or 0)
        result["detail"] = f"Will generate in-process on CUDA device {device}."
    elif resolved == "api":
        result["detail"] = "Will generate via the MiniMax API."
    elif resolved == "mlx":
        result["detail"] = "Will generate via mlx-audio on Apple Silicon."
    elif resolved == "stub":
        result["detail"] = "Will write a stub tone (dev)."
    else:
        result["detail"] = f"Will generate via {resolved}."
    return result
