from __future__ import annotations

from typing import Any

from minimax_studio.worker.probe import probe
from minimax_studio.worker.runtime import runtime


def preflight(
    kind: str, backend: str = "auto", mode: str = "t2va", speed: str = "quality"
) -> dict[str, Any]:
    hw = probe()
    result: dict[str, Any] = {
        "kind": kind,
        "mode": mode,
        "requested": backend,
        "ok": False,
        "backend": None,
        "detail": "",
        "comfy": None,
        "torch_available": hw.get("torch_available"),
        "cuda": hw.get("cuda"),
        "gpus": hw.get("gpus") or [],
        "cuda_device": runtime.config.cuda_device,
        "comfy_url": runtime.config.comfy_url,
        "ffmpeg": bool(hw.get("ffmpeg")),
        "warnings": [],
    }
    try:
        if kind == "h3":
            from minimax_studio.worker.backends.h3 import resolve_h3_backend

            resolved = resolve_h3_backend(
                backend, "ref2va" if mode == "ref2va" else "fl2va"
            )
        elif kind == "music":
            from minimax_studio.worker.backends.music import resolve_music_backend

            resolved = resolve_music_backend(backend)
        else:
            raise RuntimeError(f"unknown kind: {kind}")
    except RuntimeError as exc:
        result["detail"] = str(exc)
        return result

    result["backend"] = resolved
    if resolved == "comfy":
        result["comfy"] = {"ok": True, "detail": runtime.config.comfy_url}
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
    if (
        kind == "h3"
        and resolved in {"cuda", "comfy"}
        and not result["ffmpeg"]
    ):
        warning = "ffmpeg is not in PATH. Install it for MP4 mux and media probe."
        result["warnings"].append(warning)
        result["detail"] = f"{result['detail']} {warning}"
    if kind == "h3" and speed.strip().lower() == "fast":
        from minimax_studio.worker.backends.h3 import _find_turbo_lora

        turbo = _find_turbo_lora(mode)
        if not turbo:
            result["ok"] = False
            result["turbo"] = False
            result["detail"] = (
                "Fast needs the MiniMax H3 Turbo LoRA. Download it on Models "
                "(MiniMax H3 Turbo LoRA), or switch Inspector Speed to Quality."
            )
            return result
        result["turbo"] = True
        steps = "4 steps" if mode == "ref2va" else "8 steps"
        result["detail"] = f"{result['detail']} Fast: Turbo LoRA, {steps}."
    return result
