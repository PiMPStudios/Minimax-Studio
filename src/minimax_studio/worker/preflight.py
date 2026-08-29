from __future__ import annotations

from typing import Any

from minimax_studio.worker.probe import probe
from minimax_studio.worker.runtime import runtime


def preflight(
    kind: str,
    backend: str = "auto",
    mode: str = "t2va",
    speed: str = "quality",
    resolution: str = "768P",
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
    # 2K is a MiniMax API feature; local diffusers and ComfyUI render at 768P.
    # Say so here instead of silently downgrading at generate time.
    if (
        kind == "h3"
        and str(resolution or "").strip().upper() in {"2K", "1440P", "2048"}
        and resolved != "api"
    ):
        result["detail"] = (
            "2K is a MiniMax API feature — local H3 (diffusers and ComfyUI "
            "INT8) renders at 768P. Switch Inspector Backend to API "
            "(key in Settings) or pick 768P."
        )
        return result
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
        if kind == "music":
            result["detail"] += (
                " The Music 3 endpoint takes prompt + lyrics only — Duration, "
                "Seed, Steps and CFG shape local generation, not this call."
            )
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
    # A detached trainer holds the GPU for hours and does not know about us.
    # Say so here, where pressing Generate is still free.
    try:
        from minimax_studio.worker.train_runs import live_runs

        live = live_runs()
    except Exception:
        live = []
    if live:
        names = ", ".join(str(row.get("name") or row.get("id")) for row in live[:3])
        warning = (
            f"{len(live)} training run{' is' if len(live) == 1 else 's are'} "
            f"live ({names}) and want the whole GPU — a generation now can OOM "
            "one or stall the other. Stop it on Train LoRA for a clean shot."
        )
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
