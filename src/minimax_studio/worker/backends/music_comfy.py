from __future__ import annotations

import random
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from minimax_studio.worker.backends.h3_comfy import (
    _first_media,
    _poll_history,
    _submit_graph,
    comfy_base,
    comfy_missing_message,
    comfy_reachable,
    comfy_resolve_file,
)
from minimax_studio.worker.jobs import JobRequest, update_job
from minimax_studio.worker.runtime import runtime

DIT_INT8 = "minimax_music3_dit_int8_convrot.safetensors"
DIT_FP16 = "minimax_music3_dit_fp16.safetensors"
CLIP_INT8 = "minimax_music3_text_encoder_pruned_int8_convrot.safetensors"
VAE_NAME = "minimax_music3_dav.safetensors"


def comfy_music_files_missing() -> list[str]:
    """Music 3 model files missing from Comfy's model roots ([] = OK/unknown).

    The fp16 DiT counts as a substitute for the INT8 one.
    """
    dit = comfy_resolve_file("UNETLoader", "unet_name", DIT_INT8) or comfy_resolve_file(
        "UNETLoader", "unet_name", DIT_FP16
    )
    clip = comfy_resolve_file("CLIPLoader", "clip_name", CLIP_INT8)
    vae = comfy_resolve_file("VAELoader", "vae_name", VAE_NAME)
    missing: list[str] = []
    if dit is None:
        missing.append(f"{DIT_INT8} (or {DIT_FP16})")
    if clip is None:
        missing.append(CLIP_INT8)
    if vae is None:
        missing.append(VAE_NAME)
    return missing


def _pick_music_model_names() -> tuple[str, str, str]:
    """Resolve (dit, clip, vae) names exactly as Comfy lists them."""
    dit = comfy_resolve_file("UNETLoader", "unet_name", DIT_INT8) or DIT_FP16
    clip = comfy_resolve_file("CLIPLoader", "clip_name", CLIP_INT8) or CLIP_INT8
    vae = comfy_resolve_file("VAELoader", "vae_name", VAE_NAME) or VAE_NAME
    return dit, clip, vae


def generate_music_comfy(job_id: str, request: JobRequest) -> dict[str, Any]:
    if not comfy_reachable():
        raise RuntimeError(
            "Comfy-Org Music 3 INT8 needs a running ComfyUI. "
            f"Start it at {comfy_base()} (Settings → ComfyUI URL), "
            "or download the official MiniMax-Music3 CUDA pack."
        )
    dest = runtime.config.history_root() / job_id
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / "audio.wav"
    seed = int(request.seed) if request.seed >= 0 else random.randint(0, 2_147_483_647)
    steps = int(request.steps) if request.steps else 30
    duration = max(1.0, min(300.0, float(request.duration_s)))
    caption = request.prompt or ""
    lyrics = request.lyrics or ""
    cfg = float(request.cfg or 1.7)
    update_job(job_id, message="Checking ComfyUI model roots", progress=0.1)
    missing = comfy_music_files_missing()
    if missing:
        raise RuntimeError(comfy_missing_message(missing))
    dit, clip_name, vae_name = _pick_music_model_names()
    graph = build_music_comfy_graph(
        caption=caption,
        lyrics=lyrics,
        duration_s=duration,
        seed=seed,
        steps=steps,
        cfg=cfg,
        dit_name=dit,
        clip_name=clip_name,
        vae_name=vae_name,
        prefix=f"minimax_studio/{job_id}",
    )
    update_job(job_id, message="Queued Music 3 on ComfyUI", progress=0.2)
    with httpx.Client(timeout=30.0) as client:
        submitted = _submit_graph(client, graph, client_id=job_id)
        if submitted.status_code >= 400 and DIT_INT8 in (submitted.text or ""):
            graph = build_music_comfy_graph(
                caption=caption,
                lyrics=lyrics,
                duration_s=duration,
                seed=seed,
                steps=steps,
                cfg=cfg,
                dit_name=DIT_FP16,
                clip_name=clip_name,
                vae_name=vae_name,
                prefix=f"minimax_studio/{job_id}",
            )
            submitted = _submit_graph(client, graph, client_id=job_id)
        if submitted.status_code >= 400:
            detail = submitted.text
            try:
                detail = str(submitted.json().get("error") or submitted.json())
            except Exception:
                pass
            raise RuntimeError(f"ComfyUI rejected the music graph: {detail}")
        prompt_id = submitted.json().get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI returned no prompt_id: {submitted.text}")
        outputs = _poll_history(client, prompt_id, job_id)
        media = _first_media(outputs)
        if not media:
            raise RuntimeError(f"ComfyUI finished without audio: {outputs}")
        update_job(job_id, message="Downloading ComfyUI audio", progress=0.92)
        query = urlencode(
            {
                "filename": media.get("filename") or "",
                "subfolder": media.get("subfolder") or "",
                "type": media.get("type") or "output",
            }
        )
        audio = client.get(f"{comfy_base()}/view?{query}", timeout=120.0)
        audio.raise_for_status()
        raw = audio.content
        _write_wav(out_path, raw, media.get("filename") or "out.flac")
    return {"output_path": str(out_path), "backend": "comfy", "media_type": "audio"}


def build_music_comfy_graph(
    *,
    caption: str,
    lyrics: str,
    duration_s: float,
    seed: int,
    steps: int,
    cfg: float = 1.7,
    dit_name: str = DIT_INT8,
    clip_name: str = CLIP_INT8,
    vae_name: str = VAE_NAME,
    prefix: str = "minimax_studio/music",
) -> dict[str, Any]:
    return {
        "unet": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": dit_name, "weight_dtype": "default"},
        },
        "clip": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": clip_name,
                "type": "minimax",
                "device": "default",
            },
        },
        "vae": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        },
        "encode": {
            "class_type": "MiniMaxMusic3TextEncode",
            "inputs": {
                "clip": ["clip", 0],
                "caption": caption,
                "lyrics": lyrics,
                "seed": int(seed),
                "max_duration": float(duration_s),
                "cfg_scale": float(cfg),
                "top_k": 50,
            },
        },
        "zero": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["encode", 0]},
        },
        "empty": {
            "class_type": "EmptyMiniMaxMusic3LatentAudio",
            "inputs": {
                "seconds": ["encode", 1],
                "batch_size": 1,
            },
        },
        "sample": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["unet", 0],
                "positive": ["encode", 0],
                "negative": ["zero", 0],
                "latent_image": ["empty", 0],
                "seed": int(seed),
                "steps": int(steps),
                "cfg": float(cfg),
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "decode": {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]},
        },
        "save": {
            "class_type": "SaveAudio",
            "inputs": {
                "audio": ["decode", 0],
                "filename_prefix": prefix,
            },
        },
    }


def _write_wav(dest: Path, payload: bytes, filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix == ".wav":
        dest.write_bytes(payload)
        return
    scratch = dest.with_suffix(suffix or ".flac")
    scratch.write_bytes(payload)
    try:
        import soundfile as sf

        audio, rate = sf.read(str(scratch))
        sf.write(str(dest), audio, rate)
    except Exception:
        dest.write_bytes(payload)
    finally:
        if scratch != dest and scratch.is_file() and dest.is_file() and dest.stat().st_size:
            try:
                scratch.unlink()
            except OSError:
                pass
