from __future__ import annotations

import random
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from minimax_studio.worker.jobs import JobRequest, is_cancelled, update_job
from minimax_studio.worker.runtime import runtime

UNET_FL2VA = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
CLIP_NAME = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"


def comfy_base() -> str:
    return (runtime.config.comfy_url or "http://127.0.0.1:8188").rstrip("/")


def comfy_reachable() -> bool:
    try:
        response = httpx.get(f"{comfy_base()}/system_stats", timeout=3.0)
        if response.status_code < 400:
            return True
    except httpx.HTTPError:
        pass
    try:
        response = httpx.get(f"{comfy_base()}/queue", timeout=3.0)
        return response.status_code < 400
    except httpx.HTTPError:
        return False


def generate_h3_comfy(job_id: str, request: JobRequest) -> dict[str, Any]:
    if request.mode == "ref2va":
        raise RuntimeError(
            "Reference mode is not wired through Comfy-Org INT8 yet. "
            "Download the official Ref2VA transformer pack, or use the MiniMax API."
        )
    if not comfy_reachable():
        raise RuntimeError(
            "Comfy-Org INT8 H3 needs a running ComfyUI. "
            f"Start it at {comfy_base()} (Settings → ComfyUI URL), "
            "or download the official FL2VA diffusers pack for in-process generate."
        )

    from minimax_studio.worker.backends.h3 import duration_to_frames, resolve_dims

    dest = runtime.config.history_root() / job_id
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / "video.mp4"
    frames = duration_to_frames(request.duration_s)
    width, height = resolve_dims(request.resolution, request.ratio, request.width, request.height)
    seed = int(request.seed) if request.seed >= 0 else random.randint(0, 2_147_483_647)
    steps = int(request.steps)
    lora_name = None
    lora_strength = 1.0
    if request.speed == "fast":
        if steps >= 16:
            steps = 8
        from minimax_studio.worker.backends.h3 import _find_turbo_lora

        turbo_path = _find_turbo_lora()
        if turbo_path:
            lora_name = Path(turbo_path).name
    elif request.loras:
        first = request.loras[0]
        path = first.get("id") or first.get("path")
        if path:
            lora_name = Path(str(path)).name
            lora_strength = float(first.get("strength") or 1.0)

    update_job(job_id, message="Uploading to ComfyUI", progress=0.12)
    first_name = _maybe_upload(request, "first_frame")
    last_name = _maybe_upload(request, "last_frame")
    graph = build_h3_comfy_graph(
        prompt=request.prompt,
        width=width,
        height=height,
        length=frames,
        seed=seed,
        steps=steps,
        first_image=first_name,
        last_image=last_name,
        lora_name=lora_name,
        lora_strength=lora_strength,
        prefix=f"minimax_studio/{job_id}",
    )
    client_id = uuid.uuid4().hex
    update_job(job_id, message="Queued on ComfyUI", progress=0.2)
    with httpx.Client(timeout=30.0) as client:
        submitted = client.post(
            f"{comfy_base()}/prompt",
            json={"prompt": graph, "client_id": client_id},
        )
        if submitted.status_code >= 400:
            detail = submitted.text
            try:
                detail = str(submitted.json().get("error") or submitted.json())
            except Exception:
                pass
            raise RuntimeError(f"ComfyUI rejected the graph: {detail}")
        prompt_id = submitted.json().get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI returned no prompt_id: {submitted.text}")
        outputs = _poll_history(client, prompt_id, job_id)
        media = _first_media(outputs)
        if not media:
            raise RuntimeError(f"ComfyUI finished without a video: {outputs}")
        update_job(job_id, message="Downloading ComfyUI result", progress=0.92)
        query = urlencode(
            {
                "filename": media.get("filename") or "",
                "subfolder": media.get("subfolder") or "",
                "type": media.get("type") or "output",
            }
        )
        video = client.get(f"{comfy_base()}/view?{query}", timeout=120.0)
        video.raise_for_status()
        out_path.write_bytes(video.content)
    return {"output_path": str(out_path), "backend": "comfy", "media_type": "video"}


def build_h3_comfy_graph(
    *,
    prompt: str,
    width: int,
    height: int,
    length: int,
    seed: int,
    steps: int,
    first_image: str | None = None,
    last_image: str | None = None,
    lora_name: str | None = None,
    lora_strength: float = 1.0,
    unet_name: str = UNET_FL2VA,
    prefix: str = "minimax_studio/h3",
) -> dict[str, Any]:
    use_lora = bool(lora_name)
    model_src = ["lora", 0] if use_lora else ["unet", 0]
    graph: dict[str, Any] = {
        "unet": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet_name, "weight_dtype": "default"},
        },
        "clip": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": CLIP_NAME,
                "type": "minimax",
                "device": "default",
            },
        },
        "vae": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VIDEO_VAE},
        },
        "avae": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": AUDIO_VAE},
        },
        "cond": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["clip", 0],
                "vae": ["vae", 0],
                "prompt": prompt,
                "width": int(width),
                "height": int(height),
                "length": int(length),
            },
        },
        "shift": {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {
                "model": model_src,
                "shift_video": 12.0,
                "shift_audio": 3.0,
            },
        },
        "noise": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": int(seed)},
        },
        "sampler": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "res_multistep"},
        },
        "sched": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": ["shift", 0],
                "scheduler": "simple",
                "steps": int(steps),
                "denoise": 1.0,
            },
        },
        "guide": {
            "class_type": "BasicGuider",
            "inputs": {
                "model": ["shift", 0],
                "conditioning": ["cond", 0],
            },
        },
        "sample": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["noise", 0],
                "guider": ["guide", 0],
                "sampler": ["sampler", 0],
                "sigmas": ["sched", 0],
                "latent_image": ["cond", 1],
            },
        },
        "vdec": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]},
        },
        "adec": {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["sample", 0], "vae": ["avae", 0]},
        },
        "video": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["vdec", 0],
                "audio": ["adec", 0],
                "fps": 24,
                "bit_depth": 8,
            },
        },
        "save": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["video", 0],
                "filename_prefix": prefix,
                "format": "auto",
                "codec": {"codec": "auto"},
            },
        },
    }
    if first_image:
        graph["first"] = {
            "class_type": "LoadImage",
            "inputs": {"image": first_image},
        }
        graph["cond"]["inputs"]["first_frame"] = ["first", 0]
    if last_image:
        graph["last"] = {
            "class_type": "LoadImage",
            "inputs": {"image": last_image},
        }
        graph["cond"]["inputs"]["last_frame"] = ["last", 0]
    if use_lora:
        graph["lora"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["unet", 0],
                "lora_name": lora_name,
                "strength_model": float(lora_strength),
            },
        }
    return graph


def _maybe_upload(request: JobRequest, role: str) -> str | None:
    for item in request.assets:
        if item.get("role") == role and item.get("path"):
            return _upload_image(item["path"])
    return None


def _upload_image(path: str) -> str:
    source = Path(path)
    if not source.is_file():
        raise RuntimeError(f"Missing image: {path}")
    with source.open("rb") as handle:
        response = httpx.post(
            f"{comfy_base()}/upload/image",
            files={"image": (source.name, handle, "application/octet-stream")},
            data={"overwrite": "true"},
            timeout=60.0,
        )
    response.raise_for_status()
    body = response.json()
    name = body.get("name") or source.name
    sub = body.get("subfolder") or ""
    return f"{sub}/{name}" if sub else str(name)


def _poll_history(client: httpx.Client, prompt_id: str, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 45 * 60
    while time.monotonic() < deadline:
        if is_cancelled(job_id):
            try:
                client.post(f"{comfy_base()}/interrupt")
            except httpx.HTTPError:
                pass
            raise RuntimeError("Cancelled")
        history = client.get(f"{comfy_base()}/history/{prompt_id}", timeout=15.0)
        if history.status_code == 404:
            time.sleep(1.5)
            continue
        history.raise_for_status()
        payload = history.json()
        record = payload.get(prompt_id) if isinstance(payload, dict) else None
        if record:
            status = record.get("status") or {}
            if status.get("status_str") == "error":
                messages = status.get("messages") or record.get("messages") or status
                raise RuntimeError(f"ComfyUI job failed: {messages}")
            outputs = record.get("outputs") or {}
            done = status.get("status_str") == "success" or status.get("completed") is True
            if done:
                return outputs
        update_job(job_id, message="Sampling on ComfyUI", progress=0.45)
        time.sleep(1.5)
    raise RuntimeError("Timed out waiting for ComfyUI.")


def _first_media(outputs: dict[str, Any]) -> dict[str, Any] | None:
    for node in outputs.values():
        if not isinstance(node, dict):
            continue
        for items in node.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict) and item.get("filename"):
                    return item
    return None
