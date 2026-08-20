from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from minimax_studio.worker.catalog import PACKS
from minimax_studio.worker.downloads import pack_status
from minimax_studio.worker.jobs import JobRequest, update_job
from minimax_studio.worker.runtime import runtime


def generate_music(job_id: str, request: JobRequest) -> dict[str, Any]:
    backend = _resolve_backend(request.backend)
    dest = runtime.config.history_root() / job_id
    dest.mkdir(parents=True, exist_ok=True)
    wav_path = dest / "audio.wav"
    if backend == "stub":
        update_job(job_id, message="Writing stub tone", progress=0.5)
        _write_stub(wav_path, min(float(request.duration_s), 1.0))
        return {"output_path": str(wav_path), "backend": "stub", "media_type": "audio"}
    if backend == "api":
        raise RuntimeError("MiniMax Music API is not wired yet. Use Local.")
    if backend == "mlx":
        return _generate_mlx(job_id, request, wav_path)
    return _generate_cuda(job_id, request, wav_path)


def _resolve_backend(requested: str) -> str:
    name = requested.lower()
    if name == "stub" or os.environ.get("MINIMAX_STUDIO_STUB") == "1":
        return "stub"
    if name in {"auto", "local"}:
        root = runtime.config.models_root()
        cuda = pack_status(PACKS["music3-cuda"], root)["ready"]
        mlx = pack_status(PACKS["music3-mlx"], root)["ready"]
        from minimax_studio.worker.probe import probe

        hw = probe()
        if name == "local" or name == "auto":
            if hw.get("cuda") and cuda:
                return "cuda"
            if hw.get("apple_silicon") and mlx:
                return "mlx"
            if cuda:
                return "cuda"
            if mlx:
                return "mlx"
            if os.environ.get("MINIMAX_STUDIO_STUB") == "1":
                return "stub"
            raise RuntimeError(
                "No Music 3 pack is installed. Open Models and download "
                "MiniMax-Music3 (CUDA) or the MLX pack."
            )
    if name in {"cuda", "mlx", "api", "stub"}:
        return name
    raise RuntimeError(f"unknown music backend: {requested}")


def _generate_cuda(job_id: str, request: JobRequest, wav_path: Path) -> dict[str, Any]:
    from minimax_studio.worker.catalog import PACKS

    pack_dir = runtime.config.models_root() / PACKS["music3-cuda"].local_dir
    if not pack_status(PACKS["music3-cuda"], runtime.config.models_root())["ready"]:
        raise RuntimeError("Download the MiniMax-Music3 CUDA pack first.")
    update_job(job_id, message="Loading MiniMax-Music3", progress=0.15)
    try:
        import torch
        from diffusers import ModularPipeline
    except ImportError as exc:
        raise RuntimeError(
            "Local Music 3 needs torch and diffusers in this Python env. "
            "Install: pip install torch diffusers accelerate soundfile"
        ) from exc

    pipe = runtime.music_pipe
    if pipe is None or runtime.music_pipe_path != str(pack_dir):
        pipe = ModularPipeline.from_pretrained(str(pack_dir))
        pipe.load_components(dtype=torch.bfloat16)
        if torch.cuda.is_available():
            try:
                pipe.to("cuda")
            except Exception:
                from diffusers import ComponentsManager

                manager = ComponentsManager()
                manager.enable_auto_cpu_offload(device="cuda")
                pipe = ModularPipeline.from_pretrained(
                    str(pack_dir), components_manager=manager
                )
                pipe.load_components(dtype=torch.bfloat16)
        runtime.music_pipe = pipe
        runtime.music_pipe_path = str(pack_dir)

    seed = int(request.seed)
    generator = None
    if seed >= 0:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device).manual_seed(seed)

    update_job(job_id, message="Sampling", progress=0.35)
    _apply_loras(pipe, request.loras)
    audio = pipe(
        prompt=request.prompt,
        lyrics=request.lyrics,
        audio_duration=float(request.duration_s),
        num_inference_steps=int(request.steps),
        generator=generator,
        output="audios",
    )[0]
    update_job(job_id, message="Writing WAV", progress=0.9)
    array = audio.T.float().cpu().numpy() if hasattr(audio, "T") else np.asarray(audio)
    rate = int(getattr(pipe, "sampling_rate", 32000))
    sf.write(str(wav_path), array, rate)
    return {"output_path": str(wav_path), "backend": "cuda", "media_type": "audio"}


def _generate_mlx(job_id: str, request: JobRequest, wav_path: Path) -> dict[str, Any]:
    from minimax_studio.worker.catalog import PACKS

    pack_dir = runtime.config.models_root() / PACKS["music3-mlx"].local_dir
    if not pack_dir.exists():
        raise RuntimeError("Download the MiniMax-Music3 MLX pack first.")
    update_job(job_id, message="Loading mlx-audio Music 3", progress=0.2)
    try:
        from mlx_audio.music import load
    except ImportError as exc:
        raise RuntimeError(
            "Mac Music 3 needs mlx-audio. "
            "See https://github.com/Blaizzy/mlx-audio"
        ) from exc
    model = load(str(pack_dir))
    update_job(job_id, message="Sampling", progress=0.4)
    result = next(
        model.generate(
            text=request.prompt,
            lyrics=request.lyrics or "[instrumental]",
            duration=float(request.duration_s),
            steps=int(request.steps),
            seed=None if request.seed < 0 else int(request.seed),
        )
    )
    update_job(job_id, message="Writing WAV", progress=0.9)
    audio = np.asarray(result.audio)
    rate = int(getattr(result, "sample_rate", 44100))
    if audio.ndim == 1:
        pass
    elif audio.shape[0] == 2 and audio.shape[1] != 2:
        audio = audio.T
    sf.write(str(wav_path), audio, rate)
    return {"output_path": str(wav_path), "backend": "mlx", "media_type": "audio"}


def _apply_loras(pipe: Any, loras: list[dict[str, Any]]) -> None:
    if not loras:
        return
    loader = getattr(pipe, "load_lora_weights", None)
    if loader is None:
        return
    for item in loras:
        path = item.get("id") or item.get("path")
        if not path:
            continue
        try:
            loader(path)
        except Exception:
            continue


def _write_stub(path: Path, duration_s: float) -> None:
    import struct
    import wave

    rate = 32000
    n = max(1, int(rate * duration_s))
    t = np.linspace(0, duration_s, n, endpoint=False)
    samples = (0.08 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"".join(struct.pack("<hh", int(s), int(s)) for s in samples))
