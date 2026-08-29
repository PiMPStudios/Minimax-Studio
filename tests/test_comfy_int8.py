from pathlib import Path

import pytest

from minimax_studio.worker.backends import h3_comfy
from minimax_studio.worker.backends.h3 import INT8_NEEDS_COMFY, resolve_h3_backend
from minimax_studio.worker.backends.h3_comfy import (
    AUDIO_VAE,
    CLIP_NAME,
    UNET_FL2VA,
    UNET_REF2VA,
    VIDEO_VAE,
    _comfy_error_text,
    build_h3_comfy_graph,
    comfy_resolve_file,
)
from minimax_studio.worker.backends.music_comfy import DIT_INT8, build_music_comfy_graph
from minimax_studio.worker.catalog import PACKS
from minimax_studio.worker.jobs import JobRequest
from minimax_studio.worker.model_paths import pack_status, parse_extra_model_paths
from minimax_studio.worker.runtime import runtime


def _touch_int8(root: Path) -> None:
    files = PACKS["h3-fl2va"].marker_files
    for marker in files:
        path = root / marker
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


def test_pack_status_finds_nested_minimax_h3(tmp_path: Path) -> None:
    nested = tmp_path / "ComfyUI" / "models" / "minimax-h3"
    _touch_int8(nested)
    studio = tmp_path / "studio-models"
    studio.mkdir()
    status = pack_status(
        PACKS["h3-fl2va"],
        studio,
        extra_roots=[tmp_path / "ComfyUI" / "models"],
    )
    assert status["ready"] is True
    assert status["source"] == "comfy"
    assert "minimax-h3" in status["path"]
    assert status["missing"] == []


def test_pack_status_studio_dest_wins(tmp_path: Path) -> None:
    studio = tmp_path / "models"
    _touch_int8(studio / "h3-comfy")
    status = pack_status(PACKS["h3-fl2va"], studio, extra_roots=[studio])
    assert status["ready"] is True
    assert status["source"] == "studio"


def test_list_packs_isolated_from_host_comfy(studio_home: Path) -> None:
    from minimax_studio.worker.downloads import list_packs

    rows = {item["id"]: item for item in list_packs()}
    assert rows["h3-fl2va"]["ready"] is False


def test_pack_status_missing_stays_not_ready(tmp_path: Path) -> None:
    studio = tmp_path / "models"
    studio.mkdir()
    status = pack_status(PACKS["h3-fl2va"], studio, extra_roots=[studio])
    assert status["ready"] is False
    assert status["missing"]


def test_comfy_graph_t2va_has_core_nodes() -> None:
    graph = build_h3_comfy_graph(
        prompt="a red fox",
        width=1344,
        height=768,
        length=124,
        seed=1,
        steps=20,
    )
    assert graph["unet"]["inputs"]["unet_name"].endswith("int8_convrot.safetensors")
    assert graph["clip"]["inputs"]["type"] == "minimax"
    assert graph["cond"]["class_type"] == "MiniMaxH3ImageToVideo"
    assert "first" not in graph
    assert graph["sample"]["inputs"]["latent_image"] == ["cond", 1]
    assert graph["save"]["inputs"]["codec"] == "auto"
    assert graph["save"]["inputs"]["format"] == "auto"
    assert "lora" not in graph


def test_comfy_error_text_uses_exception_message() -> None:
    status = {
        "messages": [
            ["execution_start", {"prompt_id": "x"}],
            [
                "execution_error",
                {
                    "node_type": "SaveVideo",
                    "exception_message": "SaveVideo.execute() missing 1 required positional argument: 'codec'\n",
                },
            ],
        ]
    }
    text = _comfy_error_text(status)
    assert "SaveVideo" in text
    assert "codec" in text
    assert "execution_start" not in text


def test_comfy_graph_fl2va_uploads_and_lora() -> None:
    graph = build_h3_comfy_graph(
        prompt="x",
        width=960,
        height=544,
        length=73,
        seed=2,
        steps=8,
        first_image="start.png",
        last_image="end.png",
        lora_name="turbo.safetensors",
        lora_strength=0.8,
    )
    assert graph["cond"]["inputs"]["first_frame"] == ["first", 0]
    assert graph["cond"]["inputs"]["last_frame"] == ["last", 0]
    assert graph["shift"]["inputs"]["model"] == ["lora", 0]
    assert graph["lora"]["inputs"]["strength_model"] == 0.8


def test_comfy_graph_stacks_second_lora() -> None:
    graph = build_h3_comfy_graph(
        prompt="x",
        width=960,
        height=544,
        length=73,
        seed=2,
        steps=8,
        lora_name="turbo.safetensors",
        extra_loras=[{"id": "style.safetensors", "strength": 0.6}],
    )
    assert graph["lora"]["inputs"]["lora_name"] == "turbo.safetensors"
    assert graph["lora1"]["inputs"]["lora_name"] == "style.safetensors"
    assert graph["lora1"]["inputs"]["model"] == ["lora", 0]
    assert graph["shift"]["inputs"]["model"] == ["lora1", 0]


def test_comfy_graph_ref2va_wires_autogrow() -> None:
    graph = build_h3_comfy_graph(
        prompt="Use <Picture 1> as identity",
        width=1344,
        height=768,
        length=124,
        seed=3,
        steps=20,
        mode="ref2va",
        ref_images=["hero.png"],
        ref_videos=["walk.mp4"],
        ref_audios=["voice.wav"],
        unet_name=UNET_REF2VA,
        scheduler="beta",
        sage=True,
    )
    assert graph["unet"]["inputs"]["unet_name"] == UNET_REF2VA
    assert graph["cond"]["class_type"] == "MiniMaxH3ReferenceToVideo"
    assert graph["cond"]["inputs"]["ref_images.ref_image_0"] == ["img0", 0]
    assert graph["cond"]["inputs"]["ref_videos.ref_video_0"] == ["vcomp0", 0]
    assert graph["cond"]["inputs"]["ref_video_audios.ref_video_audio_0"] == ["vcomp0", 1]
    assert graph["cond"]["inputs"]["ref_audios.ref_audio_0"] == ["aud0", 0]
    assert graph["vcomp0"]["class_type"] == "GetVideoComponents"
    assert graph["sage"]["class_type"] == "PathchSageAttentionKJ"
    assert graph["shift"]["inputs"]["model"] == ["sage", 0]
    assert graph["sched"]["inputs"]["scheduler"] == "beta"


def test_parse_extra_model_paths_minimax_layout(tmp_path: Path) -> None:
    base = tmp_path / "models"
    (base / "minimax-h3" / "diffusion_models").mkdir(parents=True)
    yaml = f"""
# comment
tweaver_models:
  base_path: {base}
  is_default: false
  loras: loras/
  diffusion_models: |
    ltx/
    minimax-h3/diffusion_models/
  text_encoders: |
    minimax-h3/text_encoders/
"""
    roots = {str(path) for path in parse_extra_model_paths(yaml)}
    assert str(base) in roots
    assert str(base / "minimax-h3" / "diffusion_models") in roots
    assert str(base / "loras") in roots
    assert not any(item.endswith("false") for item in roots)


def test_pack_status_finds_nested_music3(tmp_path: Path) -> None:
    nested = tmp_path / "models" / "minimax-music-3"
    for marker in PACKS["music3-comfy"].marker_files:
        path = nested / marker
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    studio = tmp_path / "studio-models"
    studio.mkdir()
    status = pack_status(
        PACKS["music3-comfy"],
        studio,
        extra_roots=[tmp_path / "models"],
    )
    assert status["ready"] is True
    assert "minimax-music-3" in status["path"]


def test_music_comfy_graph_core_nodes() -> None:
    graph = build_music_comfy_graph(
        caption="lofi",
        lyrics="[Verse]\nhi",
        duration_s=30,
        seed=1,
        steps=30,
    )
    assert graph["unet"]["inputs"]["unet_name"] == DIT_INT8
    assert graph["clip"]["inputs"]["type"] == "minimax"
    assert graph["encode"]["class_type"] == "MiniMaxMusic3TextEncode"
    assert graph["empty"]["inputs"]["seconds"] == ["encode", 1]
    assert graph["sample"]["class_type"] == "KSampler"
    assert graph["save"]["class_type"] == "SaveAudio"


def test_mac_h3_local_is_gated(monkeypatch, studio_home: Path) -> None:
    monkeypatch.setattr(
        "minimax_studio.worker.probe.probe",
        lambda: {"apple_silicon": True, "cuda": False, "ram_gb": 32},
    )
    with pytest.raises(RuntimeError, match="Apple Silicon"):
        resolve_h3_backend("auto")
    with pytest.raises(RuntimeError, match="Apple Silicon"):
        resolve_h3_backend("local")


def test_auto_prefers_int8_on_10gb_when_comfy_up(monkeypatch, studio_home: Path) -> None:
    monkeypatch.setattr(
        "minimax_studio.worker.backends.h3.pack_status",
        lambda pack, root: {
            "ready": pack.id in {"h3-diffusers-fl2va", "h3-fl2va"},
            "path": "x",
        },
    )
    monkeypatch.setattr(
        "minimax_studio.worker.probe.probe",
        lambda: {
            "cuda": True,
            "torch_available": True,
            "gpus": [{"name": "RTX 3080", "vram_gb": 10.0}],
            "vram_gb": 10.0,
        },
    )
    monkeypatch.setattr(
        "minimax_studio.worker.backends.h3_comfy.comfy_reachable",
        lambda: True,
    )
    assert resolve_h3_backend("auto") == "comfy"


def test_auto_backend_prefers_official(monkeypatch, studio_home: Path) -> None:
    monkeypatch.setattr(
        "minimax_studio.worker.backends.h3.pack_status",
        lambda pack, root: {"ready": pack.id == "h3-diffusers-fl2va", "path": "x"},
    )
    monkeypatch.setattr(
        "minimax_studio.worker.probe.probe",
        lambda: {"cuda": True, "torch_available": True},
    )
    assert resolve_h3_backend("auto") == "cuda"


def test_auto_backend_int8_without_comfy_is_clear(monkeypatch, studio_home: Path) -> None:
    monkeypatch.setattr(
        "minimax_studio.worker.backends.h3.pack_status",
        lambda pack, root: {
            "ready": pack.id == "h3-fl2va",
            "path": "/models/minimax-h3",
        },
    )
    monkeypatch.setattr(
        "minimax_studio.worker.probe.probe",
        lambda: {"cuda": True},
    )
    monkeypatch.setattr(
        "minimax_studio.worker.backends.h3_comfy.comfy_reachable",
        lambda: False,
    )
    with pytest.raises(RuntimeError, match="convrot"):
        resolve_h3_backend("auto")
    with pytest.raises(RuntimeError, match="convrot"):
        resolve_h3_backend("local")


def test_auto_backend_int8_with_comfy(monkeypatch, studio_home: Path) -> None:
    monkeypatch.setattr(
        "minimax_studio.worker.backends.h3.pack_status",
        lambda pack, root: {
            "ready": pack.id == "h3-fl2va",
            "path": "/models/minimax-h3",
        },
    )
    monkeypatch.setattr(
        "minimax_studio.worker.probe.probe",
        lambda: {"cuda": True},
    )
    monkeypatch.setattr(
        "minimax_studio.worker.backends.h3_comfy.comfy_reachable",
        lambda: True,
    )
    assert resolve_h3_backend("auto") == "comfy"
    assert INT8_NEEDS_COMFY.startswith("Comfy-Org")


# --- Comfy-side file visibility (object_info) -----------------------------

_ALL_FILES = {UNET_FL2VA, CLIP_NAME, VIDEO_VAE, AUDIO_VAE}


def _patch_objects(monkeypatch, listed):
    """Make _object_names answer with `listed` per class (None = unknown)."""

    def fake(cls, field):
        value = listed.get(cls, "all")
        return _ALL_FILES if value == "all" else value

    monkeypatch.setattr(h3_comfy, "_object_names", fake)


def test_comfy_resolve_file_exact_subfolder_and_unknown() -> None:
    import time as _t

    h3_comfy._OBJ_CACHE.clear()
    h3_comfy._OBJ_CACHE["LoraLoaderModelOnly.lora_name"] = (
        _t.monotonic(),
        {"h3-comfy/turbo.safetensors"},
    )
    assert (
        comfy_resolve_file("LoraLoaderModelOnly", "lora_name", "turbo.safetensors")
        == "h3-comfy/turbo.safetensors"
    )
    h3_comfy._OBJ_CACHE["LoraLoaderModelOnly.lora_name"] = (
        _t.monotonic(),
        {"turbo.safetensors"},
    )
    assert (
        comfy_resolve_file("LoraLoaderModelOnly", "lora_name", "turbo.safetensors")
        == "turbo.safetensors"
    )
    h3_comfy._OBJ_CACHE["LoraLoaderModelOnly.lora_name"] = (
        _t.monotonic(),
        {"other.safetensors"},
    )
    assert comfy_resolve_file("LoraLoaderModelOnly", "lora_name", "turbo.safetensors") is None
    h3_comfy._OBJ_CACHE["LoraLoaderModelOnly.lora_name"] = (_t.monotonic(), None)
    assert (
        comfy_resolve_file("LoraLoaderModelOnly", "lora_name", "turbo.safetensors")
        == "turbo.safetensors"
    )


def test_resolve_h3_backend_comfy_blocks_when_comfy_cannot_see_files(
    studio_home: Path, monkeypatch
) -> None:
    _touch_int8(runtime.config.models_root() / "h3-comfy")
    monkeypatch.setattr(h3_comfy, "comfy_reachable", lambda: True)
    _patch_objects(monkeypatch, {"CLIPLoader": {UNET_FL2VA, VIDEO_VAE, AUDIO_VAE}})
    try:
        resolve_h3_backend("comfy")
        raise AssertionError("expected a not-visible-files error")
    except RuntimeError as exc:
        assert "cannot see" in str(exc)
        assert CLIP_NAME in str(exc)


def test_resolve_h3_backend_comfy_ok_when_visible(
    studio_home: Path, monkeypatch
) -> None:
    _touch_int8(runtime.config.models_root() / "h3-comfy")
    monkeypatch.setattr(h3_comfy, "comfy_reachable", lambda: True)
    _patch_objects(monkeypatch, {})  # "all"
    assert resolve_h3_backend("comfy") == "comfy"


def test_generate_h3_comfy_fails_before_uploading_when_missing(
    studio_home: Path, monkeypatch
) -> None:
    from minimax_studio.worker.backends.h3_comfy import generate_h3_comfy

    _touch_int8(runtime.config.models_root() / "h3-comfy")
    monkeypatch.setattr(h3_comfy, "comfy_reachable", lambda: True)
    _patch_objects(monkeypatch, {"UNETLoader": {CLIP_NAME, VIDEO_VAE, AUDIO_VAE}})

    def no_upload(path):
        raise AssertionError("must not upload when model files are missing")

    monkeypatch.setattr(h3_comfy, "_upload_file", no_upload)
    request = JobRequest(kind="h3", backend="comfy", mode="t2va", prompt="a fox")
    try:
        generate_h3_comfy("jobfail0000", request)
        raise AssertionError("expected a not-visible-files error")
    except RuntimeError as exc:
        assert "cannot see" in str(exc)
        assert UNET_FL2VA in str(exc)


def test_music_comfy_missing_files_lists_all_three(monkeypatch) -> None:
    from minimax_studio.worker.backends.music_comfy import (
        CLIP_INT8,
        DIT_FP16,
        DIT_INT8,
        VAE_NAME,
        comfy_music_files_missing,
    )

    _patch_objects(monkeypatch, {"UNETLoader": set(), "CLIPLoader": set(), "VAELoader": set()})
    missing = comfy_music_files_missing()
    assert any(DIT_INT8 in item for item in missing)
    assert CLIP_INT8 in missing
    assert VAE_NAME in missing

    _patch_objects(monkeypatch, {"UNETLoader": {DIT_FP16}})
    missing = comfy_music_files_missing()
    assert not any(DIT_INT8 in item for item in missing)
    h3_comfy._OBJ_CACHE.clear()
