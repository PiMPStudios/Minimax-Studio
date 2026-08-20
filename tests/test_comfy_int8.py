from pathlib import Path

import pytest

from minimax_studio.worker.backends.h3 import INT8_NEEDS_COMFY, resolve_h3_backend
from minimax_studio.worker.backends.h3_comfy import UNET_REF2VA, build_h3_comfy_graph
from minimax_studio.worker.backends.music_comfy import DIT_INT8, build_music_comfy_graph
from minimax_studio.worker.catalog import PACKS
from minimax_studio.worker.model_paths import pack_status, parse_extra_model_paths


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
    assert "lora" not in graph


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
