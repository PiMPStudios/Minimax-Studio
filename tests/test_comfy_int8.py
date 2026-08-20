from pathlib import Path

import pytest

from minimax_studio.worker.backends.h3 import INT8_NEEDS_COMFY, resolve_h3_backend
from minimax_studio.worker.backends.h3_comfy import build_h3_comfy_graph
from minimax_studio.worker.catalog import PACKS
from minimax_studio.worker.model_paths import pack_status


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


def test_auto_backend_prefers_official(monkeypatch, studio_home: Path) -> None:
    monkeypatch.setattr(
        "minimax_studio.worker.backends.h3.pack_status",
        lambda pack, root: {"ready": pack.id == "h3-diffusers-fl2va", "path": "x"},
    )
    monkeypatch.setattr(
        "minimax_studio.worker.probe.probe",
        lambda: {"cuda": True},
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
