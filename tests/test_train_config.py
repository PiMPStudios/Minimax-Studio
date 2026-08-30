"""PLAN-V2 S0: training config generation and preflight — all off-GPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from minimax_studio.worker.train_config import (
    PRESETS,
    SIMPLETUNER_PIN,
    simpletuner_version,
    train_preflight,
    validate_music_dataset_dir,
    write_run_config,
)

STUB = """
import sys
if "--version" in sys.argv:
    print("simpletuner {version}")
    sys.exit(0)
print("train invoked with", *sys.argv[1:])
"""


def _stub_bin(tmp_path: Path, monkeypatch, version: str = SIMPLETUNER_PIN) -> None:
    script = tmp_path / "simpletuner_stub.py"
    script.write_text(STUB.format(version=version), encoding="utf-8")
    monkeypatch.setenv(
        "MINIMAX_STUDIO_SIMPLETUNER_BIN",
        f'"{sys.executable}" "{script}"',
    )


def _dataset(tmp_path: Path) -> Path:
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "song.wav").write_bytes(b"RIFF")
    (clips / "song.txt").write_text("moody folk", encoding="utf-8")
    return clips


def _ready_packs(studio_home: Path) -> None:
    from minimax_studio.worker.catalog import PACKS
    from minimax_studio.worker.runtime import runtime

    for pack_id in ("music3-cuda", "music3-train-encoder"):
        root = runtime.config.models_root() / PACKS[pack_id].local_dir
        for marker in PACKS[pack_id].marker_files:
            path = root / marker
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")


def test_validate_dataset_missing_folder(tmp_path: Path) -> None:
    errors = validate_music_dataset_dir(tmp_path / "nope")
    assert errors and "not found" in errors[0]


def test_validate_dataset_needs_audio_and_captions(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert validate_music_dataset_dir(empty)

    clips = _dataset(tmp_path)
    assert validate_music_dataset_dir(clips) == []

    (clips / "second.wav").write_bytes(b"RIFF")
    errors = validate_music_dataset_dir(clips)
    assert any("second.wav" in e and "second.txt" in e for e in errors)


def test_cheap_music_check_sees_mp3(tmp_path: Path) -> None:
    folder = tmp_path / "mp3s"
    folder.mkdir()
    (folder / "song.mp3").write_bytes(b"ID3")
    (folder / "song.txt").write_text("caption", encoding="utf-8")
    assert validate_music_dataset_dir(folder) == []


def test_write_run_config_contract(studio_home: Path, tmp_path: Path) -> None:
    clips = _dataset(tmp_path)
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    paths = write_run_config(run_dir, "run-1", clips, "24g", steps=250)
    config = json.loads(Path(paths["config"]).read_text())
    backends = json.loads(Path(paths["multidatabackend"]).read_text())

    assert config["model_family"] == "minimaxmusic"
    assert config["model_flavour"] == "music3"
    assert config["model_type"] == "lora"
    # LoRAs must load in our picker / Comfy path without conversion.
    assert config["lora_format"] == "comfyui"
    assert config["max_train_steps"] == 250
    assert config["lora_rank"] == PRESETS["24g"].lora_rank
    assert config["base_model_precision"] == "int8-quanto"
    assert config["pretrained_model_name_or_path"].endswith("music3-cuda")
    assert config["pretrained_vae_model_name_or_path"].endswith(
        "music3-train-encoder"
    )
    # Everything SimpleTuner writes stays inside the run dir.
    assert config["output_dir"] == str(run_dir / "checkpoints")
    assert backends[0]["cache_dir_vae"].startswith(str(run_dir))
    assert backends[1]["cache_dir"].startswith(str(run_dir))
    assert backends[0]["instance_data_dir"] == str(clips.resolve())
    assert backends[0]["caption_strategy"] == "textfile"
    assert backends[0]["audio"]["lyrics_filename_format"] == "{filename}.lyrics"
    assert backends[0]["audio"]["max_duration_seconds"] == 300.0


def _normalised(path: str, run_dir: Path):
    """Parse-then-normalise: raw JSON escapes backslashes on Windows, so text
    surgery on the file is the wrong comparison."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    def walk(value):
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, str):
            return value.replace(str(run_dir), "RUN")
        return value

    return walk(data)


def test_write_run_config_is_deterministic(studio_home: Path, tmp_path: Path) -> None:
    clips = _dataset(tmp_path)
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    pa = write_run_config(first, "same", clips, "24g")
    pb = write_run_config(second, "same", clips, "24g")
    for key in pa:
        # run dir differs by design; the config *content* must not drift.
        assert _normalised(pa[key], first) == _normalised(pb[key], second)


def test_write_run_config_rejects_unknown_preset(studio_home: Path, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="preset"):
        write_run_config(tmp_path, "r", _dataset(tmp_path), "8-hundread-g")


def test_simpletuner_version_parses_pin(tmp_path: Path, monkeypatch) -> None:
    _stub_bin(tmp_path, monkeypatch, version="9.9.9")
    assert simpletuner_version() == "9.9.9"
    monkeypatch.delenv("MINIMAX_STUDIO_SIMPLETUNER_BIN")
    monkeypatch.setattr("shutil.which", lambda _name: None)
    assert simpletuner_version() is None


def _all_green(studio_home: Path, tmp_path: Path, monkeypatch) -> None:
    _stub_bin(tmp_path, monkeypatch)
    _ready_packs(studio_home)
    monkeypatch.setattr(
        "minimax_studio.worker.probe.probe",
        lambda: {
            "cuda": True,
            "free_vram_gb": 30.0,
            "gpus": [{"name": "RTX", "vram_gb": 32, "free_vram_gb": 30.0}],
        },
    )


def test_train_preflight_green_path(studio_home: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MINIMAX_STUDIO_TRAIN_RUNS", str(tmp_path / "runs"))
    _all_green(studio_home, tmp_path, monkeypatch)
    result = train_preflight("24g")
    assert result["ok"] is True, result["detail"]
    assert not result["problems"]
    assert "Ready to train" in result["detail"]


def test_train_preflight_names_every_problem(
    studio_home: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MINIMAX_STUDIO_TRAIN_RUNS", str(tmp_path / "runs"))
    _stub_bin(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "minimax_studio.worker.probe.probe",
        lambda: {
            "cuda": True,
            "free_vram_gb": 18.4,
            "gpus": [{"name": "RTX", "vram_gb": 24, "free_vram_gb": 18.4}],
        },
    )
    result = train_preflight("24g")
    assert result["ok"] is False
    joined = " ".join(result["problems"])
    # Packs missing, VRAM short — each named with its number, no mystery OOM.
    assert "MiniMax-Music3 (CUDA / diffusers)" in joined
    assert "Training Encoder" in joined
    assert "24 GB" in joined and "18.4" in joined


def test_train_preflight_blocks_while_generating(
    studio_home: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MINIMAX_STUDIO_TRAIN_RUNS", str(tmp_path / "runs"))
    _all_green(studio_home, tmp_path, monkeypatch)
    from minimax_studio.worker.runtime import runtime

    runtime.jobs["job-1"] = {"id": "job-1", "status": "running"}
    result = train_preflight("24g")
    assert result["ok"] is False
    assert any("share the GPU" in problem for problem in result["problems"])


def test_train_preflight_warns_on_unpinned_simpletuner(
    studio_home: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MINIMAX_STUDIO_TRAIN_RUNS", str(tmp_path / "runs"))
    _all_green(studio_home, tmp_path, monkeypatch)
    _stub_bin(tmp_path, monkeypatch, version="4.9.0")
    result = train_preflight("24g")
    assert result["ok"] is True
    assert any("4.9.0" in warning for warning in result["warnings"])


def test_train_preflight_cuda_only(studio_home: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MINIMAX_STUDIO_TRAIN_RUNS", str(tmp_path / "runs"))
    _stub_bin(tmp_path, monkeypatch)
    _ready_packs(studio_home)
    monkeypatch.setattr(
        "minimax_studio.worker.probe.probe",
        lambda: {"cuda": False, "apple_silicon": True, "gpus": []},
    )
    result = train_preflight("24g")
    assert result["ok"] is False
    assert any("CUDA-only" in problem for problem in result["problems"])
