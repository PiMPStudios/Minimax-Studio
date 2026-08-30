"""PLAN-V2 S4, the off-GPU half: H3 presets, configs, preflight, gating.

None of this touches a GPU or SimpleTuner. What it pins down is the part that
can be wrong quietly: which tiers exist, what config an H3 run writes, which
packs preflight demands, and the refusal that stops a Music preset ever being
pointed at a video dataset. The keys SimpleTuner has not been *seen* to accept
for H3 are listed in ``train_config.H3_UNVERIFIED_KEYS`` and announced by
preflight — the metal session is what retires that warning, not a green test run.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from minimax_studio.worker import datasets, train_config, train_runs
from minimax_studio.worker import probe as probe_module

TRAINER_STUB = """
import pathlib, sys, time
if "--version" in sys.argv:
    print("simpletuner 4.8.0"); sys.exit(0)
pathlib.Path("checkpoints/step-10").mkdir(parents=True, exist_ok=True)
pathlib.Path("checkpoints/step-10/lora-one.safetensors").write_bytes(b"x")
print("steps: 10 loss: 0.100"); sys.exit(0)
"""

MUSIC_SPEC: dict[str, Any] = {"kind": "music", "audio_files": 2}


@pytest.fixture
def h3_env(studio_home, tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_STUDIO_TRAIN_RUNS", str(tmp_path / "runs"))
    monkeypatch.setenv("MINIMAX_STUDIO_DATASETS", str(tmp_path / "datasets"))
    # The GPU gates are tested directly against train_preflight below; what
    # start_run tests is that the family refusal sits outside this switch.
    monkeypatch.setenv("MINIMAX_STUDIO_TRAIN_FORCE", "1")
    monkeypatch.setenv(
        "MINIMAX_STUDIO_SIMPLETUNER_BIN",
        f'"{sys.executable}" "{_trainer_stub(tmp_path)}"',
    )
    train_runs._PROCS.clear()
    yield tmp_path
    for _run_id, proc in list(train_runs._PROCS.items()):
        proc.kill()
    train_runs._PROCS.clear()


def _trainer_stub(tmp_path: Path) -> Path:
    script = tmp_path / "trainer_stub.py"
    script.write_text(TRAINER_STUB, encoding="utf-8")
    return script


@pytest.fixture
def gpu24(monkeypatch):
    """A 24 GB card with plenty of RAM and no other job on the bus."""
    monkeypatch.setattr(
        probe_module,
        "probe",
        lambda: {"cuda": True, "vram_gb": 24.0, "free_vram_gb": 24.0, "ram_gb": 96.0},
    )


def _h3_weights(ready: bool = True) -> Path:
    root = train_config._models_root() / "h3-diffusers"
    (root / "transformer").mkdir(parents=True, exist_ok=True)
    (root / "text_encoder").mkdir(parents=True, exist_ok=True)
    files = [
        root / "modular_model_index.json",
        root / "transformer" / "config.json",
        root / "text_encoder" / "config.json",
    ]
    for path in files:
        if ready:
            path.write_text("{}", encoding="utf-8")
        elif path.is_file():
            path.unlink()
    return root


def _video_folder(tmp_path: Path, stills: int = 1, clips: int = 0) -> Path:
    folder = tmp_path / f"shots-{stills}-{clips}"
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(stills):
        (folder / f"still{index}.png").write_bytes(b"\x00")
        (folder / f"still{index}.txt").write_text("a frame", encoding="utf-8")
    for index in range(clips):
        (folder / f"clip{index}.mp4").write_bytes(b"\x00")
        (folder / f"clip{index}.txt").write_text("a move", encoding="utf-8")
    return folder


def _music_folder(tmp_path: Path) -> Path:
    import wave

    folder = tmp_path / "songs"
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(2):
        with wave.open(str(folder / f"song{index}.wav"), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(8000)
            handle.writeframes(b"\x00\x00" * 8000 * 4)
        (folder / f"song{index}.txt").write_text("moody folk", encoding="utf-8")
    return folder


def _written_config(run_dir: Path, run_id: str) -> dict[str, Any]:
    return json.loads(
        (run_dir / "config" / run_id / "config.json").read_text(encoding="utf-8")
    )


# --- the tier table ----------------------------------------------------------


def test_the_h3_tiers_are_the_four_simpletuner_names() -> None:
    h3 = {name: row for name, row in train_config.PRESETS.items() if row.family == "h3"}
    assert sorted(h3) == ["h3-24g", "h3-32g", "h3-48g", "h3-80g"]
    assert [row.vram_floor_gb for _, row in sorted(h3.items())] == [24, 32, 48, 80]
    # Only the 24 GB tier pays for RamTorch, and only it says so in its title.
    assert [name for name, row in h3.items() if row.ram_torch] == ["h3-24g"]
    assert "RamTorch" in h3["h3-24g"].title
    assert h3["h3-24g"].flavour == "convrot-int8"
    assert h3["h3-80g"].flavour == "fl2va"
    assert {row.packs_any for row in h3.values()} == {
        ("h3-diffusers-fl2va", "h3-diffusers-ref2va")
    }


def test_music_presets_stay_music_only() -> None:
    music = {name: row for name, row in train_config.PRESETS.items() if row.family == "music"}
    assert sorted(music) == ["24g", "48g"]
    assert train_config.DEFAULT_PRESET in music


# --- what an H3 run writes ---------------------------------------------------


def test_an_h3_run_writes_the_video_config(h3_env, tmp_path) -> None:
    _h3_weights()
    run_dir = h3_env / "runs" / "r1"
    train_config.write_run_config(
        run_dir,
        "r1",
        _video_folder(tmp_path, stills=2),
        preset_name="h3-24g",
        steps=200,
        dataset_spec={
            "kind": "video",
            "has_stills": True,
            "has_clips": False,
            "h3_target_mode": "video",
        },
    )
    config = _written_config(run_dir, "r1")
    assert config["model_family"] == "minimaxh3"
    assert config["model_flavour"] == "convrot-int8"
    assert config["pretrained_model_name_or_path"].endswith("h3-diffusers")
    assert config["ramtorch"] is True
    assert config["ramtorch_text_encoder"] is True
    assert "ram_torch" not in config
    assert config["base_model_precision"] == "no_change"
    assert config["resolution"] == 480
    assert config["resolution_type"] == "pixel_area"
    assert config["flow_schedule_shift"] == 12.0
    assert config["minimax_h3_target_mode"] == "video"
    assert config["max_train_steps"] == 200
    assert config["distillation_method"] == "h3_drift"
    # Music-only keys must not leak into an H3 run — that is how a wrong model
    # turns a config review into a guessing game.
    for key in ("validation_lyrics", "validation_audio_duration"):
        assert key not in config

    backends = json.loads(
        (run_dir / "config" / "r1" / "multidatabackend.json").read_text()
    )
    assert backends[0]["dataset_type"] == "image"  # stills only
    assert "audio" not in backends[0]
    assert backends[0]["cache_dir_vae"] == str(run_dir / "cache" / "vae")


def test_clips_use_the_video_backend_and_carry_the_chosen_mode(h3_env, tmp_path) -> None:
    _h3_weights()
    train_config.write_run_config(
        h3_env / "runs" / "r2",
        "r2",
        _video_folder(tmp_path, stills=0, clips=3),
        preset_name="h3-48g",
        dataset_spec={
            "kind": "video",
            "has_stills": False,
            "has_clips": True,
            "h3_target_mode": "av",
        },
    )
    config = _written_config(h3_env / "runs" / "r2", "r2")
    backends = json.loads(
        (h3_env / "runs" / "r2" / "config" / "r2" / "multidatabackend.json").read_text()
    )
    assert config["minimax_h3_target_mode"] == "av"
    assert "ramtorch" not in config  # only the 24 GB tier pays for it
    assert config["model_flavour"] == "convrot-int8"
    assert config["resolution"] == 768
    assert backends[0]["dataset_type"] == "video"


def test_a_music_dataset_never_meets_an_h3_preset(h3_env, tmp_path) -> None:
    with pytest.raises(RuntimeError, match="trains MiniMax H3"):
        train_config.write_run_config(
            h3_env / "runs" / "r3",
            "r3",
            _music_folder(tmp_path),
            preset_name="h3-24g",
            dataset_spec=MUSIC_SPEC,
        )
    with pytest.raises(RuntimeError, match="trains MiniMax Music 3"):
        train_config.write_run_config(
            h3_env / "runs" / "r4",
            "r4",
            _video_folder(tmp_path),
            preset_name="24g",
            dataset_spec={"kind": "video", "has_stills": True},
        )


# --- preflight ---------------------------------------------------------------


def test_preflight_names_the_h3_weights_it_wants(h3_env, gpu24) -> None:
    _h3_weights(ready=False)
    check = train_config.train_preflight("h3-24g")
    assert check["ok"] is False
    assert check["family"] == "h3"
    problem = " ".join(check["problems"])
    assert "diffusers layout" in problem
    assert "The Comfy packs will not do" in problem

    _h3_weights(ready=True)
    check = train_config.train_preflight("h3-24g")
    assert not any("diffusers layout" in p for p in check["problems"]), check["problems"]
    # Music's encoder pack is not an H3 requirement.
    assert not any("Training Encoder" in p for p in check["problems"])


def test_preflight_says_which_h3_keys_nobody_has_seen_yet(h3_env, gpu24) -> None:
    _h3_weights()
    check = train_config.train_preflight("h3-24g")
    warning = " ".join(check["warnings"])
    assert "documentation, not its output" in warning
    assert "minimax_h3_target_mode" in warning
    assert "RamTorch keeps layers in system RAM" in warning
    # Music preflight carries no such warning: it has run.
    music = train_config.train_preflight("24g")
    assert not any("documentation" in text for text in music["warnings"])


def test_preflight_checks_the_picked_dataset_against_the_picked_preset(
    h3_env, gpu24, tmp_path
) -> None:
    _h3_weights()
    shots = _video_folder(tmp_path, stills=2, clips=1)
    check = train_config.train_preflight("24g", shots)
    assert any("trains MiniMax Music 3" in p for p in check["problems"])

    check = train_config.train_preflight("h3-32g", shots)
    assert not any("trains MiniMax" in p for p in check["problems"])
    assert any("mixes 2 still(s) with 1 clip(s)" in w for w in check["warnings"])


def test_the_h3_cache_floor_is_larger_and_says_so(h3_env, gpu24, tmp_path, monkeypatch) -> None:
    _h3_weights()
    monkeypatch.setattr(
        train_config.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=0, used=0, free=25 * 1024**3),
    )
    music = train_config.train_preflight("24g")
    assert not any("GB free on the training volume" in p for p in music["problems"])

    h3 = train_config.train_preflight("h3-32g")
    problem = " ".join(h3["problems"])
    assert "Only 25 GB free" in problem and "about 40 GB" in problem


# --- starting and re-running -------------------------------------------------


def test_start_run_refuses_the_wrong_pair_before_writing_anything(
    h3_env, tmp_path
) -> None:
    shots = _video_folder(tmp_path, stills=1)
    with pytest.raises(RuntimeError, match="holds a video dataset"):
        train_runs.start_run("wrong", shots, "24g", steps=10)
    songs = _music_folder(tmp_path)
    with pytest.raises(RuntimeError, match="holds a music dataset"):
        train_runs.start_run("wrong", songs, "h3-24g", steps=10)
    # Nothing was written: the refusal happens before there is a run to clean up.
    assert not (h3_env / "runs").exists()


def test_an_h3_run_is_launched_with_its_own_kind_recorded(h3_env, tmp_path) -> None:
    """The spec travels in state.json: the resumer must write the same kind of
    config even after the dataset folder has moved or gone."""
    _h3_weights()
    shots = _video_folder(tmp_path, stills=1, clips=1)
    state = train_runs.start_run("h3 lora", shots, "h3-32g", steps=12)
    run_dir = train_runs.runs_root() / state["id"]
    assert state["dataset_kind"] == "video" and state["family"] == "h3"
    assert state["dataset_spec"]["has_clips"] is True
    assert _written_config(run_dir, state["id"])["model_family"] == "minimaxh3"


def test_resume_reuses_the_recorded_spec(h3_env, tmp_path) -> None:
    _h3_weights()
    shots = _video_folder(tmp_path, stills=2)
    state = train_runs.start_run("h3 lora", shots, "h3-24g", steps=12)
    run_dir = train_runs.runs_root() / state["id"]
    checkpoint = run_dir / "checkpoints" / "step-10" / "lora-one.safetensors"
    deadline = time.time() + 25
    while not checkpoint.is_file() and time.time() < deadline:
        time.sleep(0.15)
    assert checkpoint.is_file(), "the stub trainer never wrote a checkpoint"

    resumed = train_runs.resume_run(state["id"])
    assert resumed["resume_count"] == 1
    config = _written_config(run_dir, state["id"])
    assert config["model_family"] == "minimaxh3"
    assert config["resume_from_checkpoint"] == "latest"


# --- dataset_spec ------------------------------------------------------------


def test_dataset_spec_reads_a_manifest_when_there_is_one(h3_env) -> None:
    manifest = datasets.create_dataset("stills", "video")
    folder, _ = datasets.get_dataset(manifest["id"])
    (folder / "a.png").write_bytes(b"\x00")
    datasets.set_h3_target_mode(manifest["id"], "video")
    spec = datasets.dataset_spec(folder)
    assert spec["kind"] == "video" and spec["has_stills"] and not spec["has_clips"]
    assert spec["h3_target_mode"] == "video"


def test_a_loose_folder_has_its_kind_inferred(h3_env, tmp_path) -> None:
    shots = _video_folder(tmp_path, stills=1, clips=2)
    assert datasets.dataset_spec(shots)["kind"] == "video"
    assert datasets.dataset_spec(_music_folder(tmp_path))["kind"] == "music"
    # An empty folder is music-shaped by default and stays untrainable anyway.
    empty = tmp_path / "empty"
    empty.mkdir()
    assert datasets.dataset_spec(empty)["kind"] == "music"


def test_the_80g_tier_writes_official_fl2va_flavour(h3_env, tmp_path) -> None:
    _h3_weights()
    train_config.write_run_config(
        h3_env / "runs" / "r80",
        "r80",
        _video_folder(tmp_path, stills=1),
        preset_name="h3-80g",
        dataset_spec={"kind": "video", "has_stills": True, "has_clips": False},
    )
    config = _written_config(h3_env / "runs" / "r80", "r80")
    assert config["model_flavour"] == "fl2va"
    assert config["base_model_precision"] == "no_change"
    assert "ramtorch" not in config


def test_music_configs_keep_their_old_shape(h3_env, tmp_path) -> None:
    """The refactor that added H3 must not disturb the config that already
    trains — this is the golden test for the Music side."""
    train_config.write_run_config(
        h3_env / "runs" / "music",
        "music",
        _music_folder(tmp_path),
        preset_name="24g",
        steps=42,
    )
    config = _written_config(h3_env / "runs" / "music", "music")
    backends = json.loads(
        (h3_env / "runs" / "music" / "config" / "music" / "multidatabackend.json").read_text()
    )
    assert config["model_family"] == "minimaxmusic"
    assert config["resolution"] == 512
    assert config["validation_audio_duration"] == 15
    assert backends[0]["dataset_type"] == "audio"
    assert backends[0]["audio"]["duration_interval"] == 3.0
    assert backends[0]["audio"]["max_duration_seconds"] == 300.0


# --- routes ------------------------------------------------------------------


def test_preflight_route_takes_the_dataset(h3_env, gpu24, tmp_path) -> None:
    from fastapi.testclient import TestClient

    from minimax_studio.worker.server import app

    _h3_weights()
    client = TestClient(app)
    shots = _video_folder(tmp_path, stills=1)
    payload = client.get(
        f"/train/preflight?preset=h3-24g&dataset_dir={shots}"
    ).json()
    assert payload["family"] == "h3"
    assert payload["dataset_kind"] == "video"
    assert payload["presets"]["h3-80g"]["vram_floor_gb"] == 80


def test_target_mode_route_refuses_with_a_reason(h3_env) -> None:
    from fastapi.testclient import TestClient

    from minimax_studio.worker.server import app

    client = TestClient(app)
    manifest = datasets.create_dataset("silent-set", "video")
    folder, _ = datasets.get_dataset(manifest["id"])
    (folder / "a.mp4").write_bytes(b"\x00")
    (folder / "a.txt").write_text("a shot", encoding="utf-8")

    assert client.post(
        f"/datasets/{manifest['id']}/target-mode", json={"mode": "video"}
    ).status_code == 200
    refusal = client.post(
        f"/datasets/{manifest['id']}/target-mode", json={"mode": "av"}
    )
    assert refusal.status_code == 409
    assert "could not be measured" in refusal.json()["detail"]
