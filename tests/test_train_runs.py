"""PLAN-V2 S0: detached training runs against a stub trainer — no GPU, CI-safe.

The stub stands in for SimpleTuner: same invocation (`<bin> train env=<id>`),
same stdout, same checkpoints/ output. The real-metal check lives in PLAN-V2 S0.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from minimax_studio.worker import train_runs

TRAINER_STUB = """
import pathlib, sys, time
if "--version" in sys.argv:
    print("simpletuner 4.8.0")
    sys.exit(0)
mode = __import__("os").environ.get("STUB_TRAINER_MODE", "finish")
print("[simpler_tuner] step: 5 loss: 0.512 lr: 5e-05")
sys.stdout.flush()
if mode == "finish":
    pathlib.Path("checkpoints/step-10").mkdir(parents=True, exist_ok=True)
    pathlib.Path("checkpoints/step-10/lora-one.safetensors").write_bytes(b"x")
    print("steps: 10 loss: 0.100")
    sys.exit(0)
print("still training...")
sys.stdout.flush()
time.sleep(120)
"""


@pytest.fixture
def trainer_stub(tmp_path: Path, monkeypatch) -> Path:
    script = tmp_path / "trainer_stub.py"
    script.write_text(TRAINER_STUB, encoding="utf-8")
    monkeypatch.setenv(
        "MINIMAX_STUDIO_SIMPLETUNER_BIN", f'"{sys.executable}" "{script}"'
    )
    monkeypatch.setenv("MINIMAX_STUDIO_TRAIN_RUNS", str(tmp_path / "runs"))
    monkeypatch.setenv("MINIMAX_STUDIO_TRAIN_FORCE", "1")  # skip GPU gates here
    monkeypatch.setattr(
        "minimax_studio.worker.train_runs.train_preflight",
        lambda _preset: {"ok": True, "detail": "", "warnings": []},
    )
    return tmp_path


def _dataset(tmp_path: Path) -> Path:
    clips = tmp_path / "clips"
    clips.mkdir(exist_ok=True)
    (clips / "song.wav").write_bytes(b"RIFF")
    (clips / "song.txt").write_text("moody folk", encoding="utf-8")
    return clips


def _wait_status(run_id: str, wanted: str, timeout: float = 25.0) -> dict:
    deadline = time.time() + timeout
    state = {}
    while time.time() < deadline:
        _run_dir, state = train_runs.get_run(run_id)
        if state["status"] == wanted:
            return state
        time.sleep(0.15)
    pytest.fail(f"run never reached {wanted}; last state: {state}")


def test_start_run_sets_cuda_visible_devices(
    trainer_stub: Path, studio_home: Path, monkeypatch
) -> None:
    from minimax_studio.worker.runtime import runtime

    runtime.config.cuda_device = 1
    captured: dict[str, object] = {}
    real = train_runs.subprocess.Popen

    def wrapped(*args, **kwargs):
        captured["env"] = kwargs.get("env")
        return real(*args, **kwargs)

    monkeypatch.setattr(train_runs.subprocess, "Popen", wrapped)
    train_runs.start_run("gpu pin", _dataset(trainer_stub), "24g", steps=2)
    env = captured.get("env") or {}
    assert env.get("CUDA_VISIBLE_DEVICES") == "1"
    pythonpath = str(env.get("PYTHONPATH") or "")
    assert "st_startup" in pythonpath
    assert "expandable_segments:True" in str(env.get("PYTORCH_CUDA_ALLOC_CONF") or "")


def test_start_run_writes_run_dir_and_state(trainer_stub: Path) -> None:
    clips = _dataset(trainer_stub)
    state = train_runs.start_run("my song lora", clips, "24g", steps=42)
    run_dir = train_runs.runs_root() / state["id"]
    assert state["status"] == "running"
    assert (run_dir / "state.json").is_file()
    assert (run_dir / "config" / state["id"] / "config.json").is_file()
    assert (run_dir / "train.log").is_file()
    # Datasets are read-only to the trainer: the backend points AT them.
    import json as _json

    backends = _json.loads(
        (run_dir / "config" / state["id"] / "multidatabackend.json").read_text()
    )
    assert any(
        row.get("instance_data_dir") == str(clips.resolve()) for row in backends
    )
    _wait_status(state["id"], "completed")


def test_progress_parses_simpletuner_tqdm_from_metal() -> None:
    """S0 metal 2026-08-30: tqdm writes Steps: 100% | 200/200 and step_loss=."""
    from minimax_studio.worker.train_runs import LOSS_RE, STEP_RE, TQDM_STEP_RE

    line = (
        "Epoch 40/40, Steps: 100%|██████████| 200/200 "
        "[03:55<00:00,  1.18s/it, lr=5e-5, step_loss=1.05]"
    )
    assert STEP_RE.findall(line)[-1] == "100"  # the percent — why TQDM_STEP_RE exists
    assert TQDM_STEP_RE.findall(line)[-1] == ("200", "200")
    assert LOSS_RE.findall(line)[-1] == "1.05"


def test_run_completes_and_progress_parses_log(trainer_stub: Path) -> None:
    state = train_runs.start_run("finisher", _dataset(trainer_stub), "24g")
    final = _wait_status(state["id"], "completed")
    assert final["exit_code"] == 0
    progress = train_runs.progress(state["id"])
    assert progress["step"] == 10
    assert progress["total_steps"] == 1000
    assert progress["loss"] == pytest.approx(0.1)
    assert any("lora-one.safetensors" in c for c in progress["checkpoints"])
    tail = train_runs.log_tail(state["id"])
    assert any("step" in line for line in tail)


def test_install_adapter_lands_in_picker(trainer_stub: Path, studio_home: Path) -> None:
    from minimax_studio.worker.loras import list_loras
    from minimax_studio.worker.runtime import runtime

    state = train_runs.start_run("installable", _dataset(trainer_stub), "24g")
    _wait_status(state["id"], "completed")
    row = train_runs.install_adapter(state["id"])
    assert row["trained_run"] == state["id"]
    assert Path(row["path"]).is_file()
    assert row["path"].startswith(str(runtime.config.models_root() / "loras"))
    assert any(item["path"] == row["path"] for item in list_loras())
    assert Path(row["path"]).name.startswith("installable-")


def test_two_runs_do_not_install_over_the_same_lora_filename(
    trainer_stub: Path, studio_home: Path
) -> None:
    from minimax_studio.worker.loras import list_loras

    first = train_runs.start_run("alpha", _dataset(trainer_stub), "24g")
    _wait_status(first["id"], "completed")
    second = train_runs.start_run("beta", _dataset(trainer_stub), "24g")
    _wait_status(second["id"], "completed")
    a = train_runs.install_adapter(first["id"])
    b = train_runs.install_adapter(second["id"])
    assert Path(a["path"]).name != Path(b["path"]).name
    assert Path(a["path"]).is_file() and Path(b["path"]).is_file()
    assert Path(a["path"]).read_bytes() == Path(b["path"]).read_bytes()
    names = {Path(item["path"]).name for item in list_loras()}
    assert Path(a["path"]).name in names and Path(b["path"]).name in names


def test_cancel_stops_the_whole_group(trainer_stub: Path, monkeypatch) -> None:
    monkeypatch.setenv("STUB_TRAINER_MODE", "sleep")
    state = train_runs.start_run("cancellable", _dataset(trainer_stub), "24g")
    _wait_status(state["id"], "running", timeout=5)
    final = train_runs.cancel_run(state["id"])
    assert final["cancel_requested"] is True
    done = _wait_status(state["id"], "cancelled")
    assert done["exit_code"] not in (None,)


@pytest.mark.skipif(
    os.name == "nt",
    reason=(
        "Windows cancel is taskkill /F (exit 1). SIGTERM-exits-0 is how "
        "SimpleTuner dies on POSIX, which is the case this test pins."
    ),
)
def test_cancel_is_still_cancelled_when_the_trainer_exits_zero(
    trainer_stub: Path, monkeypatch
) -> None:
    """SimpleTuner often SIGTERM-exits 0. That is not a successful training run."""
    script = trainer_stub / "clean_cancel.py"
    script.write_text(
        "import signal, sys, time\n"
        "if '--version' in sys.argv:\n"
        "    print('simpletuner 4.8.0'); sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))\n"
        "print('ready', flush=True)\n"
        "time.sleep(120)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "MINIMAX_STUDIO_SIMPLETUNER_BIN",
        f'"{sys.executable}" "{script}"',
    )
    state = train_runs.start_run("clean-cancel", _dataset(trainer_stub), "24g")
    _wait_status(state["id"], "running", timeout=5)
    log = train_runs.runs_root() / state["id"] / "train.log"
    deadline = time.time() + 5
    while time.time() < deadline:
        if log.is_file() and b"ready" in log.read_bytes():
            break
        time.sleep(0.05)
    else:
        pytest.fail("trainer never printed ready")
    train_runs.cancel_run(state["id"])
    done = _wait_status(state["id"], "cancelled")
    assert done["exit_code"] == 0
    assert done["status"] == "cancelled"


def test_a_silent_live_run_is_marked_lost(trainer_stub: Path, monkeypatch) -> None:
    monkeypatch.setenv("STUB_TRAINER_MODE", "sleep")
    monkeypatch.setattr(train_runs, "LOST_HEARTBEAT_S", 0.05)
    state = train_runs.start_run("hung", _dataset(trainer_stub), "24g")
    _wait_status(state["id"], "running", timeout=5)
    run_dir = train_runs.runs_root() / state["id"]
    old = time.time() - 10
    payload = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    payload["started_at"] = old
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")
    os.utime(run_dir / "train.log", (old, old))
    lost = _wait_status(state["id"], "lost", timeout=5)
    assert lost["status"] == "lost"
    assert state["id"] in {row["id"] for row in train_runs.live_runs()}
    train_runs.cancel_run(state["id"])
    _wait_status(state["id"], "cancelled")


def test_reattach_after_worker_restart(trainer_stub: Path) -> None:
    run_dir = trainer_stub / "runs" / "20260101-000000-orphan"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        '{"id": "20260101-000000-orphan", "pid": 2147483646, '
        '"status": "running", "cancel_requested": false, "exit_code": null, '
        '"started_at": 1, "name": "orphan", "steps": 10}',
        encoding="utf-8",
    )
    rows = train_runs.list_runs()
    orphan = next(row for row in rows if row["id"] == "20260101-000000-orphan")
    assert orphan["status"] == "ended-unknown"  # honest, never "completed"


def test_start_run_refuses_bad_dataset_and_preset(trainer_stub: Path) -> None:
    with pytest.raises(RuntimeError, match="not ready"):
        train_runs.start_run("nope", trainer_stub / "missing", "24g")
    with pytest.raises(RuntimeError, match="preset"):
        train_runs.start_run("nope", _dataset(trainer_stub), "9001g")


def test_train_api_endpoints(trainer_stub: Path, studio_home: Path) -> None:
    from fastapi.testclient import TestClient

    from minimax_studio.worker.server import app

    client = TestClient(app)
    clips = _dataset(trainer_stub)

    started = client.post(
        "/train/runs",
        json={"name": "api run", "dataset_dir": str(clips), "preset": "24g"},
    )
    assert started.status_code == 200
    run_id = started.json()["id"]

    _wait_status(run_id, "completed")
    detail = client.get(f"/train/runs/{run_id}?tail=10")
    assert detail.status_code == 200
    body = detail.json()
    assert body["status"] == "completed"
    assert body["progress"]["step"] == 10
    assert body["log_tail"]

    listed = client.get("/train/runs").json()
    assert any(row["id"] == run_id for row in listed)

    installed = client.post(f"/train/runs/{run_id}/install")
    assert installed.status_code == 200
    assert installed.json()["trained_run"] == run_id

    missing = client.get("/train/runs/does-not-exist")
    assert missing.status_code == 404
    bad = client.post(
        "/train/runs",
        json={"name": "x", "dataset_dir": str(trainer_stub / "gone"), "preset": "24g"},
    )
    assert bad.status_code == 409


def test_run_rows_carry_the_folder_the_ui_opens(trainer_stub: Path) -> None:
    """"Open folder" is the only way to reach a detached run's logs and
    checkpoints once Studio was closed while it ran."""
    clips = _dataset(trainer_stub)
    state = train_runs.start_run("folder please", clips, "24g", steps=42)
    assert state["path"] == str(train_runs.runs_root() / state["id"])
    rows = train_runs.list_runs()
    assert rows and rows[0]["path"] == state["path"]
    train_runs.cancel_run(state["id"])


def test_live_run_warns_generate_preflight(
    trainer_stub: Path, studio_home: Path, monkeypatch
) -> None:
    """GPU etiquette cuts both ways. Training refuses to join a generation;
    and while a run holds the card — ours or one from a previous launch of the
    app — Generate has to say so before you press it."""
    monkeypatch.setenv("STUB_TRAINER_MODE", "sleep")
    monkeypatch.setenv("MINIMAX_STUDIO_STUB", "1")
    clips = _dataset(trainer_stub)
    state = train_runs.start_run("overnight", clips, "24g", steps=100)
    try:
        assert [row["id"] for row in train_runs.live_runs()] == [state["id"]]
        from minimax_studio.worker.preflight import preflight

        check = preflight("music", "stub")
        assert check["ok"] is True  # a warning, not a wall
        assert any(
            "overnight" in warning and "training run" in warning
            for warning in check["warnings"]
        ), check["warnings"]
    finally:
        # Wait for the death, don't just ask for it: a stub still sleeping 120 s
        # after the test ends is a stray process on the runner, and on Windows
        # that is how a test job turns into a hung one.
        train_runs.cancel_run(state["id"])
        _wait_status(state["id"], "cancelled")


def test_install_adapter_refuses_a_file_outside_the_run(
    trainer_stub: Path, tmp_path: Path
) -> None:
    state = train_runs.start_run("installable", _dataset(trainer_stub), "24g")
    _wait_status(state["id"], "completed")
    outsider = tmp_path / "not-this-run.safetensors"
    outsider.write_bytes(b"\x00")
    with pytest.raises(RuntimeError, match="not a checkpoint of this run"):
        train_runs.install_adapter(state["id"], str(outsider))


def test_run_ids_cannot_walk_out_of_the_root(trainer_stub: Path) -> None:
    with pytest.raises(RuntimeError, match="No training run"):
        train_runs.get_run("..")


def test_progress_checkpoint_paths_are_posix(trainer_stub: Path) -> None:
    state = train_runs.start_run("posix", _dataset(trainer_stub), "24g")
    _wait_status(state["id"], "completed")
    for path in train_runs.progress(state["id"])["checkpoints"]:
        assert "\\" not in path
        assert "/" in path or path.endswith(".safetensors")
