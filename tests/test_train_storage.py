"""PLAN-V2 S5: long-run hardening — retention, caches, resume, export. No GPU.

A run that goes well writes tens of gigabytes and SimpleTuner has no reason to
tidy up after itself, so the app does. These tests are the guarantee that the
tidying never eats the wrong thing: an installed checkpoint is never pruned,
nothing is deleted while a trainer holds the files open, and an import never
quietly mixes two runs' weights.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from minimax_studio.worker import train_config, train_runs

# Stands in for `simpletuner train env=<id>`: survives, ignores SIGTERM-ish
# shutdown only long enough for the test to look at it, then exits cleanly.
FAKE_TRAINER = (
    "import signal,sys,time;"
    "signal.signal(signal.SIGTERM,lambda *a: sys.exit(0));"
    "print('training');sys.stdout.flush();time.sleep(30)"
)


@pytest.fixture
def runs_home(studio_home, tmp_path, monkeypatch):
    root = tmp_path / "runs"
    monkeypatch.setenv("MINIMAX_STUDIO_TRAIN_RUNS", str(root))
    monkeypatch.setattr(
        train_runs,
        "simpletuner_command_prefix",
        lambda: [sys.executable, "-c", FAKE_TRAINER],
    )
    train_runs._PROCS.clear()
    train_runs._invalidate_storage_cache()
    yield root
    for run_id, proc in list(train_runs._PROCS.items()):
        try:
            os.kill(proc.pid, signal.SIGTERM)
        except OSError:
            pass
    train_runs._PROCS.clear()
    train_runs._invalidate_storage_cache()


def _make_run(
    run_id: str = "20260829-summer",
    name: str = "Summer",
    steps: int = 800,
    status: str = "finished",
) -> dict:
    """A run folder as a finished run would leave it: config, state, log."""
    run_dir = train_runs.runs_root() / run_id
    dataset = run_dir.parent.parent / f"ds-{run_id}"
    (dataset / "clip.wav").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "clip.wav").write_bytes(b"RIFF")
    train_config.write_run_config(
        run_dir, run_id, dataset, preset_name="24g", steps=steps, rank=32
    )
    (run_dir / "train.log").write_text("steps: 10 loss: 0.4\n", encoding="utf-8")
    state = {
        "id": run_id,
        "name": name,
        "dataset_dir": str(dataset),
        "preset": "24g",
        "steps": steps,
        "rank": 32,
        "cmd": ["simpletuner", "train", f"env={run_id}"],
        "pid": 0,
        "started_at": time.time() - 7200,
        "status": status,
        "cancel_requested": False,
        "exit_code": 0 if status == "finished" else None,
        "finished_at": time.time() - 600 if status == "finished" else None,
    }
    train_runs._write_state(run_dir, state)
    return {**state, "path": str(run_dir)}


def _running(run) -> dict:
    """Pretend this run's trainer is live (pid = this process): the one state in
    which deleting anything must be refused."""
    state = train_runs._read_state(Path(run["path"]))
    state.update(status="running", pid=os.getpid(), exit_code=None)
    train_runs._write_state(Path(run["path"]), state)
    return {**state, "path": run["path"]}


def _checkpoint(run, step: int, *, kb: int = 8, age_s: float = 0.0) -> Path:
    path = Path(run["path"]) / "checkpoints" / f"step-{step}" / f"lora-{step}.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * (kb * 1024))
    stamp = time.time() - age_s
    os.utime(path, (stamp, stamp))
    return path


def _install(run, path: Path) -> Path:
    """Install like the Adapters page does, and return the **copy** in the LoRA
    folder — proving later on that retention never reaches that far."""
    row = train_runs.install_adapter(run["id"], str(path))
    return Path(row["path"])


def _dir_bytes(folder: Path) -> int:
    return sum(item.stat().st_size for item in folder.rglob("*") if item.is_file())


def _cache(run, *, kb: int = 16) -> Path:
    cache = Path(run["path"]) / "cache"
    (cache / "vae").mkdir(parents=True, exist_ok=True)
    (cache / "text").mkdir(parents=True, exist_ok=True)
    (cache / "vae" / "0000.pt").write_bytes(b"\0" * (kb * 1024))
    return cache


# --- what a run is made of ---------------------------------------------------


def test_storage_names_caches_and_checkpoints_separately(runs_home):
    run = _make_run()
    _cache(run, kb=16)
    _checkpoint(run, 100, kb=8)
    report = train_runs.storage(run["id"])
    assert report["cache_bytes"] == 16 * 1024
    assert report["checkpoint_bytes"] == 8 * 1024
    assert report["bytes"] == 24 * 1024
    assert report["free_gb"] > 0  # a real filesystem number, not a placeholder
    assert len(report["checkpoints"]) == 1


def test_checkpoint_rows_are_newest_first_and_say_which_were_installed(runs_home):
    run = _make_run()
    _checkpoint(run, 100, age_s=500)
    _install(run, _checkpoint(run, 200, age_s=100))
    _checkpoint(run, 300, age_s=10)
    rows = train_runs.checkpoint_rows(run["id"])
    assert [row["path"] for row in rows] == [
        "checkpoints/step-300/lora-300.safetensors",
        "checkpoints/step-200/lora-200.safetensors",
        "checkpoints/step-100/lora-100.safetensors",
    ]
    # "installed" is the only best-checkpoint signal this app really has.
    assert [row["installed"] for row in rows] == [False, True, False]


def test_storage_report_totals_every_run(runs_home):
    run_a = _make_run("2026-run-a", name="A")
    _install(run_a, _checkpoint(run_a, 50, kb=2, age_s=5))
    run_b = _make_run("2026-run-b", name="B")
    _cache(run_b, kb=4)
    report = train_runs.storage_report()
    assert [row["id"] for row in report["runs"]] == ["2026-run-b", "2026-run-a"]
    assert {row["name"] for row in report["runs"]} == {"A", "B"}
    assert report["total_bytes"] == 6 * 1024
    assert report["runs_root"] == str(runs_home)


def test_unknown_run_says_so_by_name(runs_home):
    with pytest.raises(RuntimeError, match="No training run 'ghost'"):
        train_runs.storage("ghost")


# --- retention ---------------------------------------------------------------


def test_prune_keeps_the_newest_plus_every_installed_checkpoint(runs_home):
    run = _make_run()
    oldest = _checkpoint(run, 100, kb=6, age_s=500)
    best = _checkpoint(run, 200, kb=6, age_s=400)
    _install(run, best)
    middle = _checkpoint(run, 300, kb=6, age_s=300)
    newest = _checkpoint(run, 400, kb=6, age_s=200)
    result = train_runs.prune_checkpoints(run["id"], keep=1)
    assert newest.is_file() and best.is_file()
    assert not oldest.is_file() and not middle.is_file()
    assert result["freed_bytes"] == 12 * 1024
    assert sorted(result["removed"]) == ["checkpoints/step-100", "checkpoints/step-300"]
    assert len(result["kept"]) == 2


def test_prune_takes_the_optimizer_state_with_the_checkpoint(runs_home):
    """SimpleTuner writes accelerator state next to the weights. Deleting only
    the .safetensors would free a tenth of what the dialog promised."""
    run = _make_run()
    old = _checkpoint(run, 100, kb=4, age_s=300)
    state = old.parent / "prmt_state.bin"  # the fat one
    state.write_bytes(b"\0" * (40 * 1024))
    _checkpoint(run, 200, kb=4, age_s=5)
    result = train_runs.prune_checkpoints(run["id"], keep=1)
    assert result["freed_bytes"] == 44 * 1024
    assert not old.parent.exists()
    assert result["removed"] == [f"checkpoints/{old.parent.name}"]


def test_a_step_folder_shared_by_two_checkpoints_survives_for_its_kept_one(runs_home):
    run = _make_run()
    folder = Path(run["path"]) / "checkpoints" / "step-both"
    folder.mkdir(parents=True)
    doomed = folder / "lora-a.safetensors"
    doomed.write_bytes(b"\0" * (4 * 1024))
    os.utime(doomed, (time.time() - 400, time.time() - 400))
    kept = folder / "lora-b.safetensors"
    kept.write_bytes(b"\0" * (4 * 1024))
    _install(run, kept)
    stamp = time.time() - 100
    os.utime(kept, (stamp, stamp))
    _checkpoint(run, 900, kb=2, age_s=1)  # the newest, always kept

    result = train_runs.prune_checkpoints(run["id"], keep=1)
    assert kept.is_file() and folder.is_dir()
    assert not doomed.is_file()
    assert result["freed_bytes"] == 4 * 1024  # the file, not the folder's contents
    assert result["removed"] == ["checkpoints/step-both/lora-a.safetensors"]


def test_prune_never_empties_the_run_even_when_asked_to_keep_zero(runs_home):
    run = _make_run()
    _checkpoint(run, 100, age_s=200)
    newest = _checkpoint(run, 200, age_s=5)
    train_runs.prune_checkpoints(run["id"], keep=0)
    assert newest.is_file()


def test_prune_sweeps_the_step_folders_it_empties(runs_home):
    run = _make_run()
    gone = _checkpoint(run, 100, age_s=200)
    _checkpoint(run, 200, age_s=5)
    train_runs.prune_checkpoints(run["id"], keep=1)
    assert not gone.parent.exists()
    assert (Path(run["path"]) / "checkpoints" / "step-200").is_dir()


def test_prune_on_a_run_that_never_saved_a_checkpoint_is_a_no_op(runs_home):
    run = _make_run()
    assert train_runs.prune_checkpoints(run["id"], keep=3) == {
        "kept": [],
        "removed": [],
        "freed_bytes": 0,
        "dry_run": False,
    }


def test_a_dry_run_names_the_number_and_touches_nothing(runs_home):
    """The dialog asks “how much?” before it asks “are you sure?”, so the plan
    has to be the same arithmetic the real prune performs — and delete nothing."""
    run = _make_run()
    old = _checkpoint(run, 100, kb=4, age_s=300)
    (old.parent / "prmt_state.bin").write_bytes(b"\0" * (40 * 1024))
    _checkpoint(run, 200, kb=4, age_s=200)
    newest = _checkpoint(run, 300, kb=4, age_s=5)

    plan = train_runs.prune_checkpoints(run["id"], keep=1, dry_run=True)
    assert plan["dry_run"] is True
    assert plan["freed_bytes"] == 48 * 1024  # weights *plus* the optimiser state
    assert plan["removed"] == [
        "checkpoints/step-200",
        f"checkpoints/{old.parent.name}",
    ]  # newest first, because that is the order the run wrote them
    assert old.is_file() and newest.is_file()

    done = train_runs.prune_checkpoints(run["id"], keep=1)
    assert done["freed_bytes"] == plan["freed_bytes"], "the promise was the number"
    assert done["removed"] == plan["removed"]


def test_nothing_is_deleted_while_the_trainer_is_live(runs_home):
    """The Windows-proof rule: a live process holds these files open, so
    "cleanup" there does not free disk, it leaves a half-deleted run."""
    run = _running(_make_run())
    oldest = _checkpoint(run, 100, age_s=200)
    _checkpoint(run, 200, age_s=5)
    _cache(run)
    with pytest.raises(RuntimeError, match="still training"):
        train_runs.prune_checkpoints(run["id"], keep=1)
    with pytest.raises(RuntimeError, match="still training"):
        train_runs.clear_cache(run["id"])
    with pytest.raises(RuntimeError, match="still training"):
        train_runs.delete_run(run["id"])
    with pytest.raises(RuntimeError, match="still training"):
        train_runs.resume_run(run["id"])
    assert oldest.is_file() and (Path(run["path"]) / "cache").is_dir()


def test_clear_cache_frees_the_cache_and_says_by_how_much(runs_home):
    run = _make_run()
    _cache(run, kb=16)
    _checkpoint(run, 100)
    result = train_runs.clear_cache(run["id"])
    assert result["cleared"] is True and result["freed_bytes"] == 16 * 1024
    assert not (Path(run["path"]) / "cache").exists()
    assert (Path(run["path"]) / "checkpoints").is_dir()  # weights are not cache
    # Clearing twice is a no-op with a number, not an error.
    assert train_runs.clear_cache(run["id"])["freed_bytes"] == 0


def test_delete_run_removes_the_folder_and_leaves_installed_adapters(runs_home):
    run = _make_run()
    best = _checkpoint(run, 200, kb=4)
    installed = _install(run, best)
    _cache(run, kb=4)
    before = _dir_bytes(Path(run["path"]))
    result = train_runs.delete_run(run["id"])
    assert not Path(run["path"]).exists()
    assert result["freed_bytes"] == before  # the whole folder, weighed first
    assert before >= 8 * 1024
    # Installed adapters are copies — the picker still has it after the run is gone.
    assert installed.is_file()
    assert train_runs.list_runs() == []


# --- resume ------------------------------------------------------------------


def test_resume_continues_the_same_run_in_the_same_folder(runs_home):
    run = _make_run(steps=800)
    _cache(run)
    best = _checkpoint(run, 400)
    _install(run, best)
    resumed = train_runs.resume_run(run["id"])
    assert resumed["path"] == run["path"]
    assert resumed["status"] == "running" and resumed["pid"] > 0
    assert resumed["resume_count"] == 1
    assert resumed["resumed_from"] == "latest"
    config = json.loads(
        (Path(run["path"]) / "config" / run["id"] / "config.json").read_text()
    )
    assert config["resume_from_checkpoint"] == "latest"
    assert config["max_train_steps"] == 800  # resuming is not restarting smaller
    assert (Path(run["path"]) / "cache").is_dir()  # the cache is why it's fast


def test_resume_takes_a_chosen_checkpoint_not_just_the_newest(runs_home):
    run = _make_run()
    _checkpoint(run, 200, age_s=300)
    older = _checkpoint(run, 100, age_s=400)
    _install(run, older)
    resumed = train_runs.resume_run(run["id"], f"checkpoints/step-100/{older.name}")
    assert resumed["resumed_from"] == str(older.parent)


def test_resume_without_a_checkpoint_names_the_run_instead_of_asserting(runs_home):
    run = _make_run()
    with pytest.raises(RuntimeError, match="no checkpoint"):
        train_runs.resume_run(run["id"])


def test_resume_refuses_a_checkpoint_belonging_to_another_run(runs_home):
    mine = _make_run("2026-mine")
    other = _make_run("2026-other")
    foreign = _checkpoint(other, 100)
    with pytest.raises(RuntimeError, match="not a checkpoint of this run"):
        train_runs.resume_run(mine["id"], str(foreign))
    with pytest.raises(RuntimeError, match="not a checkpoint of this run"):
        train_runs.resume_run(mine["id"], "../../etc/passwd")
    with pytest.raises(RuntimeError, match="not a checkpoint of this run"):
        train_runs.resume_run(mine["id"], "train.log")  # in the folder, not a weight


def test_an_explicit_checkpoint_is_refused_for_its_own_reason(runs_home):
    """A run with nothing saved yet must still say why *that* path was refused,
    rather than the generic 'nothing to resume from'."""
    run = _make_run()
    elsewhere = run["path"] + "/../2026-other/checkpoints/step-100/x.safetensors"
    with pytest.raises(RuntimeError, match="not a checkpoint of this run"):
        train_runs.resume_run(run["id"], elsewhere)


# --- moving a run between machines -------------------------------------------


def test_export_keeps_weights_and_provenance_and_drops_the_cache(runs_home, tmp_path):
    run = _make_run()
    _cache(run, kb=16)
    _checkpoint(run, 400, kb=8)
    out = train_runs.export_run(run["id"], tmp_path / "out")
    folder = Path(out["path"])
    assert (folder / "checkpoints" / "step-400" / "lora-400.safetensors").is_file()
    assert (folder / "config" / run["id"] / "config.json").is_file()
    assert (folder / "state.json").is_file()
    assert (folder / "train.log").is_file()
    assert not (folder / "cache").exists()  # the bulk, and recomputable
    manifest = json.loads((folder / "EXPORT.json").read_text())
    assert manifest["cache_included"] is False and manifest["id"] == run["id"]
    # What the manifest counts is what is actually there, weights and log included.
    assert manifest["bytes"] == out["bytes"] == _dir_bytes(folder) - (
        folder / "EXPORT.json"
    ).stat().st_size
    assert out["files"] == len([item for item in folder.rglob("*") if item.is_file()]) - 1


def test_export_can_include_the_cache_when_asked(runs_home, tmp_path):
    run = _make_run()
    _cache(run, kb=16)
    out = train_runs.export_run(run["id"], tmp_path / "out", include_cache=True)
    assert (Path(out["path"]) / "cache" / "vae" / "0000.pt").is_file()
    assert out["bytes"] >= 16 * 1024


def test_export_refuses_to_overwrite_a_folder_it_did_not_write(runs_home, tmp_path):
    run = _make_run()
    train_runs.export_run(run["id"], tmp_path / "out")
    with pytest.raises(RuntimeError, match="already exists"):
        train_runs.export_run(run["id"], tmp_path / "out")


def test_import_picks_a_run_up_after_a_folder_round_trip(runs_home, tmp_path):
    run = _make_run(name="Holiday")
    _install(run, _checkpoint(run, 400, kb=8, age_s=10))
    _checkpoint(run, 500, kb=8, age_s=1)
    exported = train_runs.export_run(run["id"], tmp_path / "out")
    train_runs.delete_run(run["id"])  # the machine it left is now empty

    imported = train_runs.import_run(exported["path"])
    assert imported["id"] == run["id"] and imported["name"] == "Holiday"
    assert imported["path"] == str(train_runs.runs_root() / run["id"])
    assert len(train_runs.checkpoint_rows(run["id"])) == 2
    assert (Path(imported["path"]) / "train.log").is_file()
    assert not (Path(imported["path"]) / "cache").exists()  # never travelled
    # It came back as the same run, so retention still knows which one you kept.
    assert [row["installed"] for row in train_runs.checkpoint_rows(run["id"])] == [
        False,
        True,
    ]


def test_import_refuses_to_merge_onto_a_run_that_is_already_here(runs_home, tmp_path):
    run = _make_run()
    _checkpoint(run, 400)
    exported = train_runs.export_run(run["id"], tmp_path / "out")
    with pytest.raises(RuntimeError, match="already in"):
        train_runs.import_run(exported["path"])


def test_import_says_which_folder_it_wanted(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_STUDIO_TRAIN_RUNS", str(tmp_path / "runs"))
    wrong = tmp_path / "loose-checkpoints"
    wrong.mkdir()
    (wrong / "lora-500.safetensors").write_bytes(b"\0")
    with pytest.raises(RuntimeError, match="no state.json"):
        train_runs.import_run(wrong)


def test_import_refuses_an_id_that_walks_out_of_the_runs_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_STUDIO_TRAIN_RUNS", str(tmp_path / "runs"))
    (tmp_path / "runs").mkdir()
    folder = tmp_path / "export"
    folder.mkdir()
    (folder / "state.json").write_text(
        json.dumps({"id": "..", "name": "nope", "status": "finished"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="No training run"):
        train_runs.import_run(folder)


# --- routes ------------------------------------------------------------------


def test_storage_routes_answer_without_a_gpu(runs_home, studio_home):
    from fastapi.testclient import TestClient

    from minimax_studio.worker.server import app

    client = TestClient(app)
    run = _make_run()
    _cache(run, kb=16)
    _checkpoint(run, 100, kb=8, age_s=10)
    _checkpoint(run, 200, kb=8, age_s=1)

    assert client.get("/train/storage").json()["total_bytes"] == 32 * 1024
    detail = client.get(f"/train/runs/{run['id']}/storage").json()
    assert detail["cache_bytes"] == 16 * 1024
    assert client.get("/train/runs/ghost/storage").status_code == 404

    pruned = client.post(f"/train/runs/{run['id']}/prune", json={"keep": 1}).json()
    assert pruned["freed_bytes"] == 8 * 1024
    assert client.post(f"/train/runs/{run['id']}/cache/clear").json()["cleared"]
    assert client.delete(f"/train/runs/{run['id']}").json()["id"] == run["id"]
    assert client.get("/train/storage").json()["runs"] == []


def test_storage_routes_report_busy_runs_as_409(runs_home, studio_home):
    from fastapi.testclient import TestClient

    from minimax_studio.worker.server import app

    client = TestClient(app)
    run = _running(_make_run())
    _checkpoint(run, 100, age_s=10)
    _checkpoint(run, 200, age_s=1)
    assert (
        client.post(f"/train/runs/{run['id']}/prune", json={"keep": 1}).status_code
        == 409
    )
    body = client.post(f"/train/runs/{run['id']}/prune", json={"keep": 1}).json()
    assert "still training" in body["detail"]  # the reason, not just a code
