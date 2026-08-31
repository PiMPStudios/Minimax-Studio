"""Detached SimpleTuner training runs (PLAN-V2 S0).

Training runs for hours and must survive the GUI (and therefore the worker)
closing: we launch with ``start_new_session=True``, keep state in
``<runs-root>/<run-id>/state.json``, and re-derive status from pid liveness on
every listing — there is no in-process truth to lose.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from minimax_studio.worker.train_config import (
    DEFAULT_PRESET,
    PRESETS,
    simpletuner_command_prefix,
    train_preflight,
    validate_music_dataset_dir,
    validate_video_dataset_dir,
    write_run_config,
)

# Popen handles for runs this worker process launched (exit-code reaping).
# Runs launched by a previous worker only ever appear via state.json + pid.
_PROCS: dict[str, subprocess.Popen] = {}

# PLAN-V2: log mtime is the heartbeat. A pid that lives while the log is
# silent for this long is "lost" — still holding the GPU, not making progress.
LOST_HEARTBEAT_S = 300.0
_LIVE_STATUSES = {"running", "queued", "lost"}

# Storage walks can mean thousands of cache files, so its report is cached and
# invalidated by anything that deletes — never polled by the UI's 2 s tick.
_STORAGE_TTL_S = 15.0
_STORAGE_CACHE: dict[str, Any] | None = None
_STORAGE_CACHE_AT = 0.0


def _invalidate_storage_cache() -> None:
    global _STORAGE_CACHE
    _STORAGE_CACHE = None

# Stub trainer: "steps: 10 loss: 0.100". SimpleTuner 4.8.0 tqdm on metal:
# "Steps: 100%|...| 200/200 [03:55<00:00,  1.18s/it, lr=5e-5, step_loss=1.05]"
# The old `steps?: 100` pattern grabbed tqdm's *percent*, not 200/200.
STEP_RE = re.compile(r"(?<![a-z_])steps?[:=\s]+(\d+)", re.IGNORECASE)
TQDM_STEP_RE = re.compile(r"(\d+)/(\d+)\s*\[")
LOSS_RE = re.compile(
    r"(?:step_loss|(?<![a-z_])loss)[:=\s]+([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)


def runs_root() -> Path:
    override = os.environ.get("MINIMAX_STUDIO_TRAIN_RUNS")
    if override:
        return Path(override)
    from minimax_studio.worker.runtime import runtime

    root = Path(runtime.config.output_dir or ".") / "training"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9-]+", "-", (text or "training").lower()).strip("-")
    return cleaned[:40] or "training"


def start_run(
    name: str,
    dataset_dir: str | Path,
    preset: str = DEFAULT_PRESET,
    steps: int = 1000,
    rank: int | None = None,
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if preset not in PRESETS:
        raise RuntimeError(
            f"Unknown training preset '{preset}' "
            f"(known: {', '.join(sorted(PRESETS))})."
        )
    from minimax_studio.worker.datasets import assert_trainable, dataset_spec

    spec = dataset_spec(dataset_dir)
    # Family before fire. An H3 preset pointed at a Music dataset (or the other
    # way round) would train the wrong model and then credit the wrong
    # provenance to the result, so this refusal is outside the force switch.
    preset_row = PRESETS[preset]
    wanted = "video" if preset_row.family == "h3" else "music"
    if spec["kind"] != wanted:
        raise RuntimeError(
            f"Preset '{preset_row.title}' trains "
            f"{'MiniMax H3 — stills and short clips' if preset_row.family == 'h3' else 'MiniMax Music 3 — audio clips'}"
            f", but {dataset_dir} holds a {spec['kind']} dataset "
            f"({spec['stills']} stills, {spec['clips']} clips, "
            f"{spec['audio_files']} audio). Pick the preset for this kind."
        )
    errors = (
        validate_video_dataset_dir(dataset_dir)
        if spec["kind"] == "video"
        else validate_music_dataset_dir(dataset_dir)
    )
    if errors:
        raise RuntimeError("Dataset is not ready to train: " + " ".join(errors[:3]))
    # If it's an app-managed dataset (has a manifest), it must validate clean.

    assert_trainable(Path(dataset_dir).resolve())
    if not os.environ.get("MINIMAX_STUDIO_TRAIN_FORCE"):
        check = train_preflight(preset, dataset_dir)
        if not check["ok"]:
            raise RuntimeError(check["detail"])
    prefix = simpletuner_command_prefix()
    if not prefix:
        raise RuntimeError(
            "SimpleTuner is not installed — run: pip install "
            "'minimax-studio[train]'."
        )

    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{_slug(name)}"
    run_dir = runs_root() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_run_config(
        run_dir,
        run_id,
        dataset_dir,
        preset_name=preset,
        steps=steps,
        rank=rank,
        validation=validation,
        dataset_spec=spec,
    )
    log = open(run_dir / "train.log", "ab")
    cmd = [*prefix, "train", f"env={run_id}"]
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=run_dir,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=_trainer_env(),
        )
    finally:
        log.close()
    state = {
        "id": run_id,
        "name": name,
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "dataset_kind": spec["kind"],
        # Kept with the run, not re-derived at resume: if the dataset folder is
        # gone or moved, resuming must still write the same kind of config.
        "dataset_spec": spec,
        "family": preset_row.family,
        "preset": preset,
        "steps": int(steps),
        "rank": int(rank or PRESETS[preset].lora_rank),
        "cmd": cmd,
        "pid": proc.pid,
        "started_at": time.time(),
        "status": "running",
        "cancel_requested": False,
        "exit_code": None,
        "finished_at": None,
    }
    _write_state(run_dir, state)
    _PROCS[run_id] = proc
    # `path` is for the UI (Open folder, Open log); state.json stays portable.
    return {**state, "path": str(run_dir)}


def list_runs() -> list[dict[str, Any]]:
    root = runs_root()
    if not root.is_dir():
        return []
    rows = []
    for child in sorted(root.iterdir(), reverse=True):
        state = _read_state(child)
        if state:
            rows.append({**_refresh(child, state), "path": str(child)})
    return rows


def live_runs() -> list[dict[str, Any]]:
    """Runs whose process is still up.

    GPU etiquette cuts both ways: ``start_run`` refuses to join an active
    generation, and generate preflight warns when a run is already holding the
    card. Warn rather than block — cancelling someone's three-hour run should
    stay their decision, not a side effect of pressing Generate.
    """
    return [row for row in list_runs() if row.get("status") in _LIVE_STATUSES]


def _require_id(value: str, what: str) -> str:
    name = str(value or "")
    if not name or Path(name).name != name or name in {".", ".."}:
        raise RuntimeError(f"No {what} '{value}'.")
    return name


def get_run(run_id: str) -> tuple[Path, dict[str, Any]]:
    run_dir = runs_root() / _require_id(run_id, "training run")
    state = _read_state(run_dir)
    if not state:
        raise RuntimeError(f"No training run '{run_id}'.")
    return run_dir, {**_refresh(run_dir, state), "path": str(run_dir)}


def cancel_run(run_id: str) -> dict[str, Any]:
    run_dir, state = get_run(run_id)
    if state["status"] not in {"running", "lost"}:
        return state
    state["cancel_requested"] = True
    _write_state(run_dir, state)
    pid = int(state.get("pid") or 0)
    if not pid:
        return state
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                timeout=15,
            )
        else:
            # start_new_session made the pid its own process group; signal
            # the group so SimpleTuner's DataLoader children die too.
            os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError, subprocess.SubprocessError):
        pass
    return _refresh(run_dir, state)


def log_tail(run_id: str, lines: int = 80) -> list[str]:
    run_dir, _state = get_run(run_id)
    path = run_dir / "train.log"
    if not path.is_file():
        return []
    data = path.read_bytes()[-64_000:]
    return data.decode("utf-8", "replace").splitlines()[-lines:]


def progress(run_id: str) -> dict[str, Any]:
    run_dir, state = get_run(run_id)
    out: dict[str, Any] = {
        "step": None,
        "total_steps": state.get("steps"),
        "percent": None,
        "loss": None,
        "checkpoints": [],
    }
    log = run_dir / "train.log"
    if log.is_file():
        text = log.read_text(encoding="utf-8", errors="replace")[-200_000:]
        tqdm_hits = TQDM_STEP_RE.findall(text)
        if tqdm_hits:
            out["step"] = int(tqdm_hits[-1][0])
        else:
            steps = STEP_RE.findall(text)
            if steps:
                out["step"] = int(steps[-1])
        losses = LOSS_RE.findall(text)
        if losses:
            out["loss"] = float(losses[-1])
    total = out.get("total_steps")
    if out["step"] is not None and total:
        out["percent"] = min(1.0, out["step"] / float(total))
    out["checkpoints"] = [
        path.relative_to(run_dir).as_posix()
        for path in sorted((run_dir / "checkpoints").rglob("*.safetensors"))
    ]
    return out


def install_adapter(run_id: str, path: str | None = None) -> dict[str, Any]:
    """Copy the newest (or chosen) trained .safetensors into the LoRA picker."""
    from minimax_studio.worker import adapters
    from minimax_studio.worker.loras import import_lora

    run_dir, state = get_run(run_id)
    if path:
        source = Path(path)
        if not source.is_absolute():
            source = run_dir / path
        try:
            resolved = source.resolve()
        except OSError:
            resolved = source
        if (
            not resolved.is_file()
            or resolved.suffix.lower() != ".safetensors"
            or run_dir.resolve() not in resolved.parents
        ):
            raise RuntimeError(
                f"“{path}” is not a checkpoint of this run — pick a "
                ".safetensors inside the run folder."
            )
        source = resolved
    else:
        candidates = sorted(
            (run_dir / "checkpoints").rglob("*.safetensors"),
            key=lambda item: item.stat().st_mtime,
        )
        if not candidates:
            raise RuntimeError(
                "No .safetensors checkpoint in this run yet — check the log."
            )
        source = candidates[-1]
    # SimpleTuner's default export is often ``pytorch_lora_weights.safetensors``.
    # Two runs must not install on top of each other; the run name is the
    # disambiguator, and import_lora still de-collides if that name is taken.
    dest_name = f"{_slug(str(state.get('name') or run_id))}-{source.name}"
    family = str(state.get("family") or "")
    lora_kind = (
        "h3"
        if family == "h3" or str(state.get("dataset_kind") or "") == "video"
        else "music"
    )
    row = import_lora(str(source), dest_name=dest_name, kind=lora_kind)
    # import_lora filed it as "imported"; this run is the real provenance, and
    # the upsert keys on the file name, so the row is upgraded, not duplicated.
    adapter = adapters.record_trained(state, row, source)
    row["trained_run"] = run_id
    row["adapter"] = adapter
    return row


# --- Long-run hardening (PLAN-V2 S5) ----------------------------------------
#
# A run that goes well writes tens of GB: checkpoints every N steps plus a VAE
# and text-embed cache. Nothing in SimpleTuner cleans that up after itself, so
# the app does — with the numbers named before anything is deleted, and never
# while the process is still holding those files open (on Windows that is the
# difference between freeing disk and a half-deleted run).


def _folder_bytes(path: Path) -> int:
    if not path.is_dir():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def checkpoint_rows(run_id: str) -> list[dict[str, Any]]:
    """Every .safetensors this run wrote, newest first, with what retention
    actually needs: size, when it landed, and whether it was **installed**.

    An adapter the user chose to keep is the only "best checkpoint" signal this
    contract has — there is no eval score in SimpleTuner's stdout for us to
    rank by, so newest-N plus whatever you kept is the honest policy, and it is
    written down rather than invented.
    """
    run_dir, _state = get_run(run_id)
    root = run_dir / "checkpoints"
    if not root.is_dir():
        return []
    from minimax_studio.worker import adapters

    registry = adapters.load_registry()
    installed_names = {str(row.get("file", "")).lower() for row in registry}
    installed_sources = {str(row.get("checkpoint", "")) for row in registry}
    rows = []
    for path in sorted(
        root.rglob("*.safetensors"), key=lambda item: item.stat().st_mtime, reverse=True
    ):
        rows.append(
            {
                # POSIX separators on purpose: this string is part of the JSON
                # contract (dialog rows, prune plans, export manifests), and a
                # run folder copied off a Windows machine should read the same.
                "path": path.relative_to(run_dir).as_posix(),
                "abs": str(path),
                "bytes": path.stat().st_size,
                "written_at": path.stat().st_mtime,
                "installed": (
                    path.name.lower() in installed_names
                    or str(path) in installed_sources
                ),
            }
        )
    return rows


def storage(run_id: str) -> dict[str, Any]:
    """One run's footprint, on demand — walking a cache is not a 2-second-poll
    kind of operation."""
    import shutil

    run_dir, state = get_run(run_id)
    cache = _folder_bytes(run_dir / "cache")
    checkpoints = _folder_bytes(run_dir / "checkpoints")
    return {
        "id": run_id,
        "name": state.get("name"),
        "status": state.get("status"),
        "path": str(run_dir),
        "cache_bytes": cache,
        "checkpoint_bytes": checkpoints,
        "bytes": cache + checkpoints,
        "free_gb": round(shutil.disk_usage(run_dir).free / (1024**3), 1),
        "checkpoints": checkpoint_rows(run_id),
    }


def storage_report() -> dict[str, Any]:
    """Across all runs, for the storage dialog. Cached: the dialog is opened,
    not polled, and the walk is the expensive part."""
    import shutil

    global _STORAGE_CACHE, _STORAGE_CACHE_AT
    root = runs_root()
    now = time.monotonic()
    if _STORAGE_CACHE is not None and now - _STORAGE_CACHE_AT < _STORAGE_TTL_S:
        return _STORAGE_CACHE
    rows = []
    children = sorted(root.iterdir(), reverse=True) if root.is_dir() else []
    for child in children:
        state = _read_state(child)
        if not state:
            continue
        cache = _folder_bytes(child / "cache")
        checkpoints = _folder_bytes(child / "checkpoints")
        rows.append(
            {
                "id": state.get("id") or child.name,
                "name": state.get("name") or child.name,
                "status": state.get("status"),
                "cache_bytes": cache,
                "checkpoint_bytes": checkpoints,
                "bytes": cache + checkpoints,
            }
        )
    report = {
        "runs_root": str(root),
        "free_gb": round(shutil.disk_usage(root).free / (1024**3), 1),
        "total_bytes": sum(row["bytes"] for row in rows),
        "runs": rows,
    }
    _STORAGE_CACHE, _STORAGE_CACHE_AT = report, now
    return report


def _refuse_if_running(run_id: str, action: str) -> tuple[Path, dict[str, Any]]:
    run_dir, state = get_run(run_id)
    if state.get("status") in _LIVE_STATUSES:
        raise RuntimeError(
            f"Run “{state.get('name')}” is still training (pid "
            f"{state.get('pid')}) — not {action} files under it while it lives. "
            "Cancel it or wait for the step to finish."
        )
    return run_dir, state


def clear_cache(run_id: str) -> dict[str, Any]:
    """Delete the VAE and text-embed caches. Derived data: the next run
    rebuilds them — slowly, but correctly."""
    import shutil

    run_dir, _state = _refuse_if_running(run_id, "clearing")
    cache = run_dir / "cache"
    if not cache.is_dir():
        return {"freed_bytes": 0, "cleared": False}
    freed = _folder_bytes(cache)
    shutil.rmtree(cache)
    _invalidate_storage_cache()
    return {"freed_bytes": freed, "cleared": True}


def _prune_plan(
    rows: list[dict[str, Any]], keep: int, run_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Decide what retention would do, without touching the disk.

    Returns ``(survivors, doomed)`` where each doomed entry carries the absolute
    path to remove, the run-relative path to show, the bytes that removal frees,
    and whether the whole step folder goes or only one weight file. The dialog
    asks its question with these numbers, so the figure in the confirmation is
    the figure in the result.
    """
    keep = max(1, int(keep))
    survivors = rows[:keep] + [row for row in rows[keep:] if row["installed"]]
    keep_abs = {row["abs"] for row in survivors}
    kept_dirs = {str(Path(row["abs"]).parent) for row in survivors}
    doomed: list[dict[str, Any]] = []
    seen_dirs: set[str] = set()
    for row in rows:
        if row["abs"] in keep_abs:
            continue
        folder = str(Path(row["abs"]).parent)
        if folder in seen_dirs:
            continue
        seen_dirs.add(folder)
        if folder not in kept_dirs:
            # The weights plus whatever else that step wrote (accelerator state).
            doomed.append(
                {
                    "abs": folder,
                    "path": Path(folder).relative_to(run_dir).as_posix(),
                    "whole_folder": True,
                    "bytes": _folder_bytes(Path(folder)),
                }
            )
        else:
            doomed.append(
                {
                    "abs": row["abs"],
                    "path": row["path"],
                    "whole_folder": False,
                    "bytes": int(row["bytes"]),
                }
            )
    return survivors, doomed


def prune_checkpoints(
    run_id: str, keep: int = 3, dry_run: bool = False
) -> dict[str, Any]:
    """Keep the newest ``keep`` checkpoints plus every installed one.

    ``keep`` is clamped to at least 1 — an empty run folder is not disk relief,
    it is a lost weekend. Installed adapters are never touched.

    A pruned checkpoint takes its **whole step folder** with it, not just the
    ``.safetensors``: SimpleTuner leaves accelerator/optimiser state beside the
    weights, and deleting only the weights would free a tenth of what the dialog
    promised while leaving the folder behind. When two checkpoints share one
    folder and only one is doomed, the folder stays and just that file goes.

    ``dry_run`` answers "what would this free?" with the same code path that then
    does it — the dialog needs the number before the deletion, not after.
    """
    import shutil

    run_dir, _state = _refuse_if_running(run_id, "pruning")
    rows = checkpoint_rows(run_id)
    if not rows:
        return {"kept": [], "removed": [], "freed_bytes": 0, "dry_run": dry_run}
    survivors, doomed = _prune_plan(rows, keep, run_dir)
    removed: list[str] = []
    freed = 0
    if not dry_run:
        for entry in doomed:
            target = Path(entry["abs"])
            try:
                if entry["whole_folder"]:
                    shutil.rmtree(target)
                else:
                    target.unlink()
            except OSError:
                continue
            removed.append(entry["path"])
            freed += int(entry["bytes"])
        # Whatever step folders a single-file removal left empty go too.
        checkpoints = run_dir / "checkpoints"
        if checkpoints.is_dir():
            for folder in sorted(checkpoints.rglob("*"), reverse=True):
                try:
                    if folder.is_dir() and not any(folder.iterdir()):
                        folder.rmdir()
                except OSError:
                    continue
        _invalidate_storage_cache()
    return {
        "kept": [row["path"] for row in survivors],
        "removed": removed if not dry_run else [entry["path"] for entry in doomed],
        "freed_bytes": sum(int(entry["bytes"]) for entry in doomed) if dry_run else freed,
        "dry_run": bool(dry_run),
    }


def _resume_checkpoint_arg(run_dir: Path, safetensors: Path) -> str:
    """SimpleTuner resumes from a checkpoint *directory* (or ``latest``), not
    from an exported LoRA file. ``--init_lora`` is the safetensors-only path
    and cannot combine with ``resume_from_checkpoint``."""
    parent = safetensors.parent
    checkpoints = run_dir / "checkpoints"
    try:
        if parent.resolve() == checkpoints.resolve():
            return str(safetensors)
    except OSError:
        pass
    return str(parent)


def resume_run(run_id: str, checkpoint: str | None = None) -> dict[str, Any]:
    """Continue a finished, failed or cancelled run from one of its checkpoints.

    Same run dir, same caches, new process: the weights, the log and the
    provenance keep one home instead of a second folder pretending to be a new
    project.
    """
    run_dir, state = _refuse_if_running(run_id, "resuming")
    rows = checkpoint_rows(run_id)
    if checkpoint:
        source = Path(str(checkpoint))
        if not source.is_absolute():
            source = run_dir / str(checkpoint)
        if (
            not source.is_file()
            or source.suffix.lower() != ".safetensors"
            or run_dir.resolve() not in source.resolve().parents
        ):
            # Weights from another run would train the wrong base and credit the
            # wrong provenance, so this stays a hard no — with the alternative.
            raise RuntimeError(
                f"“{checkpoint}” is not a checkpoint of this run"
                + (
                    f" — pick one of {', '.join(row['path'] for row in rows[:3])}."
                    if rows
                    else " — this run has not written one to checkpoints/ yet."
                )
            )
        resume_from = _resume_checkpoint_arg(run_dir, source)
    elif rows:
        # SimpleTuner's own token: newest complete checkpoint under output_dir.
        resume_from = "latest"
        source = Path(rows[0]["abs"])
    else:
        raise RuntimeError(
            f"Run “{state.get('name')}” has no checkpoint to resume from — "
            "nothing reached checkpoints/ in this run."
        )
    prefix = simpletuner_command_prefix()
    if not prefix:
        raise RuntimeError(
            "SimpleTuner is not installed — resuming needs the pinned [train] "
            "extra on Python 3.12."
        )
    write_run_config(
        run_dir,
        run_id,
        state.get("dataset_dir") or "",
        preset_name=str(state.get("preset") or DEFAULT_PRESET),
        steps=int(state.get("steps") or 1000),
        rank=state.get("rank"),
        resume_from_checkpoint=resume_from,
        dataset_spec=state.get("dataset_spec") or None,
    )
    log = open(run_dir / "train.log", "ab")
    try:
        proc = subprocess.Popen(
            [*prefix, "train", f"env={run_id}"],
            cwd=run_dir,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=_trainer_env(),
        )
    finally:
        log.close()
    state.update(
        {
            "status": "running",
            "pid": proc.pid,
            "started_at": time.time(),
            "exit_code": None,
            "finished_at": None,
            "cancel_requested": False,
            "resumed_from": str(resume_from),
            "resume_count": int(state.get("resume_count") or 0) + 1,
        }
    )
    _write_state(run_dir, state)
    _PROCS[run_id] = proc
    return {**state, "path": str(run_dir)}


def _trainer_env() -> dict[str, str]:
    """SimpleTuner is a subprocess: pin it to Settings → CUDA device."""
    env = os.environ.copy()
    from minimax_studio.worker.runtime import runtime

    env["CUDA_VISIBLE_DEVICES"] = str(max(0, int(runtime.config.cuda_device or 0)))
    # sitecustomize.py: skip SimpleTuner's cudagraph mark when Dynamo is off
    # (torch 2.13 inductor import crash). Trainer-only; the GUI never loads it.
    startup = str(Path(__file__).resolve().parent / "st_startup")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = startup if not existing else startup + os.pathsep + existing
    # Metal 2026-08-30 H3 24G: PyTorch held 5.75 GiB reserved-but-unallocated
    # and died on a 316 MiB alloc. SimpleTuner's own traceback asks for this.
    alloc = env.get("PYTORCH_CUDA_ALLOC_CONF", "").strip()
    if "expandable_segments" not in alloc:
        env["PYTORCH_CUDA_ALLOC_CONF"] = (
            f"{alloc},expandable_segments:True" if alloc else "expandable_segments:True"
        )
    return env


def export_run(
    run_id: str, dest: str | Path, include_cache: bool = False
) -> dict[str, Any]:
    """Copy a run out — archive it, or move it to the machine with the GPU.

    Caches stay behind by default: they are most of the bytes and the first
    thing a receiving run recomputes anyway.
    """
    import shutil

    run_dir, state = get_run(run_id)
    parent = Path(str(dest)).expanduser()
    target = parent / run_id
    staging = parent / f".{run_id}.exporting"
    if target.exists():
        raise RuntimeError(
            f"{target} already exists — move or delete it first; Studio will "
            "not overwrite a folder it did not write."
        )
    if staging.exists():
        shutil.rmtree(staging)
    copied = 0
    total = 0
    try:
        staging.mkdir(parents=True)
        for item in sorted(run_dir.rglob("*")):
            if not item.is_file():
                continue
            relative = item.relative_to(run_dir)
            if not include_cache and "cache" in relative.parts:
                continue
            into = staging / relative
            into.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, into)
            copied += 1
            total += item.stat().st_size
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    manifest = {
        "version": 1,
        "id": run_id,
        "name": state.get("name"),
        "exported_at": time.time(),
        "files": copied,
        "bytes": total,
        "cache_included": bool(include_cache),
    }
    try:
        (staging / "EXPORT.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"path": str(target), "files": copied, "bytes": total, "id": run_id}


def import_run(folder: str | Path) -> dict[str, Any]:
    """Bring an exported run folder back into the list.

    The only requirement is ``state.json`` — that file is what makes a folder a
    Studio run. An id already present is refused rather than merged: mixing two
    runs' checkpoints is how provenance becomes fiction.
    """
    import shutil

    source = Path(str(folder)).expanduser()
    state = _read_state(source)
    if not state or not state.get("id"):
        raise RuntimeError(
            f"{source} is not a Studio training run — no state.json inside. "
            "Import the run folder itself, not the checkpoints folder inside it."
        )
    run_id = _require_id(str(state["id"]), "training run")
    target = (runs_root() / run_id).resolve()
    root = runs_root().resolve()
    if target.parent != root:
        raise RuntimeError(
            f"Run “{run_id}” is not a name Studio will import under {root}."
        )
    if target.exists():
        raise RuntimeError(
            f"Run “{run_id}” is already in {runs_root()}. Delete the copy "
            "here first (its Storage dialog can) — importing would mix two "
            "runs' checkpoints."
        )
    shutil.copytree(source, target)
    _invalidate_storage_cache()
    state["imported_at"] = time.time()
    _write_state(target, state)
    return {**state, "path": str(target)}


def delete_run(run_id: str) -> dict[str, Any]:
    """Delete a finished run's folder, reporting what that freed.

    Installed adapters are *copies* in the LoRA folder, so this never breaks
    the picker or the registry.
    """
    import shutil

    run_dir, _state = _refuse_if_running(run_id, "deleting")
    freed = _folder_bytes(run_dir)
    shutil.rmtree(run_dir)
    _invalidate_storage_cache()
    return {"id": run_id, "freed_bytes": freed}


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    import ctypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    handle = ctypes.windll.kernel32.OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION, False, pid
    )
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == STILL_ACTIVE
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def _log_beat(run_dir: Path, state: dict[str, Any]) -> float:
    """Newest of log mtime and start time — a run that has not written yet
    is not lost, it is still starting."""
    beat = float(state.get("started_at") or 0)
    log = run_dir / "train.log"
    try:
        if log.is_file():
            beat = max(beat, log.stat().st_mtime)
    except OSError:
        pass
    return beat


def _maybe_lost(run_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    silent = time.time() - _log_beat(run_dir, state)
    if silent <= LOST_HEARTBEAT_S:
        if state.get("status") == "lost":
            state["status"] = "running"
            _write_state(run_dir, state)
        return state
    if state.get("status") != "lost":
        state["status"] = "lost"
        _write_state(run_dir, state)
    return state


def _refresh(run_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    if state.get("status") not in {"running", "lost"}:
        return state
    run_id = str(state.get("id"))
    proc = _PROCS.get(run_id)
    code: int | None = None
    if proc is not None:
        # We launched it: poll() reaps zombies and yields the real exit code.
        code = proc.poll()
        if code is None:
            return _maybe_lost(run_dir, state)
        _PROCS.pop(run_id, None)
    else:
        # No handle (worker restarted): pid liveness is all we have.
        pid = int(state.get("pid") or 0)
        if pid and _pid_alive(pid):
            return _maybe_lost(run_dir, state)
        if state.get("exit_code") is None:
            state["status"] = (
                "cancelled" if state.get("cancel_requested") else "ended-unknown"
            )
            state["finished_at"] = state.get("finished_at") or time.time()
            _write_state(run_dir, state)
            return state
        code = int(state["exit_code"])
    # Cancel wins over a clean exit: SimpleTuner often SIGTERM-exits 0, and
    # calling that "completed" would install a half-trained adapter as a success.
    if state.get("cancel_requested"):
        state["status"] = "cancelled"
    elif code == 0:
        state["status"] = "completed"
    else:
        state["status"] = "failed"
    state["exit_code"] = code
    state["finished_at"] = state.get("finished_at") or time.time()
    _write_state(run_dir, state)
    return state


def _read_state(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "state.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_state(run_dir: Path, state: dict[str, Any]) -> None:
    (run_dir / "state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
