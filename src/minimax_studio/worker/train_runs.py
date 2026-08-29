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
    write_run_config,
)

# Popen handles for runs this worker process launched (exit-code reaping).
# Runs launched by a previous worker only ever appear via state.json + pid.
_PROCS: dict[str, subprocess.Popen] = {}

STEP_RE = re.compile(r"(?<![a-z_])steps?[:=\s]+(\d+)", re.IGNORECASE)
LOSS_RE = re.compile(r"(?<![a-z_])loss[:=\s]+([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)")


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
    errors = validate_music_dataset_dir(dataset_dir)
    if errors:
        raise RuntimeError("Dataset is not ready to train: " + " ".join(errors[:3]))
    # If it's an app-managed dataset (has a manifest), it must validate clean.
    from minimax_studio.worker.datasets import assert_trainable

    assert_trainable(Path(dataset_dir).resolve())
    if not os.environ.get("MINIMAX_STUDIO_TRAIN_FORCE"):
        check = train_preflight(preset)
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
        )
    finally:
        log.close()
    state = {
        "id": run_id,
        "name": name,
        "dataset_dir": str(Path(dataset_dir).resolve()),
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
    return [row for row in list_runs() if row.get("status") in {"running", "queued"}]


def get_run(run_id: str) -> tuple[Path, dict[str, Any]]:
    run_dir = runs_root() / run_id
    state = _read_state(run_dir)
    if not state:
        raise RuntimeError(f"No training run '{run_id}'.")
    return run_dir, {**_refresh(run_dir, state), "path": str(run_dir)}


def cancel_run(run_id: str) -> dict[str, Any]:
    run_dir, state = get_run(run_id)
    if state["status"] != "running":
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
        steps = STEP_RE.findall(text)
        losses = LOSS_RE.findall(text)
        if steps:
            out["step"] = int(steps[-1])
        if losses:
            out["loss"] = float(losses[-1])
    total = out.get("total_steps")
    if out["step"] is not None and total:
        out["percent"] = min(1.0, out["step"] / float(total))
    out["checkpoints"] = [
        str(path.relative_to(run_dir))
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
    row = import_lora(str(source))
    # import_lora filed it as "imported"; this run is the real provenance, and
    # the upsert keys on the file name, so the row is upgraded, not duplicated.
    adapter = adapters.record_trained(state, row, source)
    row["trained_run"] = run_id
    row["adapter"] = adapter
    return row


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


def _refresh(run_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    if state.get("status") != "running":
        return state
    run_id = str(state.get("id"))
    proc = _PROCS.get(run_id)
    code: int | None = None
    if proc is not None:
        # We launched it: poll() reaps zombies and yields the real exit code.
        code = proc.poll()
        if code is None:
            return state
        _PROCS.pop(run_id, None)
    else:
        # No handle (worker restarted): pid liveness is all we have.
        pid = int(state.get("pid") or 0)
        if pid and _pid_alive(pid):
            return state
        if state.get("exit_code") is None:
            state["status"] = (
                "cancelled" if state.get("cancel_requested") else "ended-unknown"
            )
            state["finished_at"] = state.get("finished_at") or time.time()
            _write_state(run_dir, state)
            return state
        code = int(state["exit_code"])
    if code == 0:
        state["status"] = "completed"
    elif state.get("cancel_requested"):
        state["status"] = "cancelled"
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
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_state(run_dir: Path, state: dict[str, Any]) -> None:
    (run_dir / "state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
