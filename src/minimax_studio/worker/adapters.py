"""Adapter registry — provenance for every LoRA the app can load (PLAN-V2 S3).

What can be *loaded* is decided by the files on disk: :func:`loras.list_loras`
walks the folders and this module does not argue with it. What a bare filename
cannot tell you is where an adapter **came from** — which dataset, which run,
which pinned trainer, at what rank. That is what lives here, in
``<models root>/adapters.json``, one row per LoRA file name (the id the picker
already uses):

``trained``    installed from a Studio training run — full provenance
``imported``   brought in by hand through File ▸ Import
``untracked``  on disk with no row; still listed, so the picker is one honest
               list instead of two half-truths

Rows are a cache, not a lock: delete the file and the row stays and reports
``on_disk: false`` — provenance of a deleted adapter is still worth reading —
and forgetting a row never touches the file.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from minimax_studio.worker.loras import list_loras
from minimax_studio.worker.runtime import runtime

#: The PiMP loop's one-click audition: short, soft, and unmistakably a test.
AUDITION_STRENGTH = 0.8
AUDITION_SECONDS = 30.0

CLIP_SUFFIXES = {
    ".wav",
    ".flac",
    ".mp3",
    ".mp4",
    ".mov",
    ".webm",
}


def registry_path() -> Path:
    return runtime.config.models_root() / "adapters.json"


def load_registry() -> list[dict[str, Any]]:
    """Tolerant by design: a bad file costs provenance, never the app."""
    path = registry_path()
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, list):  # the pre-version shape
        payload = {"adapters": payload}
    rows = payload.get("adapters") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict) and row.get("file")]


def save_registry(rows: list[dict[str, Any]]) -> None:
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": 1, "adapters": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def record(row: dict[str, Any]) -> dict[str, Any]:
    """Upsert by file name — the same key the LoRA picker uses, so installing
    the same adapter twice updates its row instead of doubling the list."""
    file_name = str(row.get("file") or "")
    if not file_name:
        raise RuntimeError("An adapter row needs a file name.")
    existing = next(
        (item for item in load_registry() if item.get("file") == file_name), None
    )
    merged = {**(existing or {}), **row}
    merged["file"] = file_name
    merged.setdefault("id", file_name)
    merged.setdefault("kind", "music")
    merged.setdefault("created_at", time.time())
    merged["updated_at"] = time.time()
    rows = [item for item in load_registry() if item.get("file") != file_name]
    rows.append(merged)
    save_registry(rows)
    return merged


def forget(adapter_id: str) -> None:
    """Drop the provenance row. The .safetensors file is not touched."""
    key = str(adapter_id).lower()
    rows = load_registry()
    kept = [item for item in rows if str(item.get("file")).lower() != key]
    if len(kept) == len(rows):
        raise RuntimeError(f"No registry row for adapter '{adapter_id}'.")
    save_registry(kept)


def dataset_fingerprint(folder: str | Path) -> dict[str, Any]:
    """What an adapter was trained on, in a form worth checking months later.

    A hash over sorted ``name|size`` — deliberately not mtimes: copying a
    dataset, checking it out, or touching it must not look like new training
    data, while deleting or replacing a clip must.
    """
    root = Path(folder)
    out: dict[str, Any] = {
        "path": str(root),
        "exists": root.is_dir(),
        "clip_count": 0,
        "manifest_hash": None,
    }
    if not root.is_dir():
        return out
    clips = []
    for path in sorted(root.iterdir()):
        if path.suffix.lower() in CLIP_SUFFIXES and not path.name.startswith("."):
            try:
                clips.append(f"{path.name}|{path.stat().st_size}")
            except OSError:
                continue
    out["clip_count"] = len(clips)
    if clips:
        digest = hashlib.sha256("\n".join(clips).encode("utf-8")).hexdigest()
        out["manifest_hash"] = digest[:12]
    return out


def typical_caption(folder: str | Path) -> str:
    """The caption to audition with: the one the adapter actually saw most.

    Most-common beats "first file alphabetically", which is just a coincidence
    of naming. Empty when the dataset is gone — the caller decides what to say.
    """
    root = Path(folder)
    if not root.is_dir():
        return ""
    counts: Counter[str] = Counter()
    for path in sorted(root.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if text:
            counts[text] += 1
    return counts.most_common(1)[0][0] if counts else ""


def list_adapters() -> list[dict[str, Any]]:
    """Registry ∪ disk. A file nobody registered shows as ``untracked``;
    a row whose file was deleted stays, flagged, rather than vanishing."""
    rows = load_registry()
    by_file = {str(row.get("file", "")).lower(): row for row in rows}
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for lora in list_loras():
        file_name = Path(str(lora.get("path") or "")).name
        key = file_name.lower()
        seen.add(key)
        row = dict(by_file.get(key) or {})
        row.update(
            {
                "id": row.get("id") or file_name,
                "file": file_name,
                "path": lora.get("path"),
                "name": row.get("name") or lora.get("name") or Path(file_name).stem,
                "on_disk": True,
            }
        )
        row.setdefault("kind", "music")
        row.setdefault("source", "untracked")
        out.append(_decorate(row))
    for key, row in by_file.items():
        if key not in seen:
            out.append(_decorate({**row, "on_disk": False}))
    out.sort(key=lambda row: row.get("created_at") or 0, reverse=True)
    return out


def get_adapter(adapter_id: str) -> dict[str, Any]:
    key = str(adapter_id).lower()
    for row in list_adapters():
        if str(row.get("id", "")).lower() == key or str(row.get("file", "")).lower() == key:
            return row
    raise RuntimeError(
        f"No adapter '{adapter_id}' in the registry or in a LoRA folder."
    )


def _decorate(row: dict[str, Any]) -> dict[str, Any]:
    """Reality checks the UI should not have to do: is the dataset still
    there, and what would an audition sing?"""
    out = dict(row)
    out.setdefault("dataset", {})
    dataset = out.get("dataset") or {}
    path = str(dataset.get("path") or "")
    out["dataset_exists"] = bool(path and Path(path).is_dir())
    out["audition_prompt"] = (
        typical_caption(path) if out.get("dataset_exists") else ""
    )
    out["can_audition"] = bool(
        out.get("on_disk") and out.get("kind") == "music"
    )
    return out


def record_imported(lora_row: dict[str, Any]) -> dict[str, Any]:
    """A hand-imported file: we know its name and that we did not train it."""
    file_name = Path(str(lora_row.get("path") or "")).name
    return record(
        {
            "file": file_name,
            "name": lora_row.get("name") or Path(file_name).stem,
            "kind": "music",
            "source": "imported",
            "path": lora_row.get("path"),
        }
    )


def record_trained(
    run_state: dict[str, Any], lora_row: dict[str, Any], checkpoint: str | Path
) -> dict[str, Any]:
    """Full provenance for a run Studio launched. This is the row that answers
    'which twelve clips made this sound like that?' a year from now."""
    from minimax_studio.worker.train_config import SIMPLETUNER_PIN

    file_name = Path(str(lora_row.get("path") or checkpoint)).name
    return record(
        {
            "file": file_name,
            # The picker shows the file stem; so does this page. One name.
            "name": Path(file_name).stem,
            "kind": "music",
            "source": "trained",
            "path": lora_row.get("path") or str(checkpoint),
            "base_pack": "music3-cuda",
            "trainer": f"simpletuner {SIMPLETUNER_PIN}",
            "run_id": run_state.get("id"),
            "run_name": run_state.get("name"),
            "preset": run_state.get("preset"),
            "steps": run_state.get("steps"),
            "rank": run_state.get("rank"),
            "dataset": dataset_fingerprint(run_state.get("dataset_dir") or ""),
            "checkpoint": str(checkpoint),
        }
    )


def audition(
    adapter_id: str,
    prompt: str = "",
    duration_s: float | None = None,
    backend: str = "auto",
) -> dict[str, Any]:
    """Queue the short, tagged render that answers "was that worth 3 hours?"

    It is an ordinary generate job — same queue, same Inspector, same History
    row — with the adapter at 0.8 and the caption the adapter was actually
    trained on, plus ``audition: audition:<adapter>`` so History can badge it
    and restore-to-generate still works.
    """
    from minimax_studio.worker.jobs import JobRequest, start_job

    row = get_adapter(adapter_id)
    if not row.get("on_disk"):
        raise RuntimeError(
            f"“{row.get('name')}” has no file on disk to audition — reinstall "
            "it from its run, or Forget it here."
        )
    if row.get("kind") != "music":
        raise RuntimeError(
            "H3 adapters are auditioned as a still pair in PLAN-V2 S4 — Music "
            "adapters only for now."
        )
    text = (prompt or "").strip() or str(row.get("audition_prompt") or "").strip()
    if not text:
        raise RuntimeError(
            f"Nothing to audition “{row.get('name')}” with: "
            + (
                "its dataset folder is gone, so there is no caption to reuse."
                if (row.get("dataset") or {}).get("path")
                else "it was imported by hand and has no dataset behind it."
            )
            + " Type a prompt in the box, then Audition."
        )
    request = JobRequest(
        kind="music",
        backend=backend,
        mode="ttm",
        prompt=text,
        duration_s=float(duration_s or AUDITION_SECONDS),
        loras=[{"id": str(row["path"]), "strength": AUDITION_STRENGTH}],
        audition=f"audition:{row['id']}",
    )
    job = start_job(request)
    return {
        "job_id": job["id"],
        "adapter": row["id"],
        "prompt": text,
        "strength": AUDITION_STRENGTH,
        "duration_s": request.duration_s,
        "backend": backend,
    }
