"""Training datasets (PLAN-V2 S1).

A dataset is a plain folder in the layout SimpleTuner reads natively —
``track.wav`` + ``track.txt`` caption + optional ``track.lyrics`` — plus a
``dataset.json`` manifest of ours for name/kind/provenance. The trainer never
touches the manifest; the app uses it to validate before anyone burns GPU
hours, per the standing rule: named numbers, no mystery failures.

WAV duration is probed with the stdlib ``wave`` module on purpose: it needs
no ffmpeg, so CI can generate clips and test the validator honestly.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import wave
from pathlib import Path
from typing import Any

from minimax_studio.worker.runtime import runtime

KINDS = ("music", "video")
MEDIA_BY_KIND = {"music": (".wav", ".flac", ".mp3"), "video": (".mp4", ".mov", ".webm")}
MIN_SECONDS = 3.0
MAX_SECONDS = 300.0


def datasets_root() -> Path:
    override = os.environ.get("MINIMAX_STUDIO_DATASETS")
    if override:
        return Path(override)
    root = Path(runtime.config.output_dir or ".") / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9-]+", "-", (text or "dataset").lower()).strip("-")
    return cleaned[:48] or "dataset"


def create_dataset(name: str, kind: str = "music", notes: str = "") -> dict[str, Any]:
    if kind not in KINDS:
        raise RuntimeError(f"Unknown dataset kind '{kind}' (known: {', '.join(KINDS)}).")
    dataset_id = _slug(name)
    folder = datasets_root() / dataset_id
    if folder.exists():
        raise RuntimeError(
            f"A dataset named '{name}' already exists at {folder}. "
            "Pick another name or delete it first."
        )
    folder.mkdir(parents=True)
    manifest = {
        "id": dataset_id,
        "name": name,
        "kind": kind,
        "created_at": time.time(),
        "notes": notes,
        "last_validation": None,
        # The Build pages hand this straight to /train/runs as dataset_dir, and
        # "Show in folder" needs it too — the layout is ours, the path is not.
        "path": str(folder),
    }
    _write_manifest(folder, manifest)
    return manifest


def list_datasets() -> list[dict[str, Any]]:
    root = datasets_root()
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        manifest = _read_manifest(child)
        if not manifest:
            continue
        manifest = dict(manifest)
        manifest["clip_count"] = len(list_entries(child))
        manifest["path"] = str(child)
        rows.append(manifest)
    rows.sort(key=lambda item: item.get("created_at") or 0, reverse=True)
    return rows


def get_dataset(dataset_id: str) -> tuple[Path, dict[str, Any]]:
    folder = datasets_root() / dataset_id
    manifest = _read_manifest(folder)
    if not manifest:
        raise RuntimeError(f"No dataset '{dataset_id}'.")
    return folder, manifest


def delete_dataset(dataset_id: str) -> None:
    folder, _manifest = get_dataset(dataset_id)
    shutil.rmtree(folder)


def list_entries(folder: Path) -> list[dict[str, Any]]:
    media = (
        set(MEDIA_BY_KIND["music"])
        | set(MEDIA_BY_KIND["video"])
        | {".wav", ".flac", ".mp3", ".mp4", ".mov", ".webm"}
    )
    rows = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in media or path.name.startswith("."):
            continue
        rows.append(
            {
                "file": path.name,
                "has_caption": path.with_suffix(".txt").is_file(),
                "has_lyrics": path.with_suffix(".lyrics").is_file(),
            }
        )
    return rows


def import_folder(dataset_id: str, folder: str) -> dict[str, Any]:
    """Copy audio (+ sibling captions/lyrics) in. Copies, not references —
    cleaning the source folder must not gut a dataset (PLAN-V2 open Q3)."""
    dest, manifest = get_dataset(dataset_id)
    source = Path(folder)
    if not source.is_dir():
        raise RuntimeError(f"Import folder not found: {source}")
    extensions = MEDIA_BY_KIND.get(manifest["kind"], MEDIA_BY_KIND["music"])
    copied: list[str] = []
    captions = 0
    for item in sorted(source.iterdir()):
        if item.suffix.lower() not in extensions:
            continue
        target = _free_name(dest, item.name)
        shutil.copy2(item, target)
        copied.append(target.name)
        for sibling_ext in (".txt", ".lyrics"):
            sibling = item.with_suffix(sibling_ext)
            if sibling.is_file():
                shutil.copy2(sibling, target.with_suffix(sibling_ext))
                captions += 1
    return {"copied": copied, "captions": captions}


def add_from_history(dataset_id: str, history_id: str) -> dict[str, Any]:
    """Grow a dataset from the good generations — the PiMP loop's front half."""
    from minimax_studio.worker.history import get_entry

    dest, manifest = get_dataset(dataset_id)
    try:
        entry = get_entry(history_id)
    except KeyError as exc:
        raise RuntimeError(f"No history entry '{history_id}'.") from exc
    if not entry:
        raise RuntimeError(f"No history entry '{history_id}'.")
    source = Path(str(entry.get("output_path") or ""))
    if not source.is_file():
        raise RuntimeError(
            f"History entry '{history_id}' has no output file on disk to add."
        )
    expected = MEDIA_BY_KIND.get(manifest["kind"], MEDIA_BY_KIND["music"])
    if source.suffix.lower() not in expected:
        raise RuntimeError(
            f"A '{manifest['kind']}' dataset takes {', '.join(expected)} — "
            f"history entry '{history_id}' produced {source.suffix or 'nothing'}."
        )
    target = _free_name(dest, f"{_slug(history_id)}{source.suffix.lower()}")
    shutil.copy2(source, target)
    prompt = str(entry.get("prompt") or "").strip()
    target.with_suffix(".txt").write_text(
        (prompt or "studio generation") + "\n", encoding="utf-8"
    )
    lyrics = str(entry.get("lyrics") or "").strip()
    if lyrics:
        target.with_suffix(".lyrics").write_text(lyrics + "\n", encoding="utf-8")
    return {"added": target.name}


def probe_audio(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
        if rate <= 0:
            raise RuntimeError("unreadable wav (zero sample rate)")
        return {"seconds": frames / float(rate), "format": "wav"}
    # Other formats want ffprobe; that path is only honest if ffmpeg exists.
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("install ffmpeg/ffprobe to probe " + path.suffix)
    import subprocess

    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    try:
        seconds = float(proc.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"unreadable audio ({path.suffix})") from exc
    return {"seconds": seconds, "format": path.suffix.lstrip(".")}


def validate_dataset(dataset_id: str) -> dict[str, Any]:
    folder, manifest = get_dataset(dataset_id)
    report = _validate_dir(folder, manifest)
    manifest = dict(manifest)
    manifest["last_validation"] = {
        "at": report["at"],
        "ok": report["ok"],
        "checked": report["checked"],
        "with_problems": sum(1 for row in report["rows"] if not row["ok"]),
    }
    _write_manifest(folder, manifest)
    (folder / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _validate_dir(folder: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "at": time.time(),
        "ok": False,
        "kind": manifest.get("kind"),
        "checked": 0,
        "rows": [],
    }
    rows: list[dict[str, Any]] = report["rows"]
    if manifest.get("kind") != "music":
        rows.append(
            {
                "file": "(dataset)",
                "ok": False,
                "problems": [
                    "Video dataset validation and the H3 trainer arrive in "
                    "PLAN-V2 S4 — training is Music-only for now."
                ],
            }
        )
        return report
    media_exts = set(MEDIA_BY_KIND["music"])
    stems_seen: set[str] = set()
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in media_exts or path.name.startswith("."):
            continue
        report["checked"] += 1
        stems_seen.add(path.stem)
        problems: list[str] = []
        if not path.with_suffix(".txt").is_file():
            problems.append(f"missing caption {path.stem}.txt")
        if path.with_suffix(".lyrics").is_file() and not path.with_suffix(
            ".txt"
        ).is_file():
            problems.append(".lyrics without a matching .txt caption")
        try:
            seconds = probe_audio(path)["seconds"]
            if seconds < MIN_SECONDS:
                problems.append(
                    f"{seconds:.1f}s is under the {MIN_SECONDS:.0f}s floor"
                )
            elif seconds > MAX_SECONDS:
                problems.append(
                    f"{seconds:.1f}s is over the {MAX_SECONDS:.0f}s cap"
                )
        except (RuntimeError, wave.Error, OSError) as exc:
            problems.append(f"cannot read audio: {exc}")
        rows.append({"file": path.name, "ok": not problems, "problems": problems})
    for path in sorted(folder.glob("*.txt")):
        if path.stem not in stems_seen:
            rows.append(
                {
                    "file": path.name,
                    "ok": False,
                    "problems": ["caption with no matching audio file"],
                }
            )
    report["ok"] = report["checked"] > 0 and all(row["ok"] for row in rows)
    if report["checked"] == 0:
        rows.append(
            {"file": "(dataset)", "ok": False, "problems": ["no audio clips yet"]}
        )
    return report


def assert_trainable(dataset_dir: Path) -> None:
    """train_runs calls this: a folder we manage must validate clean before
    hours of GPU get burned on it."""
    manifest = _read_manifest(dataset_dir)
    if manifest is None:
        return  # not one of ours — the S0 light check still applies
    report = _validate_dir(Path(dataset_dir), manifest)
    if not report["ok"]:
        bad = [row for row in report["rows"] if not row["ok"]]
        first = bad[0]["problems"][0] if bad and bad[0]["problems"] else "see report"
        raise RuntimeError(
            f"Dataset '{manifest.get('name')}' is not ready to train: "
            f"{len(bad)} of {max(report['checked'], len(bad))} entries have "
            f"problems (first: {first}). Run validate on the Datasets page "
            "for the full report."
        )


def _free_name(dest: Path, name: str) -> Path:
    target = dest / name
    if not target.exists():
        return target
    stem, suffix = Path(name).stem, Path(name).suffix
    index = 2
    while (dest / f"{stem}-{index}{suffix}").exists():
        index += 1
    return dest / f"{stem}-{index}{suffix}"


def _read_manifest(folder: Path) -> dict[str, Any] | None:
    path = folder / "dataset.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_manifest(folder: Path, manifest: dict[str, Any]) -> None:
    (folder / "dataset.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
