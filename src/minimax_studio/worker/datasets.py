"""Training datasets (PLAN-V2 S1, video half in S4).

A dataset is a plain folder in the layout SimpleTuner reads natively —
``track.wav`` + ``track.txt`` caption + optional ``track.lyrics`` (music), or
``shot.png`` / ``shot.mp4`` + ``shot.txt`` (H3) — plus a ``dataset.json``
manifest of ours for name/kind/provenance. The trainer never touches the
manifest; the app uses it to validate before anyone burns GPU hours, per the
standing rule: named numbers, no mystery failures.

WAV duration is probed with the stdlib ``wave`` module on purpose: it needs no
ffmpeg, so CI can generate clips and test the validator honestly. Everything
else — mp3, mp4, png — is measured with ``ffprobe`` when it exists, and
reported as a **warning** when it does not: "could not check the duration" is
not the same fact as "the duration is wrong", and the validator must not
pretend otherwise.
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
CLIP_EXTS = (".mp4", ".mov", ".webm")
STILL_EXTS = (".png", ".jpg", ".jpeg", ".webp")
MEDIA_BY_KIND = {
    "music": (".wav", ".flac", ".mp3"),
    # H3 trains on stills first, short clips after (PLAN-V2 S4). One "video"
    # kind holds both: SimpleTuner's H3 backend buckets images and clips
    # together, and making someone create a second kind to add a poster frame
    # would be a taxonomy question, not a dataset one.
    "video": CLIP_EXTS + STILL_EXTS,
}
MIN_SECONDS = 3.0
MAX_SECONDS = 300.0

# "Stills/short clips only": clips with dialogue wait for proof they do not
# wreck the audio heads, so the cap is a product decision and not a crash.
MAX_VIDEO_SECONDS = 8.0
# Below this, a still is not a training target — it is a thumbnail.
MIN_EDGE = 256

#: SimpleTuner's H3 target modes. ``av`` (audio+video) costs extra VRAM and
#: disk and is a checkbox, never the default; the exact key names are part of
#: the pinned contract and get verified against real SimpleTuner in the metal
#: session, like ``STEP_RE`` before them.
H3_TARGET_MODES = ("video", "av")


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
    if kind == "video":
        # Explicit rather than implied: the picker shows the mode, and "what did
        # this train with?" is answered by the manifest, not by memory.
        manifest["h3_target_mode"] = "video"
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


def ffprobe_command() -> list[str] | None:
    """``MINIMAX_STUDIO_FFPROBE_BIN`` is a test seam with the same precedent as
    the SimpleTuner override (a whole command, shell-quoted): the stub answers
    with real ffprobe-shaped JSON, so the parsing and the decisions it drives are
    both under test without ffmpeg on PATH."""
    override = os.environ.get("MINIMAX_STUDIO_FFPROBE_BIN")
    if override:
        import shlex

        return shlex.split(override)
    found = shutil.which("ffprobe")
    return [found] if found else None


def probe_audio(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
        if rate <= 0:
            raise RuntimeError("unreadable wav (zero sample rate)")
        return {"seconds": frames / float(rate), "format": "wav"}
    # Other formats want ffprobe; that path is only honest if ffmpeg exists.
    ffprobe = ffprobe_command()
    if not ffprobe:
        raise RuntimeError("install ffmpeg/ffprobe to probe " + path.suffix)
    import subprocess

    proc = subprocess.run(
        [
            *ffprobe,
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


def probe_video(path: Path) -> dict[str, Any]:
    """Measure a still or a clip: pixel size, duration, and whether it carries
    an audio stream — the three numbers that decide whether an H3 dataset can
    train, and in which target mode.

    Raises RuntimeError when nothing can measure it; the validator turns that
    into a warning, never a false accusation against the user's clips.
    """
    ffprobe = ffprobe_command()
    if not ffprobe:
        raise RuntimeError("install ffmpeg/ffprobe to measure " + path.suffix)
    import json as _json
    import subprocess

    proc = subprocess.run(
        [
            *ffprobe,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height,duration",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    try:
        payload = _json.loads(proc.stdout or "{}")
    except _json.JSONDecodeError as exc:
        raise RuntimeError(f"unreadable media ({path.suffix})") from exc
    streams = payload.get("streams") or []
    video = next((row for row in streams if row.get("codec_type") == "video"), None)
    if video is None and path.suffix.lower() in CLIP_EXTS:
        raise RuntimeError(f"no video stream in {path.name}")
    seconds: float | None = None
    for source in (video or {}), (payload.get("format") or {}):
        try:
            seconds = float(source.get("duration"))
            break
        except (TypeError, ValueError):
            continue
    return {
        "seconds": seconds,
        "width": int((video or {}).get("width") or 0) or None,
        "height": int((video or {}).get("height") or 0) or None,
        "has_audio": any(row.get("codec_type") == "audio" for row in streams),
        "format": path.suffix.lstrip("."),
    }


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
    """One report shape for both kinds: ``rows`` per file, ``warnings`` for what
    could not be checked, ``ok`` for whether to hand it to the trainer."""
    if manifest.get("kind") == "video":
        return _validate_video_dir(folder, manifest)
    return _validate_music_dir(folder, manifest)


def _caption_rules(path: Path, problems: list[str]) -> None:
    """What every entry has in common, whatever its kind: a caption beside it."""
    if not path.with_suffix(".txt").is_file():
        problems.append(f"missing caption {path.stem}.txt")


def _orphan_captions(folder: Path, stems: set[str], rows: list[dict[str, Any]]) -> None:
    for path in sorted(folder.glob("*.txt")):
        if path.stem not in stems:
            rows.append(
                {
                    "file": path.name,
                    "ok": False,
                    "problems": ["caption with no matching media file"],
                }
            )


def _validate_music_dir(folder: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "at": time.time(),
        "ok": False,
        "kind": "music",
        "checked": 0,
        "rows": [],
        "warnings": [],
    }
    rows: list[dict[str, Any]] = report["rows"]
    media_exts = set(MEDIA_BY_KIND["music"])
    stems_seen: set[str] = set()
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in media_exts or path.name.startswith("."):
            continue
        report["checked"] += 1
        stems_seen.add(path.stem)
        problems: list[str] = []
        _caption_rules(path, problems)
        if path.with_suffix(".lyrics").is_file() and not path.with_suffix(
            ".txt"
        ).is_file():
            problems.append(".lyrics without a matching .txt caption")
        seconds = None
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
        rows.append(
            {
                "file": path.name,
                "ok": not problems,
                "problems": problems,
                "seconds": seconds,
            }
        )
    _orphan_captions(folder, stems_seen, rows)
    report["ok"] = report["checked"] > 0 and all(row["ok"] for row in rows)
    if report["checked"] == 0:
        rows.append(
            {"file": "(dataset)", "ok": False, "problems": ["no audio clips yet"]}
        )
    return report


def _validate_video_dir(folder: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """The H3 validator (PLAN-V2 S4): stills and short clips, captions, pixel
    floor, clip-length cap, and whether the set can carry ``av`` mode.

    Anything ffprobe could not look at is a **warning** on the report, not a
    problem on the file: "no ffmpeg here, so durations and sizes were not
    checked" is a different fact from "your clip is too long", and a validator
    that blurs the two teaches people to ignore it.
    """
    mode = str(manifest.get("h3_target_mode") or "video")
    report: dict[str, Any] = {
        "at": time.time(),
        "ok": False,
        "kind": "video",
        "checked": 0,
        "rows": [],
        "warnings": [],
        "target_mode": mode,
        "stills": 0,
        "clips": 0,
        "with_audio": 0,
        "av_ready": False,
    }
    rows: list[dict[str, Any]] = report["rows"]
    media_exts = set(MEDIA_BY_KIND["video"])
    stems_seen: set[str] = set()
    unmeasured = 0
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in media_exts or path.name.startswith("."):
            continue
        report["checked"] += 1
        stems_seen.add(path.stem)
        is_still = path.suffix.lower() in STILL_EXTS
        report["stills" if is_still else "clips"] += 1
        problems: list[str] = []
        _caption_rules(path, problems)
        entry: dict[str, Any] = {
            "file": path.name,
            "entry_kind": "still" if is_still else "clip",
            "seconds": None,
            "width": None,
            "height": None,
            "has_audio": False,
        }
        try:
            info = probe_video(path)
        except (RuntimeError, OSError) as exc:
            unmeasured += 1
            report["warnings"].append(f"{path.name}: {exc}")
        else:
            width, height = info.get("width"), info.get("height")
            entry["width"], entry["height"] = width, height
            entry["seconds"] = info.get("seconds")
            entry["has_audio"] = bool(info.get("has_audio"))
            if width and height and min(width, height) < MIN_EDGE:
                problems.append(
                    f"{width}×{height} is under the {MIN_EDGE} px short-edge "
                    "floor — a thumbnail cannot teach a frame"
                )
            if not is_still:
                if info.get("has_audio"):
                    report["with_audio"] += 1
                elif mode == "av":
                    problems.append(
                        "av mode trains audio+video and this clip has no audio "
                        "stream — put the dataset back to Video mode or fix the clip"
                    )
                seconds = info.get("seconds")
                if seconds is None:
                    report["warnings"].append(f"{path.name}: no duration reported")
                elif seconds > MAX_VIDEO_SECONDS:
                    problems.append(
                        f"{seconds:.1f}s is over the {MAX_VIDEO_SECONDS:.0f}s cap "
                        "— clips with dialogue wait for proof they do not wreck "
                        "the audio heads"
                    )
        rows.append({**entry, "ok": not problems, "problems": problems})
    _orphan_captions(folder, stems_seen, rows)

    if mode == "av" and report["stills"]:
        rows.append(
            {
                "file": "(dataset)",
                "ok": False,
                "problems": [
                    f"av mode trains audio+video: a still has no audio at all. "
                    f"Move the {report['stills']} still(s) out or switch the "
                    "dataset back to Video mode."
                ],
            }
        )
    if unmeasured:
        report["warnings"].append(
            f"{unmeasured} of {report['checked']} file(s) could not be measured "
            "(install ffmpeg/ffprobe) — captions were checked, pixel size and "
            "duration were not."
        )
    report["av_ready"] = bool(
        report["clips"]
        and not report["stills"]
        and report["with_audio"] == report["clips"]
        and not unmeasured
    )
    report["ok"] = report["checked"] > 0 and all(row["ok"] for row in rows)
    if report["checked"] == 0:
        rows.append(
            {
                "file": "(dataset)",
                "ok": False,
                "problems": ["no stills or clips yet — add .png/.jpg stills or short .mp4 clips"],
            }
        )
    return report


def set_h3_target_mode(dataset_id: str, mode: str) -> dict[str, Any]:
    """Choose the H3 target mode: ``video`` (default) or ``av``.

    ``av`` is a checkbox and never a default — it pays VRAM and disk for an
    audio stream the run may not need, and it only makes sense when *every*
    clip has one and there are no stills in the set. Refusal names the clips.
    """
    if mode not in H3_TARGET_MODES:
        raise RuntimeError(
            f"Unknown H3 target mode '{mode}' (known: {', '.join(H3_TARGET_MODES)})."
        )
    folder, manifest = get_dataset(dataset_id)
    if manifest.get("kind") != "video":
        raise RuntimeError(
            f"Dataset “{manifest.get('name')}” is a “{manifest.get('kind')}” "
            "dataset — only a Video (H3) dataset has a target mode."
        )
    if mode == "av":
        report = _validate_video_dir(folder, manifest)
        if not report["av_ready"]:
            silent = [
                row["file"]
                for row in report["rows"]
                if row.get("entry_kind") == "clip" and not row.get("has_audio")
            ]
            if report["stills"]:
                raise RuntimeError(
                    f"“{manifest.get('name')}” holds {report['stills']} still(s). "
                    "av mode trains audio+video, and a still has no audio track "
                    "to train — use Video mode for a set with stills in it."
                )
            raise RuntimeError(
                f"av mode needs an audio stream in every clip, and "
                f"{len(silent)} of {report['clips']} have none"
                + (f" (first: {silent[0]})" if silent else "")
                + ". Use Video mode, or re-export the clips with audio."
            )
    manifest = dict(manifest)
    manifest["h3_target_mode"] = mode
    _write_manifest(folder, manifest)
    return {**manifest, "path": str(folder)}


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
