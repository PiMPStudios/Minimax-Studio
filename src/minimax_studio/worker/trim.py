"""Cut a take with ffmpeg (PLAN-V3 S0).

S1 will hang a History row on this. The helper itself does not know History:
it takes two paths and two timestamps, refuses when ffmpeg is missing, and
never mutates the source.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv"}


def ffmpeg_command() -> list[str] | None:
    """``MINIMAX_STUDIO_FFMPEG_BIN`` is a test seam (whole command, shell-quoted),
    same shape as ``MINIMAX_STUDIO_FFPROBE_BIN``."""
    override = os.environ.get("MINIMAX_STUDIO_FFMPEG_BIN")
    if override:
        import shlex

        return shlex.split(override)
    found = shutil.which("ffmpeg")
    return [found] if found else None


def snap_video_seconds(seconds: float, fps: float = 24.0) -> float:
    """Nearest frame at ``fps``. Does not clamp to the H3 5–15 s generate grid —
    a two-second trim is still a valid take."""
    rate = float(fps) or 24.0
    return round(float(seconds) * rate) / rate


def trim_media(
    src: str | Path,
    dest: str | Path,
    start_s: float,
    end_s: float,
) -> dict[str, Any]:
    """Write ``dest`` as ``src`` cut to ``[start_s, end_s)``.

    Video in/out snap to 24 fps. Audio is left as given. Stream-copy (``-c copy``)
    so S0 does not pick an encoder; S1 metal listens for H3 audio sync.
    """
    source = Path(src)
    target = Path(dest)
    if not source.is_file():
        raise RuntimeError(f"nothing to trim — {source} is not a file")
    start = float(start_s)
    end = float(end_s)
    if source.suffix.lower() in VIDEO_EXTS:
        start = snap_video_seconds(start)
        end = snap_video_seconds(end)
    if start < 0:
        raise RuntimeError(f"trim start {start}s is before 0")
    if end <= start:
        raise RuntimeError(
            f"trim end {end}s must be after start {start}s — pick in/out points"
        )
    ffmpeg = ffmpeg_command()
    if not ffmpeg:
        raise RuntimeError(
            "install ffmpeg to trim — it is not on PATH (the same binary H3 "
            "mux uses)"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    # -ss/-to after -i: decode then cut. -c copy: no re-encode in S0.
    argv = [
        *ffmpeg,
        "-hide_banner",
        "-y",
        "-i",
        str(source),
        "-ss",
        f"{start:.6f}",
        "-to",
        f"{end:.6f}",
        "-c",
        "copy",
        str(target),
    ]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"ffmpeg could not start: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ffmpeg timed out while trimming") from exc
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"ffmpeg trim failed: {err.splitlines()[-1]}")
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError("ffmpeg trim finished but wrote no file")
    return {
        "src": str(source),
        "dest": str(target),
        "start_s": start,
        "end_s": end,
        "duration_s": end - start,
    }
