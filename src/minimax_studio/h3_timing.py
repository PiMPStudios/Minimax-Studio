"""H3 frame-grid helpers. Safe to import from the GUI (no torch)."""

from __future__ import annotations


def duration_to_frames(seconds: float) -> int:
    """Snap duration to H3's 17n+5 frame grid at 24 fps, clamped to 5–15 s."""
    raw = max(5.0, min(15.0, float(seconds)))
    target = raw * 24.0
    n = max(0, round((target - 5) / 17))
    frames = 17 * n + 5
    while frames / 24.0 < 5.0:
        n += 1
        frames = 17 * n + 5
    while frames / 24.0 > 15.0 and n > 0:
        n -= 1
        frames = 17 * n + 5
    return frames


def frames_to_seconds(frames: int) -> float:
    return int(frames) / 24.0


def format_h3_duration(seconds: float) -> str:
    frames = duration_to_frames(seconds)
    snapped = frames_to_seconds(frames)
    return f"{snapped:.1f}s / {frames}f"


def resolve_dims(
    resolution: str,
    ratio: str,
    width: int | None = None,
    height: int | None = None,
    quality: str = "native",
) -> tuple[int, int]:
    """H3 canvas: Preview ~480 short edge, Native 768, API 2K ~1088, multiple of 32."""
    if width and height and (int(width), int(height)) != (960, 544):
        return int(width), int(height)
    res = str(resolution or "").upper()
    if res.startswith("2"):
        short = 1088
    elif str(quality or "native").lower() == "preview":
        short = 480
    else:
        short = 768
    parts = str(ratio or "16:9").split(":")
    try:
        rw, rh = int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        rw, rh = 16, 9
    if rw >= rh:
        h = short
        w = int(round(short * rw / rh / 32) * 32)
    else:
        w = short
        h = int(round(short * rh / rw / 32) * 32)
    return max(32, w), max(32, h)
