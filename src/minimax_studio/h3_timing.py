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
