"""PLAN-V3 S0: ffmpeg trim helper — stub binary, no GPU."""

from __future__ import annotations

from pathlib import Path

import pytest

from minimax_studio.worker import trim as trim_mod

FFMPEG_STUB = """
import pathlib, sys
out = pathlib.Path(sys.argv[-1])
out.write_bytes(b"trimmed")
out.with_suffix(out.suffix + ".argv").write_text("\\n".join(sys.argv), encoding="utf-8")
"""


@pytest.fixture
def ffmpeg_stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    script = tmp_path / "ffmpeg_stub.py"
    script.write_text(FFMPEG_STUB, encoding="utf-8")
    import sys

    monkeypatch.setenv(
        "MINIMAX_STUDIO_FFMPEG_BIN",
        f'"{sys.executable}" "{script}"',
    )
    return script


def test_missing_ffmpeg_is_a_named_install(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MINIMAX_STUDIO_FFMPEG_BIN", raising=False)
    monkeypatch.setattr(trim_mod.shutil, "which", lambda _name: None)
    src = tmp_path / "song.wav"
    src.write_bytes(b"RIFF")
    with pytest.raises(RuntimeError, match="install ffmpeg to trim"):
        trim_mod.trim_media(src, tmp_path / "cut.wav", 0, 1)


def test_missing_source_is_named(ffmpeg_stub, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not a file"):
        trim_mod.trim_media(tmp_path / "gone.wav", tmp_path / "cut.wav", 0, 1)


def test_end_must_follow_start(ffmpeg_stub, tmp_path: Path) -> None:
    src = tmp_path / "song.wav"
    src.write_bytes(b"RIFF")
    with pytest.raises(RuntimeError, match="must be after start"):
        trim_mod.trim_media(src, tmp_path / "cut.wav", 2, 2)
    with pytest.raises(RuntimeError, match="before 0"):
        trim_mod.trim_media(src, tmp_path / "cut.wav", -0.1, 1)


def test_trim_records_in_out_and_stream_copy(
    ffmpeg_stub, tmp_path: Path
) -> None:
    src = tmp_path / "song.wav"
    src.write_bytes(b"RIFF")
    dest = tmp_path / "cut.wav"
    result = trim_mod.trim_media(src, dest, 1.5, 4.0)
    assert dest.read_bytes() == b"trimmed"
    assert result["start_s"] == 1.5
    assert result["end_s"] == 4.0
    assert result["duration_s"] == 2.5
    argv = dest.with_suffix(".wav.argv").read_text(encoding="utf-8")
    assert str(src) in argv
    assert "-ss" in argv and "1.500000" in argv
    assert "-to" in argv and "4.000000" in argv
    assert "-c" in argv and "copy" in argv
    assert "-i" in argv


def test_video_in_out_snap_to_24fps(ffmpeg_stub, tmp_path: Path) -> None:
    src = tmp_path / "take.mp4"
    src.write_bytes(b"ftyp")
    dest = tmp_path / "cut.mp4"
    # 1.02s → 24.48 frames → 24/24 = 1.0; 2.03s → 48.72 → 49/24.
    result = trim_mod.trim_media(src, dest, 1.02, 2.03)
    assert result["start_s"] == 1.0
    assert result["end_s"] == pytest.approx(49 / 24)
    argv = dest.with_suffix(".mp4.argv").read_text(encoding="utf-8")
    assert "1.000000" in argv


@pytest.mark.skipif(
    trim_mod.shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)
def test_real_ffmpeg_cuts_a_wav(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import wave

    monkeypatch.delenv("MINIMAX_STUDIO_FFMPEG_BIN", raising=False)
    src = tmp_path / "song.wav"
    with wave.open(str(src), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x00" * 8000 * 3)
    dest = tmp_path / "cut.wav"
    result = trim_mod.trim_media(src, dest, 0.5, 1.5)
    assert dest.is_file() and dest.stat().st_size > 0
    assert result["duration_s"] == pytest.approx(1.0)
