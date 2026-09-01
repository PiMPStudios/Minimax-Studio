"""Apple Silicon Music 3 depends on mlx-audio being installed.

PLAN.md promised Mac launchers would pull MLX extras. Until this extra exists,
Generate Music on Apple Silicon fails at import time even when the MXFP8 pack
is on disk. These tests keep the extra, the floor, and the Darwin-only install
from drifting.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_mlx_extra_pins_mlx_audio_at_music_module_floor() -> None:
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "mlx" in extras, "pyproject.toml must declare an [mlx] extra"
    deps = extras["mlx"]
    match = next(
        (re.fullmatch(r"mlx-audio>=(\d+)\.(\d+)\.(\d+)", item) for item in deps),
        None,
    )
    assert match is not None, f"[mlx] extra must pin mlx-audio>=x.y.z, got {deps}"
    version = tuple(int(part) for part in match.groups())
    assert version >= (0, 5, 0), (
        "mlx-audio 0.4.x has no mlx_audio.music; floor must be >=0.5.0 "
        f"(got {'.'.join(str(part) for part in version)})"
    )


def test_run_sh_installs_mlx_extra_only_on_apple_silicon() -> None:
    text = (ROOT / "scripts" / "run.sh").read_text(encoding="utf-8")
    assert "uname -s" in text and "Darwin" in text
    assert "uname -m" in text and "arm64" in text
    assert "dev,mlx" in text
    # Linux/Windows must keep the CUDA-free default extra.
    assert 'pip install -e ".[dev]"' in text or 'extras="dev"' in text


def test_windows_launcher_and_ci_do_not_pull_mlx() -> None:
    bat = (ROOT / "scripts" / "run.bat").read_text(encoding="utf-8")
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "mlx" not in bat.lower()
    assert "[mlx]" not in ci
    assert '".[dev]"' in ci
