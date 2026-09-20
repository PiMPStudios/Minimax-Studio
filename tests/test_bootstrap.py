"""The environment bootstrap contract (0.2.67).

Studio still runs on exactly one Python — .python-version, unchanged, enforced
by pyproject, CI, and app.py's startup guard. What changed is who supplies it:
the launchers ask **uv** for the pinned interpreter (reusing a system 3.12 or
downloading a standalone build) so a clone of this repo never has to install
Python to run the app. These tests keep that machinery from rotting: the pin
must flow from .python-version, uv must stay discoverable and overridable, and
the escape hatch for people who refuse uv must keep working.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def script(name: str) -> str:
    return (ROOT / "scripts" / name).read_text(encoding="utf-8")


SH = "run.sh"
BAT = "run.bat"
BOTH = [SH, BAT]

# The installer each launcher prints when uv is missing. A launcher must never
# download an executable from the internet without being asked.
UV_INSTALL_HINT = {SH: "astral.sh/uv/install.sh", BAT: "astral.sh/uv/install.ps1"}
STALE_MARK = {SH: ".venv.pre-", BAT: ".venv.stale"}
DELETE_LINE = {SH: "rm -rf .venv\n", BAT: "rmdir /s /q .venv\n"}


@pytest.mark.parametrize("name", BOTH)
def test_launchers_ask_uv_for_the_interpreter_instead_of_demanding_it(name: str) -> None:
    text = script(name)
    assert "venv --seed --python" in text
    assert UV_INSTALL_HINT[name] in text, (
        "the fix for a missing uv has to be printed, not guessed"
    )


@pytest.mark.parametrize("name", BOTH)
def test_the_pinned_version_flows_from_python_version_not_a_literal(name: str) -> None:
    """.python-version stays the single source of truth (see test_python_pin.py)."""
    text = script(name)
    assert ".python-version" in text
    assert '--python "$want"' in text or "--python %want%" in text


@pytest.mark.parametrize("name", BOTH)
def test_uv_is_overridable_like_every_other_external_tool(name: str) -> None:
    """MINIMAX_STUDIO_UV_BIN follows the ffmpeg/simpletuner override pattern."""
    assert "MINIMAX_STUDIO_UV_BIN" in script(name)


@pytest.mark.parametrize("name", BOTH)
def test_the_python_override_escape_hatch_survives(name: str) -> None:
    """Offline shops and packagers still get to hand us their own interpreter."""
    assert "MINIMAX_STUDIO_PYTHON" in script(name)


@pytest.mark.parametrize("name", BOTH)
def test_a_foreign_venv_is_moved_aside_never_deleted(name: str) -> None:
    """0.2.28's lesson: a .venv on the wrong Python installs and then cannot train."""
    text = script(name)
    assert STALE_MARK[name] in text
    assert DELETE_LINE[name] not in text


def _bare_env(tmp_path: Path) -> dict[str, str]:
    """HOME (for uv's install dirs) but no PATH entry for uv, and XDG dirs in tmp
    so a real ~/.local/bin/uv on a dev box cannot leak into the answer."""
    env = {
        "HOME": str(tmp_path),
        "USERPROFILE": str(tmp_path),
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "XDG_DATA_HOME": str(tmp_path / "share"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
    }
    for name in ("SYSTEMROOT", "COMSPEC", "TMP", "TEMP"):
        if name in os.environ:
            env[name] = os.environ[name]
    return env


@pytest.mark.skipif(
    sys.platform == "win32", reason="run.sh is the POSIX launcher; run.bat has no CI here"
)
def test_print_runtime_still_answers_when_uv_is_missing(tmp_path: Path) -> None:
    """The one state every new cloner is in on their first run."""
    scrubbed = ("/usr/bin", "/bin", "/usr/local/bin")
    if any((Path(d) / "uv").exists() for d in scrubbed):
        pytest.skip(f"uv is installed in {scrubbed}; the PATH scrub cannot prove absence")
    proc = subprocess.run(
        ["bash", str(ROOT / "scripts" / SH), "--print-runtime"],
        capture_output=True,
        text=True,
        timeout=120,
        env=_bare_env(tmp_path),
    )
    want = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    assert proc.returncode == 0, proc.stderr
    assert f"wants python   {want}" in proc.stdout
    for label in ("uv ", "override ", ".venv python "):
        assert label in proc.stdout
    assert "not found" in proc.stdout


@pytest.mark.skipif(
    sys.platform == "win32", reason="run.sh is the POSIX launcher; run.bat has no CI here"
)
def test_print_runtime_names_a_uv_found_via_the_override(tmp_path: Path) -> None:
    fake = tmp_path / "my-uv"
    fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)
    env = _bare_env(tmp_path)
    env["MINIMAX_STUDIO_UV_BIN"] = str(fake)
    proc = subprocess.run(
        ["bash", str(ROOT / "scripts" / SH), "--print-runtime"],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert str(fake) in proc.stdout


@pytest.mark.skipif(
    sys.platform == "win32", reason="run.sh is the POSIX launcher; run.bat has no CI here"
)
def test_print_runtime_never_builds_and_never_hides_a_wrong_venv() -> None:
    """Support copy-paste target: read-only, and the venv line tells the truth."""
    proc = subprocess.run(
        ["bash", str(ROOT / "scripts" / SH), "--print-runtime"],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "MINIMAX_STUDIO_PYTHON": ""},
    )
    want = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    assert proc.returncode == 0, proc.stderr
    # Either there is no .venv yet, or the one we have is the pinned one —
    # the report must never be the place a wrong interpreter hides.
    reported = proc.stdout.split(".venv python", 1)[1].strip().splitlines()[0]
    assert reported == "none" or reported.startswith(f"{want}."), reported
