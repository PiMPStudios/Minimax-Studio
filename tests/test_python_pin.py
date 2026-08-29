"""The one-Python contract.

0.2.28 shipped a ``[train]`` extra pinned to a SimpleTuner build that does not
exist for the venv it was written in (3.14) — nothing checked, so the pin was
never installable anywhere. The contract is one Python, named in five places;
these tests are what keeps those places agreeing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from minimax_studio.app import SUPPORTED_PYTHON, python_problem

ROOT = Path(__file__).resolve().parents[1]


def _wanted() -> tuple[int, int]:
    text = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    major, minor = text.split(".")[:2]
    return int(major), int(minor)


def _pin() -> str:
    return ".".join(str(part) for part in SUPPORTED_PYTHON)


def test_we_are_running_on_the_pinned_python() -> None:
    assert sys.version_info[:2] == SUPPORTED_PYTHON, (
        f"Tests must run in the pinned venv (Python {_pin()}). "
        "Re-run scripts/run.sh, which rebuilds .venv for you."
    )


def test_python_version_file_and_startup_guard_agree() -> None:
    assert _wanted() == SUPPORTED_PYTHON


def test_pyproject_requires_python_spans_exactly_one_minor() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'requires-python = "([^"]+)"', text)
    assert match, "requires-python missing from pyproject.toml"
    major, minor = SUPPORTED_PYTHON
    assert match.group(1) == f">={major}.{minor},<{major}.{minor + 1}"


def test_ci_matrix_installs_only_the_pinned_python() -> None:
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if "python:" in line]
    assert lines == [f'python: ["{_pin()}"]'], f"CI matrix drifted: {lines}"


@pytest.mark.parametrize("script", ["run.sh", "run.bat"])
def test_launchers_read_the_pin_instead_of_hardcoding(script: str) -> None:
    text = (ROOT / "scripts" / script).read_text(encoding="utf-8")
    assert ".python-version" in text, f"{script} must resolve the pin, not guess"


@pytest.mark.parametrize("version", [(3, 11), (3, 13), (3, 14), (4, 0)])
def test_startup_guard_refuses_other_pythons_by_naming_the_fix(
    version: tuple[int, int],
) -> None:
    problem = python_problem(version)
    assert problem is not None
    assert _pin() in problem
    assert ".venv" in problem


def test_startup_guard_accepts_the_pinned_python() -> None:
    assert python_problem(SUPPORTED_PYTHON) is None
