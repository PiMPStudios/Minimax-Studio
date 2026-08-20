"""Open a file or folder in the OS file manager."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path


def reveal_command(path: str | Path) -> list[str]:
    target = Path(path)
    system = platform.system()
    if system == "Windows":
        if target.is_dir():
            return ["explorer", str(target)]
        return ["explorer", "/select,", str(target)]
    if system == "Darwin":
        if target.is_dir():
            return ["open", str(target)]
        return ["open", "-R", str(target)]
    folder = target if target.is_dir() else target.parent
    return ["xdg-open", str(folder)]


def reveal_path(path: str | Path) -> None:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(str(target))
    cmd = reveal_command(target)
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
