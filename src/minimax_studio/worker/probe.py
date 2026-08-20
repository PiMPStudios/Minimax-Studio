"""Hardware probe that stays safe without torch installed."""

from __future__ import annotations

import os
import platform
import shutil
from typing import Any


def probe() -> dict[str, Any]:
    info: dict[str, Any] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cuda": False,
        "cuda_name": None,
        "vram_gb": None,
        "ram_gb": _ram_gb(),
        "apple_silicon": platform.system() == "Darwin"
        and platform.machine().lower() in {"arm64", "aarch64"},
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "torch_available": False,
        "sageattention": False,
    }
    try:
        import sageattention  # noqa: F401

        info["sageattention"] = True
    except Exception:
        pass
    try:
        import torch

        info["torch_available"] = True
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda"] = True
            info["cuda_name"] = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory
            info["vram_gb"] = round(total / (1024**3), 1)
    except Exception:
        pass
    return info


def _ram_gb() -> float | None:
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return round((page * pages) / (1024**3), 1)
    except (ValueError, OSError, AttributeError):
        pass
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return round(stat.ullTotalPhys / (1024**3), 1)
    except Exception:
        return None
