"""Hardware probe that stays safe without torch installed."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
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
        "cuda_count": 0,
        "gpus": [],
        "ram_gb": _ram_gb(),
        "apple_silicon": platform.system() == "Darwin"
        and platform.machine().lower() in {"arm64", "aarch64"},
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "torch_available": False,
        "sageattention": False,
        "cuda_source": None,
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
            count = int(torch.cuda.device_count())
            gpus = []
            for index in range(count):
                props = torch.cuda.get_device_properties(index)
                gpus.append(
                    {
                        "name": torch.cuda.get_device_name(index),
                        "vram_gb": round(props.total_memory / (1024**3), 1),
                    }
                )
            _apply_gpus(info, gpus, source="torch")
    except Exception:
        pass
    if not info["cuda"]:
        smi = _nvidia_smi_gpus()
        if smi:
            _apply_gpus(info, smi, source="nvidia-smi")
    return info


def _apply_gpus(info: dict[str, Any], gpus: list[dict[str, Any]], source: str) -> None:
    info["cuda"] = True
    info["gpus"] = gpus
    info["cuda_count"] = len(gpus)
    info["cuda_name"] = gpus[0]["name"] if len(gpus) == 1 else " + ".join(item["name"] for item in gpus)
    info["vram_gb"] = gpus[0]["vram_gb"] if len(gpus) == 1 else max(item["vram_gb"] for item in gpus)
    info["cuda_source"] = source


def _nvidia_smi_gpus() -> list[dict[str, Any]]:
    binary = shutil.which("nvidia-smi")
    if not binary:
        return []
    try:
        raw = subprocess.check_output(
            [
                binary,
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=3,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    gpus: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if "," not in line:
            continue
        name, _, mem = line.partition(",")
        name = name.strip()
        try:
            vram_gb = round(float(mem.strip()) / 1024.0, 1)
        except ValueError:
            continue
        if name:
            gpus.append({"name": name, "vram_gb": vram_gb})
    return gpus


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
