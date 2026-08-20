from __future__ import annotations

from typing import Any

import httpx

from minimax_studio.worker.runtime import runtime


def ping_services() -> dict[str, Any]:
    return {
        "minimax": _ping_minimax(),
        "llm": _ping_llm(),
        "comfy": _ping_comfy(),
    }


def _ping_minimax() -> dict[str, Any]:
    key = runtime.config.minimax_api_key
    if not key:
        return {"ok": False, "detail": "no key"}
    base = (runtime.config.minimax_api_base or "https://api.minimax.io").rstrip("/")
    try:
        response = httpx.get(
            f"{base}/v1/files",
            headers={"Authorization": f"Bearer {key}"},
            timeout=8.0,
        )
    except httpx.HTTPError as exc:
        return {"ok": False, "detail": str(exc)}
    if response.status_code in {401, 403}:
        return {"ok": False, "detail": f"auth failed ({response.status_code})"}
    return {"ok": True, "detail": f"HTTP {response.status_code}"}


def _ping_llm() -> dict[str, Any]:
    base = (runtime.config.llm_base_url or "http://127.0.0.1:8080/v1").rstrip("/")
    key = runtime.config.resolved_llm_key()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        response = httpx.get(f"{base}/models", headers=headers, timeout=8.0)
    except httpx.HTTPError as exc:
        return {"ok": False, "detail": str(exc)}
    if response.status_code in {401, 403}:
        return {"ok": False, "detail": f"auth failed ({response.status_code})"}
    if response.status_code >= 400:
        return {"ok": False, "detail": f"HTTP {response.status_code}"}
    return {"ok": True, "detail": f"HTTP {response.status_code}"}


def _ping_comfy() -> dict[str, Any]:
    base = (runtime.config.comfy_url or "http://127.0.0.1:8188").rstrip("/")
    last_error = "ComfyUI not reachable"
    for path in ("/system_stats", "/queue"):
        try:
            response = httpx.get(f"{base}{path}", timeout=3.0)
        except httpx.HTTPError as exc:
            last_error = str(exc)
            continue
        if response.status_code >= 400:
            last_error = f"HTTP {response.status_code}"
            continue
        return {"ok": True, "detail": f"HTTP {response.status_code} {path}"}
    return {"ok": False, "detail": last_error}
