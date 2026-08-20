from __future__ import annotations

from typing import Any

import httpx


class WorkerClient:
    def __init__(self, base_url: str, timeout: float = 5.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def probe(self) -> dict[str, Any]:
        return self._get("/probe")

    def _get(self, path: str) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(f"{self._base}{path}")
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError(f"unexpected worker payload from {path}")
            return data
