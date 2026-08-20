from __future__ import annotations

from typing import Any

import httpx


class WorkerClient:
    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def probe(self) -> dict[str, Any]:
        return self._get("/probe")

    def get_settings(self) -> dict[str, Any]:
        return self._get("/settings")

    def put_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/settings", payload)

    def list_packs(self) -> list[dict[str, Any]]:
        return self._get_list("/packs")

    def start_download(self, pack_id: str) -> dict[str, Any]:
        return self._post("/downloads", {"pack_id": pack_id})

    def list_downloads(self) -> list[dict[str, Any]]:
        return self._get_list("/downloads")

    def get_download(self, job_id: str) -> dict[str, Any]:
        return self._get(f"/downloads/{job_id}")

    def start_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/jobs", payload)

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self._get(f"/jobs/{job_id}")

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        return self._post(f"/jobs/{job_id}/cancel", {})

    def delete_history(self, entry_id: str) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout) as client:
            response = client.delete(f"{self._base}/history/{entry_id}")
            self._raise(response)
            data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError("unexpected delete payload")
            return data

    def write_lyrics(self, prompt: str, notes: str = "") -> dict[str, Any]:
        return self._post("/lyrics", {"prompt": prompt, "notes": notes}, timeout=180.0)

    def list_jobs(self) -> list[dict[str, Any]]:
        return self._get_list("/jobs")

    def list_history(self) -> list[dict[str, Any]]:
        return self._get_list("/history")

    def list_presets(self) -> list[dict[str, Any]]:
        return self._get_list("/presets")

    def save_preset(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/presets", payload)

    def list_loras(self) -> list[dict[str, Any]]:
        return self._get_list("/loras")

    def import_lora(self, path: str) -> dict[str, Any]:
        return self._post("/loras/import", {"path": path})

    def enhance(self, kind: str, text: str, extra: str = "") -> dict[str, Any]:
        return self._post(
            "/enhance",
            {"kind": kind, "text": text, "extra": extra},
            timeout=180.0,
        )

    def delete_preset(self, preset_id: str) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout) as client:
            response = client.delete(f"{self._base}/presets/{preset_id}")
            self._raise(response)
            data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError("unexpected delete payload")
            return data

    def _get(self, path: str) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(f"{self._base}{path}")
            self._raise(response)
            data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError(f"unexpected payload from {path}")
            return data

    def _get_list(self, path: str) -> list[dict[str, Any]]:
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(f"{self._base}{path}")
            self._raise(response)
            data = response.json()
            if not isinstance(data, list):
                raise RuntimeError(f"unexpected list payload from {path}")
            return data

    def _post(
        self, path: str, payload: dict[str, Any], timeout: float | None = None
    ) -> dict[str, Any]:
        with httpx.Client(timeout=timeout or self._timeout) as client:
            response = client.post(f"{self._base}{path}", json=payload)
            self._raise(response)
            data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError(f"unexpected payload from {path}")
            return data

    @staticmethod
    def _raise(response: httpx.Response) -> None:
        if response.is_success:
            return
        detail = None
        try:
            detail = response.json().get("detail")
        except Exception:
            detail = response.text
        raise RuntimeError(detail or f"HTTP {response.status_code}")
