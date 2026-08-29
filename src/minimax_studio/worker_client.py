from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx


def _seg(value: str) -> str:
    """One path segment, escaped. Ids are slugs today; keep it boring anyway."""
    return quote(str(value), safe="")


class WorkerClient:
    TOKEN_HEADER = "X-Minimax-Studio-Token"

    def __init__(
        self, base_url: str, timeout: float = 30.0, token: str | None = None
    ) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._headers = {}
        if token:
            self._headers[self.TOKEN_HEADER] = token

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def probe(self) -> dict[str, Any]:
        return self._get("/probe")

    def get_settings(self) -> dict[str, Any]:
        return self._get("/settings")

    def put_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/settings", payload)

    def ping(self) -> dict[str, Any]:
        return self._get("/ping")

    def comfy_status(self) -> dict[str, Any]:
        return self._get("/comfy")

    def start_comfy(self) -> dict[str, Any]:
        return self._post("/comfy/start", {})

    def preflight(
        self,
        kind: str,
        backend: str = "auto",
        mode: str = "t2va",
        speed: str = "quality",
        resolution: str = "768P",
    ) -> dict[str, Any]:
        from urllib.parse import urlencode

        query = urlencode(
            {
                "kind": kind,
                "backend": backend,
                "mode": mode,
                "speed": speed,
                "resolution": resolution,
            }
        )
        return self._get(f"/preflight?{query}")

    def list_packs(self) -> list[dict[str, Any]]:
        return self._get_list("/packs")

    def start_download(self, pack_id: str, force: bool = False) -> dict[str, Any]:
        return self._post("/downloads", {"pack_id": pack_id, "force": force})

    def delete_pack(self, pack_id: str, delete_shared: bool = False) -> dict[str, Any]:
        return self._delete(
            f"/packs/{_seg(pack_id)}?delete_shared={'true' if delete_shared else 'false'}"
        )

    def list_downloads(self) -> list[dict[str, Any]]:
        return self._get_list("/downloads")

    def get_download(self, job_id: str) -> dict[str, Any]:
        return self._get(f"/downloads/{job_id}")

    def cancel_download(self, job_id: str) -> dict[str, Any]:
        return self._post(f"/downloads/{job_id}/cancel", {})

    def start_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/jobs", payload)

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self._get(f"/jobs/{job_id}")

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        return self._post(f"/jobs/{job_id}/cancel", {})

    def delete_history(self, entry_id: str) -> dict[str, Any]:
        return self._delete(f"/history/{_seg(entry_id)}")

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

    def context_ir(
        self,
        prompt: str,
        mode: str = "t2va",
        duration_s: float = 8,
        ratio: str = "16:9",
        assets: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        return self._post(
            "/context-ir",
            {
                "prompt": prompt,
                "mode": mode,
                "duration_s": duration_s,
                "ratio": ratio,
                "assets": assets or [],
            },
            timeout=300.0,
        )

    def delete_preset(self, preset_id: str) -> dict[str, Any]:
        return self._delete(f"/presets/{_seg(preset_id)}")

    # --- Datasets (PLAN-V2 S1) + training (S0/S2) ---------------------------
    #
    # The Build pages. Anything that touches many files (import, validate) gets
    # its own timeout: the worker does real disk work, and a 30 s default would
    # give a 500-clip import a mystery failure.

    def list_datasets(self) -> list[dict[str, Any]]:
        return self._get_list("/datasets")

    def create_dataset(
        self, name: str, kind: str = "music", notes: str = ""
    ) -> dict[str, Any]:
        return self._post("/datasets", {"name": name, "kind": kind, "notes": notes})

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        return self._get(f"/datasets/{_seg(dataset_id)}")

    def delete_dataset(self, dataset_id: str) -> dict[str, Any]:
        return self._delete(f"/datasets/{_seg(dataset_id)}")

    def import_dataset_folder(self, dataset_id: str, folder: str) -> dict[str, Any]:
        return self._post(
            f"/datasets/{_seg(dataset_id)}/import",
            {"folder": folder},
            timeout=1800.0,
        )

    def add_dataset_from_history(self, dataset_id: str, history_id: str) -> dict[str, Any]:
        return self._post(
            f"/datasets/{_seg(dataset_id)}/from_history", {"history_id": history_id}
        )

    def validate_dataset(self, dataset_id: str) -> dict[str, Any]:
        return self._post(
            f"/datasets/{_seg(dataset_id)}/validate", {}, timeout=1800.0
        )

    def train_preflight(self, preset: str = "24g") -> dict[str, Any]:
        # Reads nvidia-smi on the worker — allow for the subprocess.
        return self._get(f"/train/preflight?preset={_seg(preset)}", timeout=90.0)

    def list_train_runs(self) -> list[dict[str, Any]]:
        return self._get_list("/train/runs")

    def get_train_run(self, run_id: str, tail: int = 60) -> dict[str, Any]:
        return self._get(f"/train/runs/{_seg(run_id)}?tail={int(tail)}")

    def start_train_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        # The worker runs preflight + a full dataset validation in here.
        return self._post("/train/runs", payload, timeout=300.0)

    def cancel_train_run(self, run_id: str) -> dict[str, Any]:
        return self._post(f"/train/runs/{_seg(run_id)}/cancel", {})

    def install_train_adapter(self, run_id: str, path: str | None = None) -> dict[str, Any]:
        query = f"?path={_seg(path)}" if path else ""
        return self._post(f"/train/runs/{_seg(run_id)}/install{query}", {})

    def _get(self, path: str, timeout: float | None = None) -> dict[str, Any]:
        with httpx.Client(timeout=timeout or self._timeout, headers=self._headers) as client:
            response = client.get(f"{self._base}{path}")
            self._raise(response)
            data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError(f"unexpected payload from {path}")
            return data

    def _get_list(self, path: str) -> list[dict[str, Any]]:
        with httpx.Client(timeout=self._timeout, headers=self._headers) as client:
            response = client.get(f"{self._base}{path}")
            self._raise(response)
            data = response.json()
            if not isinstance(data, list):
                raise RuntimeError(f"unexpected list payload from {path}")
            return data

    def _post(
        self, path: str, payload: dict[str, Any], timeout: float | None = None
    ) -> dict[str, Any]:
        with httpx.Client(timeout=timeout or self._timeout, headers=self._headers) as client:
            response = client.post(f"{self._base}{path}", json=payload)
            self._raise(response)
            data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError(f"unexpected payload from {path}")
            return data

    def _delete(self, path: str) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout, headers=self._headers) as client:
            response = client.delete(f"{self._base}{path}")
            self._raise(response)
            data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError(f"unexpected delete payload from {path}")
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
