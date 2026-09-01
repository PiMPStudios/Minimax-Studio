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
        self._http = httpx.Client(timeout=timeout, headers=self._headers)

    def close(self) -> None:
        self._http.close()

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

    def trim_history(
        self, entry_id: str, start_s: float, end_s: float
    ) -> dict[str, Any]:
        return self._post(
            f"/history/{_seg(entry_id)}/trim",
            {"start_s": float(start_s), "end_s": float(end_s)},
            timeout=120.0,
        )

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

    def import_lora(self, path: str, kind: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"path": path}
        if kind:
            payload["kind"] = kind
        return self._post("/loras/import", payload)

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

    def set_dataset_target_mode(self, dataset_id: str, mode: str) -> dict[str, Any]:
        # "av" is refused with the clip names when the set cannot carry it.
        return self._post(
            f"/datasets/{_seg(dataset_id)}/target-mode", {"mode": mode}, timeout=1800.0
        )

    def train_preflight(
        self, preset: str = "24g", dataset_dir: str | None = None
    ) -> dict[str, Any]:
        # Reads nvidia-smi on the worker — allow for the subprocess. Passing the
        # dataset is what lets the worker catch a Music preset aimed at an H3 set.
        query = f"?preset={_seg(preset)}"
        if dataset_dir:
            query += f"&dataset_dir={quote(str(dataset_dir))}"
        return self._get(f"/train/preflight{query}", timeout=90.0)

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

    # --- Long-run hardening (PLAN-V2 S5) ------------------------------------
    #
    # Storage calls are opened from a dialog, never polled: measuring a VAE cache
    # means walking thousands of files, and the worker caches what it measured.

    def train_storage(self) -> dict[str, Any]:
        return self._get("/train/storage", timeout=120.0)

    def train_run_storage(self, run_id: str) -> dict[str, Any]:
        return self._get(f"/train/runs/{_seg(run_id)}/storage", timeout=120.0)

    def resume_train_run(
        self,
        run_id: str,
        checkpoint: str | None = None,
        steps: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"checkpoint": checkpoint}
        if steps is not None:
            payload["steps"] = int(steps)
        return self._post(
            f"/train/runs/{_seg(run_id)}/resume", payload, timeout=120.0
        )

    def clear_train_cache(self, run_id: str) -> dict[str, Any]:
        return self._post(f"/train/runs/{_seg(run_id)}/cache/clear", {}, timeout=120.0)

    def prune_train_checkpoints(
        self, run_id: str, keep: int = 3, dry_run: bool = False
    ) -> dict[str, Any]:
        # dry_run answers “how much would this free?” with the same code that
        # then frees it — so the number in the confirmation is the real one.
        return self._post(
            f"/train/runs/{_seg(run_id)}/prune",
            {"keep": int(keep), "dry_run": bool(dry_run)},
            timeout=300.0,
        )

    def export_train_run(
        self, run_id: str, dest: str, include_cache: bool = False
    ) -> dict[str, Any]:
        # Copying tens of GB is a minute-scale request; nothing else here waits this long.
        return self._post(
            f"/train/runs/{_seg(run_id)}/export",
            {"dest": dest, "include_cache": bool(include_cache)},
            timeout=3600.0,
        )

    def import_train_run(self, folder: str) -> dict[str, Any]:
        return self._post("/train/runs/import", {"folder": folder}, timeout=3600.0)

    def delete_train_run(self, run_id: str) -> dict[str, Any]:
        return self._delete(f"/train/runs/{_seg(run_id)}")

    # --- Adapters (PLAN-V2 S3) ----------------------------------------------

    def list_adapters(self) -> list[dict[str, Any]]:
        return self._get_list("/adapters")

    def list_adapter_catalog(self) -> list[dict[str, Any]]:
        return self._get_list("/adapters/catalog")

    def audition_adapter(
        self,
        adapter_id: str,
        prompt: str = "",
        duration_s: float | None = None,
        backend: str = "auto",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"prompt": prompt, "backend": backend}
        if duration_s:
            payload["duration_s"] = float(duration_s)
        # A real generate job: the worker resolves the backend and validates.
        return self._post(f"/adapters/{_seg(adapter_id)}/audition", payload, timeout=60.0)

    def forget_adapter(self, adapter_id: str) -> dict[str, Any]:
        return self._delete(f"/adapters/{_seg(adapter_id)}")

    def _get(self, path: str, timeout: float | None = None) -> dict[str, Any]:
        response = self._http.get(
            f"{self._base}{path}", timeout=timeout or self._timeout
        )
        self._raise(response)
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"unexpected payload from {path}")
        return data

    def _get_list(self, path: str) -> list[dict[str, Any]]:
        response = self._http.get(f"{self._base}{path}", timeout=self._timeout)
        self._raise(response)
        data = response.json()
        if not isinstance(data, list):
            raise RuntimeError(f"unexpected list payload from {path}")
        return data

    def _post(
        self, path: str, payload: dict[str, Any], timeout: float | None = None
    ) -> dict[str, Any]:
        response = self._http.post(
            f"{self._base}{path}", json=payload, timeout=timeout or self._timeout
        )
        self._raise(response)
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"unexpected payload from {path}")
        return data

    def _delete(self, path: str) -> dict[str, Any]:
        response = self._http.delete(f"{self._base}{path}", timeout=self._timeout)
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
