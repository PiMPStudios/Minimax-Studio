"""PLAN-V2 S3: the adapter registry and the audition loop, stubbed end-to-end.

The loop being tested is the whole point of v2: train → install → hear it in
History, without hand-managing files. The GPU is not needed for that — the stub
music backend writes a tone, and what we check is what got *queued*: the
adapter at 0.8, the caption it learned from, and the audition badge.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from minimax_studio.worker import adapters
from minimax_studio.worker.server import app


def _clips(root: Path, captions: list[str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for index, caption in enumerate(captions):
        (root / f"clip{index}.wav").write_bytes(b"RIFF" * (index + 1))
        (root / f"clip{index}.txt").write_text(caption + "\n", encoding="utf-8")
    return root


def _lora_file(studio_home: Path, name: str = "summer-lora.safetensors") -> Path:
    folder = Path(studio_home) / "models" / "loras"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(b"\x00safetensors-stub")
    return path


def _trained_adapter(studio_home: Path, tmp_path: Path) -> dict:
    """A registry row like the one install_adapter writes after a real run."""
    dataset = _clips(tmp_path / "ds", ["moody folk", "moody folk", "synth pop"])
    source = tmp_path / "checkpoints" / "step-900" / "summer-lora.safetensors"
    source.parent.mkdir(parents=True, exist_ok=True)
    path = _lora_file(studio_home)
    run_state = {
        "id": "20260829-2015-summer",
        "name": "Summer",
        "dataset_dir": str(dataset),
        "preset": "24g",
        "steps": 900,
        "rank": 16,
    }
    return adapters.record_trained(
        run_state, {"name": "summer-lora", "path": str(path)}, source
    )


# --- registry ---------------------------------------------------------------


def test_registry_roundtrip_and_tolerant_load(studio_home: Path, tmp_path: Path) -> None:
    assert adapters.load_registry() == []
    adapters.record({"file": "a.safetensors", "name": "a", "source": "imported"})
    assert [row["file"] for row in adapters.load_registry()] == ["a.safetensors"]

    adapters.record({"file": "a.safetensors", "name": "a renamed"})
    rows = adapters.load_registry()
    assert len(rows) == 1, "the key is the file name — re-recording must not double up"
    assert rows[0]["name"] == "a renamed"
    assert rows[0]["source"] == "imported", "unset fields survive an update"
    first_created = rows[0]["created_at"]
    adapters.record({"file": "a.safetensors", "name": "again"})
    assert adapters.load_registry()[0]["created_at"] == first_created

    adapters.registry_path().write_text("{not json", encoding="utf-8")
    assert adapters.load_registry() == [], "a broken cache costs provenance, not the app"


def test_concurrent_records_do_not_drop_a_row(studio_home: Path) -> None:
    import threading

    errors: list[Exception] = []

    def write(name: str) -> None:
        try:
            adapters.record(
                {"file": f"{name}.safetensors", "name": name, "source": "imported"}
            )
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=write, args=(f"lora-{index}",)) for index in range(20)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    files = {row["file"] for row in adapters.load_registry()}
    assert files == {f"lora-{index}.safetensors" for index in range(20)}
    tmp = adapters.registry_path().with_name("adapters.json.tmp")
    assert not tmp.exists()


def test_trained_row_carries_the_provenance_a_filename_hides(
    studio_home: Path, tmp_path: Path
) -> None:
    row = _trained_adapter(studio_home, tmp_path)
    assert row["source"] == "trained"
    assert row["kind"] == "music"
    assert row["trainer"] == "simpletuner 4.8.0"
    assert row["base_pack"] == "music3-cuda"
    assert row["steps"] == 900 and row["rank"] == 16
    assert row["dataset"]["clip_count"] == 3
    assert len(row["dataset"]["manifest_hash"]) == 12
    assert row["name"] == "summer-lora", "one name for one file, on every page"


def test_an_h3_run_installs_as_an_h3_adapter(
    studio_home: Path, tmp_path: Path
) -> None:
    dataset = tmp_path / "stills"
    dataset.mkdir()
    (dataset / "a.png").write_bytes(b"\x00")
    (dataset / "a.txt").write_text("a frame\n", encoding="utf-8")
    source = tmp_path / "checkpoints" / "step-10" / "h3-lora.safetensors"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"\x00")
    path = _lora_file(studio_home, "h3-lora.safetensors")
    row = adapters.record_trained(
        {
            "id": "20260829-h3",
            "name": "H3 stills",
            "dataset_dir": str(dataset),
            "family": "h3",
            "dataset_kind": "video",
            "preset": "h3-24g",
            "steps": 200,
            "rank": 16,
        },
        {"name": "h3-lora", "path": str(path)},
        source,
    )
    assert row["kind"] == "h3"
    assert row["base_pack"] == "h3-diffusers-fl2va"
    listed = adapters.get_adapter(row["id"])
    assert listed["can_audition"] is True
    assert listed["dataset"]["clip_count"] == 1
    assert adapters.still_pair(dataset)[0] == "i2va"


def test_fingerprint_follows_clips_not_mtimes(
    studio_home: Path, tmp_path: Path
) -> None:
    dataset = _clips(tmp_path / "ds", ["a", "b"])
    before = adapters.dataset_fingerprint(dataset)
    for path in dataset.iterdir():
        path.touch()  # a copy, a checkout, an rsync
    assert adapters.dataset_fingerprint(dataset)["manifest_hash"] == before["manifest_hash"]

    (dataset / "clip1.wav").unlink()  # a clip really gone
    assert adapters.dataset_fingerprint(dataset)["manifest_hash"] != before["manifest_hash"]


def test_typical_caption_is_the_one_the_adapter_saw_most(
    studio_home: Path, tmp_path: Path
) -> None:
    dataset = _clips(tmp_path / "ds", ["moody folk", "synth pop", "moody folk"])
    assert adapters.typical_caption(dataset) == "moody folk"
    assert adapters.typical_caption(tmp_path / "nowhere") == ""


def test_disk_files_without_a_row_are_listed_as_found(
    studio_home: Path, tmp_path: Path
) -> None:
    path = _lora_file(studio_home, "someone-elses.safetensors")
    rows = adapters.list_adapters()
    assert [row["source"] for row in rows] == ["untracked"]
    assert rows[0]["on_disk"] is True
    assert rows[0]["path"] == str(path)
    assert rows[0]["can_audition"] is True


def test_a_deleted_file_keeps_its_story(studio_home: Path, tmp_path: Path) -> None:
    row = _trained_adapter(studio_home, tmp_path)
    Path(row["path"]).unlink()
    listed = adapters.get_adapter(row["id"])
    assert listed["on_disk"] is False
    assert listed["dataset"]["clip_count"] == 3, "provenance survives the file"
    assert not listed["can_audition"]


def test_forget_drops_the_row_not_the_file(
    studio_home: Path, tmp_path: Path
) -> None:
    row = _trained_adapter(studio_home, tmp_path)
    adapters.forget(row["id"])
    assert Path(row["path"]).is_file(), "Forget is not Delete"
    assert adapters.get_adapter(row["id"])["source"] == "untracked"
    with pytest.raises(RuntimeError, match="No registry row"):
        adapters.forget(row["id"])


def test_record_imported_can_mark_a_catalog_download(studio_home: Path) -> None:
    path = _lora_file(studio_home, "minimax_h3_motion_adapter_pilot_r16.safetensors")
    adapters.record_imported(
        {"name": path.stem, "path": str(path), "kind": "h3"},
        source="catalog",
    )
    listed = adapters.get_adapter(path.name)
    assert listed["source"] == "catalog"
    assert listed["kind"] == "h3"


def test_import_lora_records_that_we_did_not_train_it(
    studio_home: Path, tmp_path: Path
) -> None:
    from minimax_studio.worker.loras import import_lora

    source = tmp_path / "borrowed.safetensors"
    source.write_bytes(b"\x00safetensors-stub")
    row = import_lora(str(source))
    listed = adapters.get_adapter("borrowed.safetensors")
    assert listed["source"] == "imported"
    assert listed["path"] == row["path"]
    assert listed["kind"] == "music"
    assert listed["dataset"] == {} or not listed["dataset"].get("path")


def test_untracked_file_in_an_h3_folder_is_not_a_music_adapter(
    studio_home: Path,
) -> None:
    folder = Path(studio_home) / "models" / "h3-comfy" / "loras"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "shot.safetensors"
    path.write_bytes(b"\x00safetensors-stub")
    listed = adapters.get_adapter("shot.safetensors")
    assert listed["source"] == "untracked"
    assert listed["kind"] == "h3"
    assert listed["can_audition"] is True
    assert listed["path"] == str(path)


def test_import_from_an_h3_folder_keeps_h3_kind(
    studio_home: Path, tmp_path: Path
) -> None:
    from minimax_studio.worker.loras import import_lora

    source = tmp_path / "h3-comfy" / "loras" / "face.safetensors"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"\x00safetensors-stub")
    import_lora(str(source))
    listed = adapters.get_adapter("face.safetensors")
    assert listed["source"] == "imported"
    assert listed["kind"] == "h3"
    assert listed["can_audition"] is True


def test_import_kind_argument_wins_over_a_generic_folder(
    studio_home: Path, tmp_path: Path
) -> None:
    from minimax_studio.worker.loras import import_lora

    source = tmp_path / "Downloads" / "face.safetensors"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"\x00safetensors-stub")
    import_lora(str(source), kind="h3")
    listed = adapters.get_adapter("face.safetensors")
    assert listed["kind"] == "h3"
    assert listed["can_audition"] is True


# --- the audition loop ------------------------------------------------------


def _wait_job(client: TestClient, job_id: str, timeout: float = 60.0) -> dict:
    import time

    deadline = time.time() + timeout
    job: dict = {}
    while time.time() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job.get("status") in {"done", "error", "cancelled"}:
            return job
        time.sleep(0.05)
    return job


def test_audition_queues_the_adapter_with_the_caption_it_learned(
    studio_home: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MINIMAX_STUDIO_STUB", "1")
    row = _trained_adapter(studio_home, tmp_path)

    queued = adapters.audition(row["id"])
    assert queued["strength"] == adapters.AUDITION_STRENGTH == 0.8
    assert queued["prompt"] == "moody folk", "the caption it saw most, not the first file"
    assert queued["duration_s"] == adapters.AUDITION_SECONDS

    client = TestClient(app)
    job = _wait_job(client, queued["job_id"])
    assert job["status"] == "done", job
    assert job["audition"] == f"audition:{row['id']}"

    entry = next(
        item for item in client.get("/history").json() if item["id"] == job["id"]
    )
    assert entry["audition"] == f"audition:{row['id']}"
    assert entry["loras"] == [{"id": row["path"], "strength": 0.8}]
    assert entry["prompt"] == "moody folk"
    assert Path(entry["output_path"]).is_file()


def test_audition_refuses_when_there_is_nothing_to_sing_with(
    studio_home: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MINIMAX_STUDIO_STUB", "1")
    row = _trained_adapter(studio_home, tmp_path)
    # The dataset goes away: provenance still reads, but there is no caption.
    import shutil

    shutil.rmtree(row["dataset"]["path"])

    with pytest.raises(RuntimeError) as refused:
        adapters.audition(row["id"])
    assert "Nothing to audition" in str(refused.value)
    assert "dataset folder is gone" in str(refused.value)
    assert "Type a prompt" in str(refused.value)

    # A typed prompt is a real answer, so it is accepted.
    queued = adapters.audition(row["id"], prompt="bright synth pop")
    assert queued["prompt"] == "bright synth pop"


def test_audition_refuses_a_missing_file(
    studio_home: Path, tmp_path: Path
) -> None:
    row = _trained_adapter(studio_home, tmp_path)
    Path(row["path"]).unlink()
    with pytest.raises(RuntimeError, match="no file on disk"):
        adapters.audition(row["id"])


def _trained_h3_adapter(
    studio_home: Path, tmp_path: Path, *, stills: int = 2
) -> dict:
    dataset = tmp_path / "shots"
    dataset.mkdir()
    for index in range(stills):
        (dataset / f"s{index}.png").write_bytes(b"\x89PNG")
        (dataset / f"s{index}.txt").write_text("a golden retriever puppy\n", encoding="utf-8")
    source = tmp_path / "checkpoints" / "step-50" / "h3-lora.safetensors"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"\x00")
    path = _lora_file(studio_home, "h3-lora.safetensors")
    return adapters.record_trained(
        {
            "id": "20260830-h3",
            "name": "H3 shots",
            "dataset_dir": str(dataset),
            "family": "h3",
            "dataset_kind": "video",
            "preset": "h3-24g",
            "steps": 50,
            "rank": 16,
        },
        {"name": "h3-lora", "path": str(path)},
        source,
    )


def test_h3_audition_queues_a_still_pair(
    studio_home: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MINIMAX_STUDIO_STUB", "1")
    row = _trained_h3_adapter(studio_home, tmp_path, stills=2)
    mode, assets = adapters.still_pair(row["dataset"]["path"])
    assert mode == "fl2va" and len(assets) == 2
    assert {item["role"] for item in assets} == {"first_frame", "last_frame"}

    queued = adapters.audition(row["id"])
    assert queued["kind"] == "h3"
    assert queued["mode"] == "fl2va"
    assert queued["backend"] == "comfy"
    assert queued["strength"] == 0.8
    from minimax_studio.h3_timing import duration_to_frames, frames_to_seconds

    assert queued["duration_s"] == frames_to_seconds(
        duration_to_frames(adapters.AUDITION_H3_SECONDS)
    )
    assert queued["prompt"] == "a golden retriever puppy"

    client = TestClient(app)
    job = _wait_job(client, queued["job_id"])
    assert job["status"] == "done", job
    assert job["kind"] == "h3"
    assert job["audition"] == f"audition:{row['id']}"
    entry = next(
        item for item in client.get("/history").json() if item["id"] == job["id"]
    )
    assert entry["audition"] == f"audition:{row['id']}"
    assert entry["kind"] == "h3"
    assert entry["mode"] == "fl2va"
    assert entry["loras"] == [{"id": row["path"], "strength": 0.8}]
    assert Path(entry["output_path"]).is_file()


def test_h3_clip_dataset_auditions_as_text_to_video(
    studio_home: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MINIMAX_STUDIO_STUB", "1")
    dataset = tmp_path / "clips"
    dataset.mkdir()
    (dataset / "c.mp4").write_bytes(b"ftyp")
    (dataset / "c.txt").write_text("puppy in a meadow\n", encoding="utf-8")
    source = tmp_path / "ck" / "h3-lora.safetensors"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"\x00")
    path = _lora_file(studio_home, "h3-clips.safetensors")
    row = adapters.record_trained(
        {
            "id": "h3-clips",
            "name": "clips",
            "dataset_dir": str(dataset),
            "family": "h3",
            "dataset_kind": "video",
            "preset": "h3-24g",
            "steps": 50,
            "rank": 16,
        },
        {"name": "h3-clips", "path": str(path)},
        source,
    )
    queued = adapters.audition(row["id"])
    assert queued["mode"] == "t2va"
    job = _wait_job(TestClient(app), queued["job_id"])
    assert job["status"] == "done", job
    assert job["kind"] == "h3"


# --- API --------------------------------------------------------------------


def test_adapter_api_roundtrip(studio_home: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MINIMAX_STUDIO_STUB", "1")
    client = TestClient(app)
    row = _trained_adapter(studio_home, tmp_path)

    listed = client.get("/adapters").json()
    assert listed[0]["id"] == row["id"]
    assert listed[0]["dataset"]["manifest_hash"] == row["dataset"]["manifest_hash"]
    assert listed[0]["audition_prompt"] == "moody folk"
    assert listed[0]["can_audition"] is True

    audition = client.post(f"/adapters/{row['id']}/audition", json={})
    assert audition.status_code == 200
    body = audition.json()
    assert body["strength"] == 0.8 and body["prompt"] == "moody folk"
    assert _wait_job(client, body["job_id"])["status"] == "done"

    refused = client.post(
        "/adapters/nope.safetensors/audition", json={"prompt": "x"}
    )
    assert refused.status_code == 409
    assert "No adapter" in refused.json()["detail"]

    assert client.delete(f"/adapters/{row['id']}").json()["ok"] is True
    assert client.delete(f"/adapters/{row['id']}").status_code == 404
    assert client.get("/adapters").json()[0]["source"] == "untracked"


def test_registry_file_is_the_documented_shape(
    studio_home: Path, tmp_path: Path
) -> None:
    _trained_adapter(studio_home, tmp_path)
    payload = json.loads(adapters.registry_path().read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["adapters"][0]["file"] == "summer-lora.safetensors"
    # A hand-written registry (people edit these) must still load.
    adapters.registry_path().write_text(
        json.dumps([{"file": "legacy.safetensors", "source": "imported"}]),
        encoding="utf-8",
    )
    assert adapters.load_registry()[0]["file"] == "legacy.safetensors"
