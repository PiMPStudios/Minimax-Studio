"""PLAN-V2 S1: dataset lifecycle — real WAVs written by the stdlib, no GPU."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from minimax_studio.worker import datasets


@pytest.fixture
def datasets_env(studio_home: Path, tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "datasets"
    monkeypatch.setenv("MINIMAX_STUDIO_DATASETS", str(root))
    return root


def _wav(path: Path, seconds: float = 4.0, rate: int = 8000) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(seconds * rate))
    return path


def _clean_dataset(name: str = "hits") -> tuple[Path, dict]:
    manifest = datasets.create_dataset(name)
    folder, _ = datasets.get_dataset(manifest["id"])
    _wav(folder / "one.wav")
    (folder / "one.txt").write_text("moody folk", encoding="utf-8")
    _wav(folder / "two.wav")
    (folder / "two.txt").write_text("synth pop", encoding="utf-8")
    return folder, manifest


def test_create_list_get_delete(datasets_env: Path) -> None:
    manifest = datasets.create_dataset("My Hits", "music", "summer")
    assert manifest["kind"] == "music"
    rows = datasets.list_datasets()
    assert [row["id"] for row in rows] == [manifest["id"]]
    assert rows[0]["clip_count"] == 0

    datasets.delete_dataset(manifest["id"])
    assert datasets.list_datasets() == []


def test_duplicate_names_are_named_not_clobbered(datasets_env: Path) -> None:
    datasets.create_dataset("My Hits")
    with pytest.raises(RuntimeError, match="already exists"):
        datasets.create_dataset("My Hits!")  # same slug


def test_unknown_kind_rejected(datasets_env: Path) -> None:
    with pytest.raises(RuntimeError, match="kind"):
        datasets.create_dataset("weird", "smoothie")


def test_import_folder_copies_media_with_captions(datasets_env: Path, tmp_path: Path) -> None:
    manifest = datasets.create_dataset("imported")
    source = tmp_path / "source"
    source.mkdir()
    _wav(source / "song a.wav")
    (source / "song a.txt").write_text("caption a", encoding="utf-8")
    (source / "song a.lyrics").write_text("[verse]\nla", encoding="utf-8")
    _wav(source / "b.wav")
    (source / "notes.md").write_text("ignored", encoding="utf-8")

    result = datasets.import_folder(manifest["id"], source)
    assert sorted(result["copied"]) == ["b.wav", "song a.wav"]
    assert result["captions"] == 2  # .txt and .lyrics both travelled

    folder, _ = datasets.get_dataset(manifest["id"])
    assert (folder / "song a.wav").is_file()  # copies: source stays intact
    assert (source / "song a.wav").is_file()
    assert (folder / "song a.lyrics").read_text(encoding="utf-8") == "[verse]\nla"
    assert not (folder / "notes.md").exists()


def test_import_collision_gets_suffix(datasets_env: Path, tmp_path: Path) -> None:
    manifest = datasets.create_dataset("collide")
    folder, _ = datasets.get_dataset(manifest["id"])
    _wav(folder / "same.wav")
    source = tmp_path / "src"
    source.mkdir()
    _wav(source / "same.wav")
    result = datasets.import_folder(manifest["id"], source)
    assert result["copied"] == ["same-2.wav"]


def test_from_history_writes_caption_and_lyrics(
    datasets_env: Path, studio_home: Path, tmp_path: Path
) -> None:
    from minimax_studio.worker.history import record_entry

    clip = _wav(tmp_path / "gen.wav", seconds=12.0)
    record_entry(
        {
            "id": "job-abc",
            "kind": "music",
            "media_type": "audio",
            "prompt": "rainy bossa nova",
            "lyrics": "[verse]\ncity lights",
            "output_path": str(clip),
        }
    )
    manifest = datasets.create_dataset("from history")
    result = datasets.add_from_history(manifest["id"], "job-abc")

    folder, _ = datasets.get_dataset(manifest["id"])
    added = folder / result["added"]
    assert added.is_file()
    assert added.with_suffix(".txt").read_text(encoding="utf-8").strip() == (
        "rainy bossa nova"
    )
    assert added.with_suffix(".lyrics").read_text(encoding="utf-8").strip() == (
        "[verse]\ncity lights"
    )
    # And the dataset validates clean end-to-end.
    report = datasets.validate_dataset(manifest["id"])
    assert report["ok"] is True, report


def test_from_history_rejects_missing_or_mismatched(
    datasets_env: Path, studio_home: Path, tmp_path: Path
) -> None:
    from minimax_studio.worker.history import record_entry

    manifest = datasets.create_dataset("strict")
    with pytest.raises(RuntimeError, match="No history entry"):
        datasets.add_from_history(manifest["id"], "ghost")

    record_entry({"id": "no-file", "kind": "music", "output_path": "/nope.wav"})
    with pytest.raises(RuntimeError, match="no output file"):
        datasets.add_from_history(manifest["id"], "no-file")

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    record_entry(
        {"id": "video-one", "kind": "h3", "output_path": str(clip), "prompt": "p"}
    )
    with pytest.raises(RuntimeError, match="music"):
        datasets.add_from_history(manifest["id"], "video-one")


def test_validate_reports_named_problems(datasets_env: Path) -> None:
    folder, manifest = _clean_dataset()
    _wav(folder / "tiny.wav", seconds=1.0)
    (folder / "tiny.txt").write_text("short", encoding="utf-8")
    _wav(folder / "long.wav", seconds=350.0)
    (folder / "long.txt").write_text("epic", encoding="utf-8")
    _wav(folder / "nocaption.wav")
    (folder / "junk.wav").write_bytes(b"not a wav")
    (folder / "orphan.txt").write_text("no audio behind me", encoding="utf-8")

    report = datasets.validate_dataset(manifest["id"])
    assert report["ok"] is False
    problems = {row["file"]: row["problems"] for row in report["rows"]}
    assert any("under the 3s floor" in p for p in problems["tiny.wav"])
    assert any("over the 300s cap" in p for p in problems["long.wav"])
    assert any("nocaption.txt" in p for p in problems["nocaption.wav"])
    assert any("cannot read" in p for p in problems["junk.wav"])
    assert any("no matching audio" in p for p in problems["orphan.txt"])

    # Summary lives in the manifest; the full report travels with the folder.
    listing = {row["id"]: row for row in datasets.list_datasets()}
    assert listing[manifest["id"]]["last_validation"]["ok"] is False
    assert listing[manifest["id"]]["clip_count"] == 6  # junk.wav counts as an entry
    assert (folder / "validation.json").is_file()


def test_clean_dataset_validates_ok(datasets_env: Path) -> None:
    _folder, manifest = _clean_dataset()
    report = datasets.validate_dataset(manifest["id"])
    assert report["ok"] is True
    assert report["checked"] == 2


def test_empty_and_video_datasets(datasets_env: Path) -> None:
    empty = datasets.create_dataset("empty")
    report = datasets.validate_dataset(empty["id"])
    assert report["ok"] is False and report["checked"] == 0

    video = datasets.create_dataset("clips", "video")
    report = datasets.validate_dataset(video["id"])
    assert report["ok"] is False
    assert any("S4" in p for p in report["rows"][0]["problems"])


def test_assert_trainable_gates_managed_and_plain_folders(datasets_env: Path) -> None:
    folder, manifest = _clean_dataset()
    datasets.assert_trainable(folder)  # clean → passes

    (folder / "one.txt").unlink()
    with pytest.raises(RuntimeError, match="not ready to train"):
        datasets.assert_trainable(folder)

    plain = folder.parent.parent / "loose"  # no manifest anywhere in sight
    plain.mkdir(parents=True, exist_ok=True)
    datasets.assert_trainable(plain)  # unmanaged → S0 light check only


def test_start_run_blocks_on_broken_dataset(
    datasets_env: Path, tmp_path: Path, monkeypatch
) -> None:
    import sys

    from minimax_studio.worker import train_runs

    stub = tmp_path / "stub.py"
    stub.write_text("import sys; print('4.8.0' if '--version' in sys.argv else '')")
    monkeypatch.setenv(
        "MINIMAX_STUDIO_SIMPLETUNER_BIN", f'"{sys.executable}" "{stub}"'
    )
    monkeypatch.setenv("MINIMAX_STUDIO_TRAIN_RUNS", str(tmp_path / "runs"))
    monkeypatch.setattr(
        "minimax_studio.worker.train_runs.train_preflight",
        lambda _preset: {"ok": True, "detail": "", "warnings": []},
    )
    folder, manifest = _clean_dataset()
    (folder / "two.txt").unlink()  # silently broken after the fact
    with pytest.raises(RuntimeError, match="two.wav"):
        train_runs.start_run("blocked", folder, "24g")


def test_dataset_api_roundtrip(datasets_env: Path, tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from minimax_studio.worker.server import app

    client = TestClient(app)

    created = client.post("/datasets", json={"name": "api set"})
    assert created.status_code == 200
    dataset_id = created.json()["id"]

    dup = client.post("/datasets", json={"name": "api set"})
    assert dup.status_code == 409

    source = tmp_path / "src"
    source.mkdir()
    _wav(source / "a.wav")
    (source / "a.txt").write_text("aaa", encoding="utf-8")
    imported = client.post(
        f"/datasets/{dataset_id}/import", json={"folder": str(source)}
    )
    assert imported.json()["copied"] == ["a.wav"]

    report = client.post(f"/datasets/{dataset_id}/validate").json()
    assert report["ok"] is True

    detail = client.get(f"/datasets/{dataset_id}").json()
    assert detail["entries"][0]["file"] == "a.wav"
    assert detail["validation"]["ok"] is True
    assert detail["last_validation"]["ok"] is True
    # The Train page hands this straight back as dataset_dir, and "Show in
    # folder" opens it — so the worker, not the UI, has to say where it lives.
    assert Path(detail["path"]) == datasets.datasets_root() / dataset_id
    assert Path(detail["path"]).is_dir()

    assert client.delete(f"/datasets/{dataset_id}").json()["ok"] is True
    assert client.get(f"/datasets/{dataset_id}").status_code == 404
    assert client.post("/datasets", json={"name": "x", "kind": "soup"}).status_code == 409
