"""PLAN-V2 S1: dataset lifecycle — real WAVs written by the stdlib, no GPU."""

from __future__ import annotations

import sys
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
    assert any("no matching media" in p for p in problems["orphan.txt"])

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


def test_empty_datasets_are_refused_by_name(datasets_env: Path) -> None:
    empty = datasets.create_dataset("empty")
    report = datasets.validate_dataset(empty["id"])
    assert report["ok"] is False and report["checked"] == 0


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


# --- Video (H3) datasets: PLAN-V2 S4, stills first ---------------------------

#: Filename protocol, so a test reads as a claim about media rather than about
#: JSON: `1280x720` sets the pixel size, `t4s` the duration, `audio` an
#: audio stream. Everything else is a plain video/still with no streams.
FFPROBE_STUB = '''
import json, re, sys
name = sys.argv[-1].lower()
dims = re.search(r"(\\d+)x(\\d+)", name)
dur = re.search(r"t(\\d+(?:\\.\\d+)?)s", name)
streams = []
streams.append({
    "codec_type": "video",
    "width": int(dims.group(1)) if dims else 0,
    "height": int(dims.group(2)) if dims else 0,
})
if dur:
    streams[0]["duration"] = dur.group(1)
if "audio" in name:
    streams.append({"codec_type": "audio"})
print(json.dumps({"streams": streams,
                  "format": {"duration": dur.group(1) if dur else "N/A"}}))
'''


@pytest.fixture
def ffprobe(monkeypatch, tmp_path: Path) -> Path:
    script = tmp_path / "ffprobe_stub.py"
    script.write_text(FFPROBE_STUB, encoding="utf-8")
    monkeypatch.setenv(
        "MINIMAX_STUDIO_FFPROBE_BIN", f'"{sys.executable}" "{script}"'
    )
    return tmp_path


def _video_dataset(name: str = "stills", files: dict[str, str] | None = None) -> tuple[Path, dict]:
    """A video dataset with the given files, each key mapping to a caption (or
    None for no caption)."""
    manifest = datasets.create_dataset(name, "video")
    folder, _ = datasets.get_dataset(manifest["id"])
    for file_name, caption in (files or {}).items():
        (folder / file_name).write_bytes(b"\x00stub-media")
        if caption is not None:
            (folder / Path(file_name).with_suffix(".txt")).write_text(
                caption, encoding="utf-8"
            )
    return folder, manifest


def test_stills_and_short_clips_validate_clean(ffprobe, datasets_env: Path) -> None:
    _folder, manifest = _video_dataset(
        "campaign",
        {
            "cover_1280x720.png": "neon cover shot",
            "take_1280x720_t4s.mp4": "camera push on the band",
        },
    )
    report = datasets.validate_dataset(manifest["id"])
    assert report["ok"] is True, report["rows"]
    assert report["stills"] == 1 and report["clips"] == 1
    assert report["target_mode"] == "video"
    assert report["av_ready"] is False  # one still and a silent clip: no audio to train
    rows = {row["file"]: row for row in report["rows"]}
    assert rows["cover_1280x720.png"]["entry_kind"] == "still"
    assert rows["take_1280x720_t4s.mp4"]["seconds"] == 4.0


def test_a_thumbnail_and_a_long_clip_are_refused_with_their_numbers(
    ffprobe, datasets_env: Path
) -> None:
    _folder, manifest = _video_dataset(
        "scrap",
        {
            "tiny_128x96.png": "poster thumb",
            "talky_1920x1080_t12s.mp4": "dialogue scene",
        },
    )
    report = datasets.validate_dataset(manifest["id"])
    assert report["ok"] is False
    problems = {row["file"]: row["problems"] for row in report["rows"]}
    assert any("128×96 is under the 256 px" in p for p in problems["tiny_128x96.png"])
    (long_problem,) = problems["talky_1920x1080_t12s.mp4"]
    assert "12.0s is over the 8s cap" in long_problem
    # The reason for the cap belongs in the message, not only in the plan.
    assert "audio heads" in long_problem


def test_video_entries_need_captions_too(ffprobe, datasets_env: Path) -> None:
    _folder, manifest = _video_dataset(
        "nocaptions",
        {"shot_1280x720_t3s.mp4": None, "stray.txt": "nothing behind me"},
    )
    report = datasets.validate_dataset(manifest["id"])
    problems = {row["file"]: row["problems"] for row in report["rows"]}
    assert any("shot_1280x720_t3s.txt" in p for p in problems["shot_1280x720_t3s.mp4"])
    assert any("no matching media" in p for p in problems["stray.txt"])


def test_without_ffprobe_the_validator_warns_instead_of_accusing(
    monkeypatch, datasets_env: Path
) -> None:
    """Missing ffmpeg is our limitation, not the user's broken clip."""
    monkeypatch.setattr(datasets, "ffprobe_command", lambda: None)
    _folder, manifest = _video_dataset(
        "unmeasured", {"shot_1280x720_t999s.mp4": "whatever it is"}
    )
    report = datasets.validate_dataset(manifest["id"])
    assert report["ok"] is True  # the caption is there; nothing else was checkable
    assert report["rows"][0]["ok"] is True
    assert any("install ffmpeg/ffprobe" in warning for warning in report["warnings"])


def test_av_mode_needs_audio_in_every_clip_and_no_stills(
    ffprobe, datasets_env: Path
) -> None:
    _folder, manifest = _video_dataset(
        "av-ok",
        {
            "a_1280x720_t4s_audio.mp4": "verse",
            "b_1280x720_t5s_audio.mp4": "chorus",
        },
    )
    assert datasets.validate_dataset(manifest["id"])["av_ready"] is True
    updated = datasets.set_h3_target_mode(manifest["id"], "av")
    assert updated["h3_target_mode"] == "av"
    assert datasets.validate_dataset(manifest["id"])["target_mode"] == "av"


def test_av_mode_refusals_name_what_is_missing(ffprobe, datasets_env: Path) -> None:
    _folder, silent = _video_dataset(
        "silent-set",
        {"a_1280x720_t4s_audio.mp4": "verse", "b_1280x720_t5s.mp4": "chorus"},
    )
    with pytest.raises(RuntimeError, match="have none"):
        datasets.set_h3_target_mode(silent["id"], "av")

    _folder, with_still = _video_dataset(
        "mixed-set",
        {"a_1280x720_t4s_audio.mp4": "verse", "cover_1280x720.png": "cover"},
    )
    with pytest.raises(RuntimeError, match="1 still"):
        datasets.set_h3_target_mode(with_still["id"], "av")

    music = datasets.create_dataset("songs", "music")
    with pytest.raises(RuntimeError, match="only a Video"):
        datasets.set_h3_target_mode(music["id"], "av")

    _folder, ok = _video_dataset("strict", {"a_1280x720_t4s_audio.mp4": "verse"})
    with pytest.raises(RuntimeError, match="Unknown H3 target mode"):
        datasets.set_h3_target_mode(ok["id"], "video-with-audio")


def test_av_mode_reports_the_silent_clip_by_name_after_it_is_chosen(
    ffprobe, datasets_env: Path
) -> None:
    folder, manifest = _video_dataset(
        "gone-quiet", {"a_1280x720_t4s_audio.mp4": "verse"}
    )
    datasets.set_h3_target_mode(manifest["id"], "av")
    (folder / "b_1280x720_t5s.mp4").write_bytes(b"\x00stub-media")
    (folder / "b_1280x720_t5s.txt").write_text("added later", encoding="utf-8")
    report = datasets.validate_dataset(manifest["id"])
    assert report["ok"] is False
    problems = {row["file"]: row["problems"] for row in report["rows"]}
    assert any("no audio stream" in p for p in problems["b_1280x720_t5s.mp4"])


def test_import_folder_brings_stills_into_a_video_dataset(
    ffprobe, datasets_env: Path, tmp_path: Path
) -> None:
    manifest = datasets.create_dataset("moodboard", "video")
    source = tmp_path / "shots"
    source.mkdir()
    (source / "one_1280x720.png").write_bytes(b"\x00a")
    (source / "one_1280x720.txt").write_text("first frame", encoding="utf-8")
    (source / "song.wav").write_bytes(b"RIFF")  # not this kind of dataset

    result = datasets.import_folder(manifest["id"], str(source))
    assert result["copied"] == ["one_1280x720.png"]
    assert result["captions"] == 1


def test_a_video_dataset_gates_training_the_same_way(ffprobe, datasets_env: Path) -> None:
    """train_runs calls assert_trainable for every kind — the H3 path gets the
    same refusal as Music, not a freer pass."""
    folder, manifest = _video_dataset(
        "ready", {"cover_1280x720.png": "cover"}
    )
    datasets.assert_trainable(folder)

    (folder / "cover_1280x720.txt").unlink()
    with pytest.raises(RuntimeError, match="not ready to train"):
        datasets.assert_trainable(folder)
