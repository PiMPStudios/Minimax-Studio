from pathlib import Path

from minimax_studio.ui.reveal import reveal_command


def test_reveal_linux_opens_parent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("minimax_studio.ui.reveal.platform.system", lambda: "Linux")
    media = tmp_path / "take.wav"
    media.write_bytes(b"x")
    cmd = reveal_command(media)
    assert cmd[0] == "xdg-open"
    assert cmd[1] == str(tmp_path)


def test_reveal_windows_selects_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("minimax_studio.ui.reveal.platform.system", lambda: "Windows")
    media = tmp_path / "take.mp4"
    media.write_bytes(b"x")
    cmd = reveal_command(media)
    assert cmd[0] == "explorer"
    assert "/select," in cmd


def test_reveal_macos_reveals_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("minimax_studio.ui.reveal.platform.system", lambda: "Darwin")
    media = tmp_path / "take.wav"
    media.write_bytes(b"x")
    cmd = reveal_command(media)
    assert cmd[:2] == ["open", "-R"]
