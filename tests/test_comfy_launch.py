from pathlib import Path

from minimax_studio.worker.comfy_launch import (
    build_comfy_argv,
    detect_comfy,
    parse_comfy_listen,
    start_comfy,
)
from minimax_studio.worker.runtime import runtime


def test_parse_comfy_listen() -> None:
    assert parse_comfy_listen("http://127.0.0.1:8188") == ("127.0.0.1", 8188)
    assert parse_comfy_listen("http://0.0.0.0:9000") == ("0.0.0.0", 9000)
    assert parse_comfy_listen("http://localhost") == ("127.0.0.1", 8188)


def test_build_argv_defaults(tmp_path: Path) -> None:
    python = tmp_path / "python"
    python.write_text("")
    root = tmp_path
    argv = build_comfy_argv(python, root, "http://127.0.0.1:8188", "")
    assert argv[0] == str(python)
    assert argv[1] == str(root / "main.py")
    assert "--listen" in argv
    assert "127.0.0.1" in argv
    assert "--port" in argv
    assert "8188" in argv


def test_build_argv_extra_overrides_listen(tmp_path: Path) -> None:
    python = tmp_path / "python"
    python.write_text("")
    argv = build_comfy_argv(
        python,
        tmp_path,
        "http://127.0.0.1:8188",
        "--listen 0.0.0.0 --default-device 1",
    )
    assert argv.count("--listen") == 1
    assert "0.0.0.0" in argv
    assert "--default-device" in argv
    assert "1" in argv


def test_detect_comfy_from_config_root(studio_home: Path, tmp_path: Path) -> None:
    root = tmp_path / "ComfyUI"
    root.mkdir()
    (root / "main.py").write_text("# comfy\n")
    venv = root / ".venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text("")
    runtime.config = runtime.config.model_copy(update={"comfy_root": str(root)})
    info = detect_comfy()
    assert info["root"] == str(root.resolve())
    assert info["python"]
    assert info["argv"]
    assert "main.py" in info["argv"][1]


def test_start_already_running(studio_home: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "minimax_studio.worker.comfy_launch._comfy_running", lambda: True
    )
    result = start_comfy()
    assert result["ok"] is True
    assert result["already"] is True


def test_start_popen(studio_home: Path, tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "ComfyUI"
    root.mkdir()
    (root / "main.py").write_text("# comfy\n")
    venv = root / ".venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text("")
    runtime.config = runtime.config.model_copy(update={"comfy_root": str(root)})
    monkeypatch.setattr(
        "minimax_studio.worker.comfy_launch._comfy_running", lambda: False
    )

    class FakeProc:
        pid = 4242

        def poll(self):
            return None

    captured: dict = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr(
        "minimax_studio.worker.comfy_launch.subprocess.Popen", fake_popen
    )
    runtime.comfy_proc = None
    result = start_comfy()
    assert result["ok"] is True
    assert result["already"] is False
    assert result["pid"] == 4242
    assert captured["argv"][1].endswith("main.py")
    assert captured["kwargs"]["cwd"] == str(root.resolve())
