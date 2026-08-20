from minimax_studio.ui.ready import classify_preflight


def test_classify_preflight_ok() -> None:
    assert classify_preflight({"ok": True}) == "ok"


def test_classify_preflight_warn() -> None:
    assert classify_preflight({"ok": True, "warnings": ["ffmpeg is not in PATH"]}) == "warn"


def test_classify_preflight_block() -> None:
    assert classify_preflight({"ok": False, "detail": "no backend"}) == "block"
