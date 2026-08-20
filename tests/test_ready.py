from minimax_studio.ui.ready import classify_preflight, notify_job_result


def test_classify_preflight_ok() -> None:
    assert classify_preflight({"ok": True}) == "ok"


def test_classify_preflight_warn() -> None:
    assert classify_preflight({"ok": True, "warnings": ["ffmpeg is not in PATH"]}) == "warn"


def test_classify_preflight_block() -> None:
    assert classify_preflight({"ok": False, "detail": "no backend"}) == "block"


def test_notify_job_result_ignores_success(monkeypatch) -> None:
    called = []
    monkeypatch.setattr(
        "minimax_studio.ui.ready.QMessageBox.warning",
        lambda *args, **kwargs: called.append(args),
    )
    notify_job_result(None, {"status": "done"})  # type: ignore[arg-type]
    notify_job_result(None, {"status": "cancelled"})  # type: ignore[arg-type]
    assert called == []
    notify_job_result(None, {"status": "error", "error": "boom"})  # type: ignore[arg-type]
    assert called
