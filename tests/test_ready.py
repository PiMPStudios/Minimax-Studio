from minimax_studio.ui.ready import (
    classify_preflight,
    confirm_generate,
    format_queue_line,
    notify_job_result,
    remember_preflight,
)


def test_confirm_generate_reuses_fresh_inspector_check(monkeypatch) -> None:
    remember_preflight(
        {
            "ok": True,
            "kind": "music",
            "requested": "auto",
            "mode": "ttm",
        },
        speed="quality",
        resolution="768P",
    )
    called = []

    class Client:
        def preflight(self, *args, **kwargs):
            called.append(args)
            return {"ok": False, "detail": "should not hit the worker"}

    assert confirm_generate(None, Client(), "music", "auto", "ttm")  # type: ignore[arg-type]
    assert called == []


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


def test_format_queue_line() -> None:
    jobs = [
        {"id": "a", "kind": "h3", "status": "running"},
        {"id": "b", "kind": "h3", "status": "queued"},
        {"id": "c", "kind": "music", "status": "queued"},
    ]
    assert "1 queued" in format_queue_line(jobs, "h3", "a")
    assert "another job" not in format_queue_line(jobs, "h3", "a")
    assert "another job" in format_queue_line(jobs, "h3", "b")
    assert format_queue_line(jobs, "music", None) == "1 queued"
