from minimax_studio.h3_timing import duration_to_frames, format_h3_duration, resolve_dims
from minimax_studio.worker.backends.h3 import duration_to_frames as h3_frames


def test_frame_grid_stays_in_window() -> None:
    for seconds in (4, 5, 8, 10, 12, 15, 20):
        frames = duration_to_frames(seconds)
        assert (frames - 5) % 17 == 0
        assert 5.0 <= frames / 24.0 <= 15.0


def test_h3_reexports_timing() -> None:
    assert h3_frames(8) == duration_to_frames(8)
    assert "192f" in format_h3_duration(8)


def test_resolve_dims_768p_portrait() -> None:
    width, height = resolve_dims("768P", "9:16", 960, 544)
    assert width == 768
    assert height % 32 == 0
    assert height > width


def test_resolve_dims_preview_and_34() -> None:
    width, height = resolve_dims("768P", "16:9", 960, 544, quality="preview")
    assert height == 480
    assert width % 32 == 0
    w2, h2 = resolve_dims("768P", "3:4", 960, 544, quality="native")
    assert w2 == 768
    assert h2 > w2
