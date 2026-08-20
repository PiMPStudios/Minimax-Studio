from minimax_studio.worker.backends.h3 import duration_to_frames


def test_frame_grid_stays_in_window() -> None:
    for seconds in (4, 5, 8, 10, 12, 15, 20):
        frames = duration_to_frames(seconds)
        assert (frames - 5) % 17 == 0
        assert 5.0 <= frames / 24.0 <= 15.0
