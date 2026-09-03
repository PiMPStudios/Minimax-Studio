"""Exceptions both the GUI and the worker may import.

The GUI must not import ``minimax_studio.worker``. These types cross the
localhost HTTP boundary as status codes, not as pickled classes.
"""


class InsufficientDisk(RuntimeError):
    """The worker refused on free space.

    HTTP 507 (not 409 — that status is already train/audition conflicts).
    The UI turns this into “Download anyway?”, so it must be catchable
    without matching the error prose.
    """
