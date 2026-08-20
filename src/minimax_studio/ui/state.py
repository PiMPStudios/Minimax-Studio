from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class StudioState(QObject):
    changed = Signal()
    restore_music = Signal(dict)
    open_history = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.backend = "auto"
        self.duration = 30
        self.seed = -1
        self.steps = 30

    def set_backend(self, value: str) -> None:
        key = value.strip().lower()
        if key != self.backend:
            self.backend = key
            self.changed.emit()

    def set_duration(self, value: int) -> None:
        if value != self.duration:
            self.duration = int(value)
            self.changed.emit()

    def set_seed(self, value: int) -> None:
        if value != self.seed:
            self.seed = int(value)
            self.changed.emit()

    def set_steps(self, value: int) -> None:
        if value != self.steps:
            self.steps = int(value)
            self.changed.emit()
