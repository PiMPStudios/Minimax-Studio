from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class StudioState(QObject):
    changed = Signal()
    restore_music = Signal(dict)
    restore_video = Signal(dict)
    open_history = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.backend = "auto"
        self.duration = 30
        self.seed = -1
        self.steps = 30
        self.lora_id = ""
        self.lora_strength = 1.0
        self.lora2_id = ""
        self.lora2_strength = 1.0
        self.speed = "quality"
        self.attention = "default"
        self.ref_image_size = "match"
        self.cfg = 1.7
        self.quality = "native"

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

    def set_lora(self, lora_id: str, strength: float) -> None:
        self.lora_id = lora_id
        self.lora_strength = float(strength)
        self.changed.emit()

    def set_lora2(self, lora_id: str, strength: float) -> None:
        self.lora2_id = lora_id
        self.lora2_strength = float(strength)
        self.changed.emit()

    def lora_payload(self) -> list[dict]:
        rows = []
        if self.lora_id:
            rows.append({"id": self.lora_id, "strength": self.lora_strength})
        if self.lora2_id and self.lora2_id != self.lora_id:
            rows.append({"id": self.lora2_id, "strength": self.lora2_strength})
        return rows

    def set_speed(self, value: str) -> None:
        key = value.strip().lower()
        if key != self.speed:
            self.speed = key
            self.changed.emit()

    def set_attention(self, value: str) -> None:
        key = value.strip().lower()
        if key != self.attention:
            self.attention = key
            self.changed.emit()

    def set_ref_image_size(self, value: str) -> None:
        key = value.strip().lower()
        if key != self.ref_image_size:
            self.ref_image_size = key
            self.changed.emit()

    def set_cfg(self, value: float) -> None:
        if float(value) != self.cfg:
            self.cfg = float(value)
            self.changed.emit()

    def set_quality(self, value: str) -> None:
        key = value.strip().lower()
        if key != self.quality:
            self.quality = key
            self.changed.emit()
