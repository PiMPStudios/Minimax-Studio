"""Shared stand-ins for modal dialogs — because a modal in a test is a hung CI.

Pages call ``QMessageBox.question`` / ``QFileDialog.getExistingDirectory``
directly (that is the right thing for a person: the box is the confirmation).
For a test, the box has to answer itself, and the way to do that without
monkeypatching Qt itself is to replace the *name inside the page module*, since
Shiboken widget classes will not take attributes.

Usage::

    dialogs = Dialogs(monkeypatch, {"question": Yes, "folder": "/tmp/out"},
                      train_page, storage_dialog)
    page._prune()
    assert "3.1 GB" in dialogs.last()[2]
"""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QMessageBox

__all__ = ["Dialogs", "MessageBoxShim"]


class Dialogs:
    """Record every box a page opened, and answer them on the test's behalf."""

    def __init__(self, monkeypatch, answers: dict[str, Any], *modules) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self._answers = answers
        shim = MessageBoxShim(self)
        for module in modules:
            monkeypatch.setattr(module, "QMessageBox", shim)
            monkeypatch.setattr(module, "QFileDialog", shim, raising=False)

    def record(self, kind: str, args: tuple) -> None:
        # QMessageBox.question(parent, title, text) vs
        # QFileDialog.getExistingDirectory(parent, title, directory)
        title = str(args[1]) if len(args) > 1 else ""
        body = str(args[2]) if len(args) > 2 else ""
        self.calls.append((kind, title, body))

    def answer(self, kind: str) -> Any:
        return self._answers.get(kind, 0)

    def kinds(self) -> list[str]:
        return [call[0] for call in self.calls]

    def last(self) -> tuple[str, str, str]:
        return self.calls[-1]

    def bodies(self) -> str:
        """Every box body joined — the cheap way to assert a number was named."""
        return "\n".join(call[2] for call in self.calls)


class MessageBoxShim:
    """Duck-types the two classes pages reach for, and says no by default."""

    def __init__(self, dialogs: Dialogs) -> None:
        self._dialogs = dialogs
        self.StandardButton = QMessageBox.StandardButton
        self.Yes = QMessageBox.StandardButton.Yes
        self.No = QMessageBox.StandardButton.No

    # QMessageBox

    def warning(self, *args, **kwargs):
        self._dialogs.record("warning", args)

    def information(self, *args, **kwargs):
        self._dialogs.record("information", args)

    def question(self, *args, **kwargs):
        self._dialogs.record("question", args)
        return self._dialogs.answer("question")

    def critical(self, *args, **kwargs):
        self._dialogs.record("critical", args)

    # QFileDialog

    def getExistingDirectory(self, *args, **kwargs) -> str:
        self._dialogs.record("folder", args)
        return str(self._dialogs.answer("folder") or "")

    def getOpenFileName(self, *args, **kwargs) -> tuple[str, str]:
        self._dialogs.record("file", args)
        return str(self._dialogs.answer("file") or ""), ""

    def getSaveFileName(self, *args, **kwargs) -> tuple[str, str]:
        self._dialogs.record("file", args)
        return str(self._dialogs.answer("file") or ""), ""
