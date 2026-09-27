"""A coloured label that always carries its status as a word.

Colour alone is never the message. Every pill renders ``word`` in text next to
its colour, so the UI stays readable in greyscale and for a user with colour
vision deficiency - which is the point of a status indicator.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

from app.ui.theme import Theme

__all__ = ["StatusPill", "status_word"]


def status_word(ok: bool, *, yes: str = "yes", no: str = "no") -> str:
    """Map a boolean onto an explicit word rather than a tick or a cross."""
    return yes if ok else no


class StatusPill(QLabel):
    """A small, self-describing status label."""

    def __init__(self, theme: Theme, parent=None) -> None:  # noqa: ANN001 - Qt parent
        super().__init__(parent)
        self._theme = theme
        self.setAlignment(Qt.AlignCenter)

    def set_status(self, word: str, status: str = "info", tooltip: str = "") -> None:
        """Show ``word`` in the colour named by ``status``."""
        color = self._theme.status(status)
        self.setText(word)
        self.setStyleSheet(
            f"color: {color}; border: 1px solid {color};"
            f"border-radius: 9px; padding: 1px 10px; font-weight: 600;"
        )
        if tooltip:
            self.setToolTip(tooltip)
        self.setAccessibleName(f"Status: {word}")
