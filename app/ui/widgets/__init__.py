"""Small reusable widgets. Nothing here knows about diagnostics."""

from __future__ import annotations

from app.ui.widgets.status_pill import StatusPill, status_word
from app.ui.widgets.status_strip import (
    AppStatus,
    StatusStrip,
    app_status,
    execution_word,
)

__all__ = [
    "AppStatus",
    "StatusPill",
    "StatusStrip",
    "app_status",
    "execution_word",
    "status_word",
]
