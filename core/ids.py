"""Identifier and timestamp helpers.

All identifiers are opaque, prefixed, and time-ordered enough to be readable in
logs (``run_20260925T101500_a1b2c3``). Timestamps are always timezone-aware
UTC; the UI is responsible for local rendering.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

__all__ = ["new_id", "new_run_id", "new_session_id", "utc_now"]


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    """Create a short unique identifier such as ``ev_9f1c2d3e4a5b6c7d``."""
    if not prefix or not prefix.replace("_", "").isalnum():
        raise ValueError(f"identifier prefix must be alphanumeric, got {prefix!r}")
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def new_run_id() -> str:
    """Create an identifier for one end-to-end diagnostic run."""
    stamp = utc_now().strftime("%Y%m%dT%H%M%S")
    return f"run_{stamp}_{uuid.uuid4().hex[:8]}"


def new_session_id() -> str:
    """Create an identifier for one application session."""
    return f"sess_{uuid.uuid4().hex[:16]}"
