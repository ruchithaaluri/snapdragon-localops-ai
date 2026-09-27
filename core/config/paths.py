"""Where the application keeps its files.

Resolution is explicit and overridable so tests never touch a real user
profile:

1. an explicit value from configuration,
2. the ``LOCALOPS_DATA_DIR`` environment variable,
3. ``%LOCALAPPDATA%/SnapdragonLocalOps`` on Windows,
4. ``~/.local/share/snapdragon-localops`` elsewhere (for developers on Linux
   and macOS, and for CI).

Logs, the SQLite database and downloaded models all live under this root.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "APP_DIR_NAME",
    "DATA_DIR_ENV_VAR",
    "cache_dir",
    "data_dir",
    "ensure_dir",
    "index_dir",
    "logs_dir",
    "models_dir",
    "reports_dir",
    "runtime_dir",
]

APP_DIR_NAME = "SnapdragonLocalOps"
DATA_DIR_ENV_VAR = "LOCALOPS_DATA_DIR"


def data_dir() -> Path:
    """Return the root data directory (not created)."""
    override = os.environ.get(DATA_DIR_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            return Path(local_appdata) / APP_DIR_NAME
        return Path.home() / "AppData" / "Local" / APP_DIR_NAME
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return base / "snapdragon-localops"


def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def logs_dir() -> Path:
    """Rotating application logs."""
    return data_dir() / "logs"


def runtime_dir() -> Path:
    """SQLite database and other mutable runtime state."""
    return data_dir() / "runtime"


def cache_dir() -> Path:
    """Regenerable caches (tokenizers, compiled shapes). Safe to delete."""
    return data_dir() / "cache"


def models_dir() -> Path:
    """Model artifacts referenced by configuration.

    This directory is intentionally *not* created on start-up: the application
    must never imply that a model is present when it is not.
    """
    return data_dir() / "models"


def index_dir() -> Path:
    """Persisted retrieval indexes."""
    return data_dir() / "index"


def reports_dir() -> Path:
    """Benchmark and diagnostic reports."""
    return data_dir() / "reports"
