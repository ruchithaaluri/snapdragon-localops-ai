"""Single source of truth for the application version."""

from __future__ import annotations

from importlib import metadata

#: Fallback used when the distribution is not installed (e.g. running from a
#: source checkout). Keep in sync with ``[project] version`` in pyproject.toml.
_FALLBACK_VERSION = "0.1.0"


def get_version() -> str:
    """Return the installed distribution version, or the source fallback."""
    try:
        return metadata.version("snapdragon-localops")
    except metadata.PackageNotFoundError:
        return _FALLBACK_VERSION


__version__ = get_version()
__all__ = ["__version__", "get_version"]
