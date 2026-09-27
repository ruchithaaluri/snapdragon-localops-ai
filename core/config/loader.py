"""Loading configuration from TOML files, the environment, and overrides.

Precedence, lowest to highest:

1. the defaults in :mod:`core.config.schema`,
2. a TOML file (``--config``, or ``LOCALOPS_CONFIG_FILE``),
3. environment variables prefixed with ``LOCALOPS_`` (nested with ``__``,
   e.g. ``LOCALOPS_LOGGING__LEVEL=DEBUG``),
4. explicit keyword overrides (used by the CLI and by tests).

A missing default file is not an error; a *malformed* one always is.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from core.config.schema import AppSettings
from core.errors import ConfigurationError

__all__ = ["CONFIG_FILE_ENV_VAR", "SettingsSource", "default_config_path", "load_settings"]

CONFIG_FILE_ENV_VAR = "LOCALOPS_CONFIG_FILE"


@dataclass(frozen=True, slots=True)
class SettingsSource:
    """Provenance of a loaded configuration, for the About dialog and logs."""

    path: Path | None
    from_defaults: bool
    overrides: tuple[str, ...] = ()

    def describe(self) -> str:
        return str(self.path) if self.path else "built-in defaults"


def default_config_path() -> Path | None:
    """Return the user config path if it exists.

    Checked in the repository root (``config/localops.toml``) and then in the
    data directory. Returns ``None`` when neither exists, which is the normal
    case for a fresh checkout.
    """
    from core.config.paths import data_dir

    candidates = (
        Path.cwd() / "config" / "localops.toml",
        data_dir() / "localops.toml",
    )
    return next((c for c in candidates if c.is_file()), None)


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(
            "configuration file is not valid TOML", path=str(path), detail=str(exc)
        ) from exc
    except OSError as exc:
        raise ConfigurationError(
            "configuration file could not be read", path=str(path), detail=str(exc)
        ) from exc


def load_settings(
    config_path: str | Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
    use_environment: bool = True,
    require_file: bool = False,
) -> AppSettings:
    """Build an :class:`AppSettings` instance from the available sources.

    Parameters
    ----------
    config_path:
        Explicit TOML file. When omitted, :func:`default_config_path` is used
        if it exists.
    overrides:
        Highest-precedence values, applied last. Only top-level sections are
        merged; a section given here replaces the merged section.
    require_file:
        Raise when no configuration file is found instead of using defaults.
    """
    resolved: Path | None
    if config_path is not None:
        resolved = Path(config_path).expanduser()
        if not resolved.is_file():
            raise ConfigurationError("configuration file not found", path=str(resolved))
    else:
        resolved = default_config_path() if use_environment else None
        if require_file:
            raise ConfigurationError("no configuration file found")

    data = _read_toml(resolved) if resolved else {}
    if overrides:
        data = {**data, **overrides}

    try:
        if use_environment:
            # BaseSettings: TOML data < environment variables.
            return AppSettings(**data)
        # model_validate reads the mapping only, so the environment cannot
        # leak into a test or a reproducible run.
        return AppSettings.model_validate(data)
    except ValidationError as exc:
        raise ConfigurationError(
            "configuration validation failed",
            path=str(resolved) if resolved else "defaults",
            errors=exc.errors(include_url=False),
        ) from exc
