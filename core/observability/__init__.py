"""Public logging API: :func:`configure_logging`, :func:`get_logger`, helpers."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

from core.observability.context import LogContext, current_context, set_session_id, use_context
from core.observability.formatters import (
    REDACTED,
    ConsoleFormatter,
    JsonFormatter,
    Redactor,
)

__all__ = [
    "REDACTED",
    "ConsoleFormatter",
    "JsonFormatter",
    "LogContext",
    "Redactor",
    "configure_logging",
    "current_context",
    "get_logger",
    "log_event",
    "set_session_id",
    "shutdown_logging",
    "use_context",
]

_ROOT_LOGGER_NAME = "localops"
_CONFIGURED = False

#: Third-party chatter that is not useful for a developer running a local tool.
_NOISY_LOGGERS = ("PIL", "urllib3", "matplotlib", "shiboken6", "qt.*")


class _SafeStreamHandler(logging.StreamHandler):
    """A stream handler that tolerates a closed underlying stream.

    The console stream can disappear underneath a long-running process - a test
    harness that replaces ``sys.stderr``, a GUI app closing its console window on
    shutdown. The stdlib handler raises ``ValueError`` from :meth:`flush` in that
    state, which turns an orderly shutdown into a crash. Losing a log line is
    acceptable; refusing to exit is not.
    """

    def flush(self) -> None:
        try:
            super().flush()
        except ValueError:
            # The stream is closed; drop the reference so later emits no-op.
            self.stream = None  # type: ignore[assignment]


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the application root.

    ``get_logger(__name__)`` with a ``__main__`` module name would escape the
    hierarchy, so bare names are normalised into the ``localops`` namespace.
    """
    if name.startswith(_ROOT_LOGGER_NAME) or name == _ROOT_LOGGER_NAME:
        return logging.getLogger(name)
    suffix = name.removeprefix("__main__").lstrip(".") or "app"
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{suffix}")


def log_event(logger: logging.Logger, level: int, event: str, /, **fields: Any) -> None:
    """Emit a structured event.

    ``event`` becomes the message; keyword arguments become redacted fields.

    >>> log = get_logger("test")
    >>> log_event(log, logging.INFO, "run.started", stage="ocr")  # doctest: +SKIP
    """
    context = current_context()
    extras = {**context.as_fields(), **fields}
    logger.log(level, event, extra={"event": event, "fields": extras}, stacklevel=3)


def _build_file_handler(
    path: Path,
    *,
    level: int,
    json_output: bool,
    max_bytes: int,
    backup_count: int,
    redactor: Redactor,
) -> logging.Handler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler: logging.Handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    handler.setLevel(level)
    handler.setFormatter(
        JsonFormatter(redactor) if json_output else ConsoleFormatter(redactor)
    )
    return handler


def configure_logging(
    *,
    level: int | str = logging.INFO,
    log_file: Path | None = None,
    json_console: bool = False,
    json_file: bool = True,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5,
    redact_keys: tuple[str, ...] = (),
    quiet_third_party: bool = True,
    force: bool = False,
) -> logging.Logger:
    """Install the application log handlers and return the root logger.

    Safe to call more than once: handlers are replaced rather than stacked, so
    repeated calls (as in tests) cannot duplicate output.

    Parameters
    ----------
    level:
        Minimum level for the console handler.
    log_file:
        When given, a rotating JSON file handler is added.
    json_console:
        Emit JSON on the console instead of human-readable text.
    """
    global _CONFIGURED
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    resolved_level = logging.getLevelName(level) if isinstance(level, str) else level
    if not isinstance(resolved_level, int):
        raise ValueError(f"unknown log level: {level!r}")

    if _CONFIGURED and not force:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()

    root.setLevel(min(resolved_level, logging.DEBUG))
    root.propagate = False
    redactor = Redactor(redact_keys)

    console = _SafeStreamHandler(stream=sys.stderr)
    console.setLevel(resolved_level)
    console.setFormatter(JsonFormatter(redactor) if json_console else ConsoleFormatter(redactor))
    root.addHandler(console)

    if log_file is not None:
        root.addHandler(
            _build_file_handler(
                Path(log_file),
                level=logging.DEBUG,
                json_output=json_file,
                max_bytes=max_bytes,
                backup_count=backup_count,
                redactor=redactor,
            )
        )

    if quiet_third_party:
        for noisy in _NOISY_LOGGERS:
            logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    return root


def shutdown_logging() -> None:
    """Flush and detach every handler. Used by tests and at shutdown."""
    global _CONFIGURED
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    for handler in list(root.handlers):
        try:
            handler.flush()
        finally:
            root.removeHandler(handler)
            handler.close()
    _CONFIGURED = False
