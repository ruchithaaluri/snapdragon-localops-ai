"""Structured logging: JSON for machines, readable text for humans.

Design points:

* Every record carries the run/session context (``run_id``, ``session_id``) so
  one diagnostic run can be filtered out of the log.
* Secrets are redacted **at the formatter**, not at the call site, so a
  forgotten field cannot leak an API key into a log file.
* No third-party logging dependency: the stdlib formatter is subclassed.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    "REDACTED",
    "ConsoleFormatter",
    "JsonFormatter",
    "Redactor",
]

#: Attributes present on every LogRecord; anything else was attached by us.
_STANDARD_ATTRS = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
        "levelname", "levelno", "lineno", "message", "module", "msecs", "msg", "name",
        "pathname", "process", "processName", "relativeCreated", "stack_info",
        "thread", "threadName", "taskName",
    }
)

#: Field-name patterns whose values are always masked.
_SECRET_KEY_PATTERN = re.compile(
    r"(api[_-]?key|secret|token|password|passwd|authorization|bearer|credential)",
    re.IGNORECASE,
)

#: Value patterns for inline secrets, e.g. "Bearer eyJhbGciOi..." or a key
#: pasted into free text.
_INLINE_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/=]{8,}"),
    re.compile(r"(?i)\b(?:sk|ghp|gho|xox[baprs])-[A-Za-z0-9\-_]{8,}"),
)

REDACTED = "***redacted***"


class Redactor:
    """Masks secret-looking values before they reach a handler.

    Both directions are covered: a field *named* like a secret, and a secret
    embedded in an otherwise normal string.
    """

    def __init__(self, extra_keys: Iterable[str] = ()) -> None:
        self._extra_keys = {k.lower() for k in extra_keys}

    def key_is_secret(self, key: str) -> bool:
        return key.lower() in self._extra_keys or bool(_SECRET_KEY_PATTERN.search(key))

    def value(self, key: str, value: Any) -> Any:
        """Redact one field."""
        if self.key_is_secret(key):
            return REDACTED
        return self.text(value) if isinstance(value, str) else value

    def text(self, text: str) -> str:
        """Redact inline secrets inside a string."""
        for pattern in _INLINE_SECRET_PATTERNS:
            text = pattern.sub(REDACTED, text)
        return text

    def mapping(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Redact every field of a mapping."""
        return {key: self.value(key, val) for key, val in data.items()}


class _BaseFormatter(logging.Formatter):
    """Shared extraction of the custom fields attached by :func:`log_event`."""

    def __init__(self, redactor: Redactor | None = None) -> None:
        super().__init__()
        self.redactor = redactor or Redactor()

    def extract(self, record: logging.LogRecord) -> dict[str, Any]:
        """Return the non-standard attributes of a record, redacted."""
        fields = {k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS}
        return self.redactor.mapping(fields)

    def format_message(self, record: logging.LogRecord) -> str:
        """Render the user message with inline secrets removed."""
        raw = record.getMessage()
        return self.redactor.text(raw)


class JsonFormatter(_BaseFormatter):
    """One JSON object per line: greppable, parseable, CI-friendly."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": self.format_message(record),
        }
        payload.update(self.extract(record))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        try:
            return json.dumps(payload, default=str, ensure_ascii=False)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return json.dumps({"ts": payload["ts"], "level": payload["level"],
                               "message": "<unserializable log record>"})


class ConsoleFormatter(_BaseFormatter):
    """Human-readable single line, e.g. for an interactive terminal."""

    #: Kept short so a run of records stays readable.
    default_format = "%(asctime)s %(levelname)-5s %(name)-28s %(message)s"

    def __init__(self, redactor: Redactor | None = None, fmt: str | None = None) -> None:
        super().__init__(redactor)
        self.fmt = fmt or self.default_format

    def format(self, record: logging.LogRecord) -> str:
        created = self.formatTime(record, "%H:%M:%S")
        base = f"{created} {record.levelname:<5} {record.name:<28} {self.format_message(record)}"
        fields = self.extract(record)
        if fields:
            rendered = " ".join(f"{k}={v}" for k, v in fields.items())
            base = f"{base} | {rendered}"
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base

