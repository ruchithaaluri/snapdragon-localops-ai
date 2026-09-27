"""Ambient run/session context for log records.

Uses :mod:`contextvars` so a run id set in a worker thread does not leak into
the main thread, and so the UI can run several runs concurrently in one
process without cross-contaminating their log lines.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Iterator

from core.ids import new_run_id, new_session_id

__all__ = ["LogContext", "current_context", "set_session_id", "use_context"]

_SESSION_ID: ContextVar[str] = ContextVar("localops_session_id", default="")
_RUN_ID: ContextVar[str] = ContextVar("localops_run_id", default="")


@dataclass(frozen=True, slots=True)
class LogContext:
    """Identifiers attached to every record emitted inside the context."""

    session_id: str = ""
    run_id: str = ""
    component: str = ""

    def as_fields(self) -> dict[str, str]:
        """Render the context as log fields, omitting empty values."""
        fields = {"session_id": self.session_id, "run_id": self.run_id}
        if self.component:
            fields["component"] = self.component
        return {k: v for k, v in fields.items() if v}


def set_session_id(session_id: str | None = None) -> str:
    """Bind the process-wide session id, generating one when not supplied."""
    value = session_id or new_session_id()
    _SESSION_ID.set(value)
    return value


def current_context() -> LogContext:
    """Return the context active on this thread/task."""
    return LogContext(session_id=_SESSION_ID.get(), run_id=_RUN_ID.get())


@contextmanager
def use_context(
    *,
    run_id: str | None = None,
    component: str = "",
    session_id: str | None = None,
) -> Iterator[LogContext]:
    """Bind a run id (and optional component) for the duration of the block.

    >>> with use_context(run_id="run_demo", component="diagnostics") as ctx:
    ...     ctx.run_id
    'run_demo'
    """
    tokens: list[tuple[ContextVar[str], Token[str]]] = []
    if session_id is not None:
        tokens.append((_SESSION_ID, _SESSION_ID.set(session_id)))
    resolved_run = run_id or new_run_id()
    tokens.append((_RUN_ID, _RUN_ID.set(resolved_run)))
    try:
        yield LogContext(session_id=_SESSION_ID.get(), run_id=resolved_run, component=component)
    finally:
        for var, token in reversed(tokens):
            var.reset(token)
