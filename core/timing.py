"""Timing utilities.

Every latency reported by a provider or a benchmark goes through
:class:`Stopwatch`, which uses :func:`time.perf_counter_ns` (monotonic, highest
available resolution) and never the wall clock.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

__all__ = ["Elapsed", "Stopwatch", "measure"]


@dataclass(frozen=True)
class Elapsed:
    """Result of a timed block.

    ``iterations`` is 0 for a stopwatch that has never been started and 1 after
    a single run.
    """

    milliseconds: float
    seconds: float
    iterations: int

    def __post_init__(self) -> None:
        if self.iterations < 0:
            raise ValueError("iterations must not be negative")
        if self.milliseconds < 0.0:
            raise ValueError("elapsed time must not be negative")


def _to_elapsed(ns: int, iterations: int) -> Elapsed:
    seconds = ns / 1_000_000_000
    return Elapsed(milliseconds=ns / 1_000_000, seconds=seconds, iterations=iterations)


def measure() -> "_StopwatchContext":
    """Context manager measuring wall time of a block.

    >>> with measure() as timer:            # doctest: +SKIP
    ...     do_work()
    >>> timer.elapsed.milliseconds > 0      # doctest: +SKIP
    True
    """
    return _StopwatchContext()


class Stopwatch:
    """Reusable monotonic stopwatch.

    >>> watch = Stopwatch()
    >>> _ = watch.elapsed.milliseconds
    >>> watch.elapsed.iterations
    0
    """

    __slots__ = ("_start_ns", "_elapsed")

    def __init__(self) -> None:
        self._start_ns: int | None = None
        self._elapsed: Elapsed = Elapsed(0.0, 0.0, 0)

    def start(self) -> "Stopwatch":
        """Start (or restart) the stopwatch."""
        self._start_ns = time.perf_counter_ns()
        return self

    def stop(self) -> Elapsed:
        """Stop the stopwatch and return the accumulated time."""
        if self._start_ns is None:
            raise RuntimeError("Stopwatch.stop() called before start()")
        ns = time.perf_counter_ns() - self._start_ns
        self._start_ns = None
        self._elapsed = _to_elapsed(ns, 1)
        return self._elapsed

    @property
    def running(self) -> bool:
        """True between :meth:`start` and :meth:`stop`."""
        return self._start_ns is not None

    @property
    def elapsed(self) -> Elapsed:
        """Time so far. Live while running, frozen after :meth:`stop`."""
        if self._start_ns is not None:
            return _to_elapsed(time.perf_counter_ns() - self._start_ns, 1)
        return self._elapsed

    @property
    def milliseconds(self) -> float:
        """Convenience accessor for the elapsed milliseconds."""
        return self.elapsed.milliseconds


class _StopwatchContext:
    """Context manager returned by :func:`measure`."""

    __slots__ = ("_watch", "_elapsed")

    def __init__(self) -> None:
        self._watch = Stopwatch()
        self._elapsed: Elapsed | None = None

    def __enter__(self) -> Stopwatch:
        return self._watch.start()

    def __exit__(self, *_exc: object) -> None:
        self._elapsed = self._watch.stop()

    @property
    def elapsed(self) -> Elapsed:
        if self._elapsed is None:
            raise RuntimeError("measure() block has not finished")
        return self._elapsed

    @property
    def milliseconds(self) -> float:
        """Convenience accessor for the elapsed milliseconds."""
        return self.elapsed.milliseconds
