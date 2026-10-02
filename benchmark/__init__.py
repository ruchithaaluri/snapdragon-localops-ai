"""Benchmarking: numbers that were actually measured, or no numbers at all.

This package exists because the hardest rule in the project is that a reported
number must have been produced on the machine named in the report. The data
model in :mod:`core.contracts.benchmark` enforces most of that;
:mod:`benchmark.runner` supplies the missing half by only ever constructing a
:class:`~core.contracts.benchmark.Measurement` from real, timed samples.

There is deliberately no way to type a number in. If a case fails, the failure
becomes a note in the report - never a zero, never an omission, and never a
silently shortened list of cases.

:func:`run_default_benchmark` is the suite the CLI and the UI run: four cases
over a few hundred bytes of text, finishing in well under a second, measuring
the work a user actually waits for.
"""

from __future__ import annotations

from benchmark.cases import (
    SAMPLE_ERROR,
    build_default_cases,
    device_description,
    run_default_benchmark,
)
from benchmark.runner import LocalBenchmarkRunner, host_description

__all__ = [
    "SAMPLE_ERROR",
    "LocalBenchmarkRunner",
    "build_default_cases",
    "device_description",
    "host_description",
    "run_default_benchmark",
]
