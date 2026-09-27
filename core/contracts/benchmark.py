"""Benchmark contracts: measured numbers only.

The single hardest rule in this project is that no number appears in a report
unless it was measured on the machine named in the report. This module enforces
that structurally:

* :class:`Measurement` refuses to be constructed without provenance (what was
  run, how often, with which model, on which host),
* :class:`Measurement` refuses to record an NPU row whose execution was not
  verified,
* there is no API for "entering a number by hand".
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.contracts.base import Capability, Component
from core.errors import UnmeasuredBenchmarkError
from core.execution import AcceleratorKind, ExecutionTarget
from core.ids import new_id, utc_now

__all__ = [
    "BenchmarkCase",
    "BenchmarkReport",
    "BenchmarkRunner",
    "Measurement",
    "MeasurementProvenance",
    "Statistic",
]


class Statistic(str, Enum):
    """Aggregates computed from the raw sample list."""

    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    P95 = "p95"
    STDEV = "stdev"


@dataclass(frozen=True, slots=True)
class MeasurementProvenance:
    """Everything needed to reproduce a number."""

    method: str
    host: str
    runtime: str
    iterations: int
    warmup_iterations: int = 0
    model_id: str = ""
    artifact_sha256: str = ""
    command: str = ""
    measured_at: str = field(default_factory=lambda: utc_now().isoformat(timespec="seconds"))

    def __post_init__(self) -> None:
        if not self.method.strip():
            raise ValueError("a measurement must name the method that produced it")
        if self.iterations < 1:
            raise ValueError("a measurement needs at least one iteration")
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations must not be negative")


@dataclass(frozen=True, slots=True)
class Measurement:
    """One measured metric with its full provenance."""

    name: str
    unit: str
    value: float
    samples: tuple[float, ...]
    provenance: MeasurementProvenance
    target: ExecutionTarget
    statistics: dict[Statistic, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a measurement needs a name")
        if not self.samples:
            raise UnmeasuredBenchmarkError(
                "a measurement must carry at least one real sample", metric=self.name
            )
        if not self.provenance.method.strip():
            raise UnmeasuredBenchmarkError(
                "a measurement without a method is not a measurement", metric=self.name
            )
        if self.target.kind is AcceleratorKind.NPU and not self.target.is_verified:
            raise UnmeasuredBenchmarkError(
                "refusing to record an NPU row without a verified run",
                metric=self.name,
                state=self.target.verification.state.value,
            )

    @property
    def iteration_count(self) -> int:
        return len(self.samples)

    def statistic(self, stat: Statistic) -> float:
        """Return a computed aggregate, or the headline value when absent."""
        return self.statistics.get(stat, self.value)

    def as_row(self) -> tuple[str, str, str, str]:
        """(metric, value, target, provenance) for a table."""
        target = f"{self.target.kind.value} ({self.target.verification.state.value})"
        return (
            self.name,
            f"{self.value:.3f} {self.unit}".strip(),
            target,
            f"{self.provenance.method} x{self.iteration_count}",
        )


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """A named operation to measure.

    ``operation`` is excluded from ``repr`` and comparison so a report can be
    logged without dragging a closure along.
    """

    case_id: str
    name: str
    operation: Callable[[], Any] = field(repr=False, compare=False)
    unit: str = "ms"
    warmup_iterations: int = 1
    iterations: int = 3
    tags: tuple[str, ...] = field(default=())
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("a benchmark case needs a case_id")
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations must not be negative")


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Measured results for one host, with everything needed to audit them."""

    report_id: str
    measurements: tuple[Measurement, ...]
    host: str
    runtime: str
    created_at: str = field(default_factory=lambda: utc_now().isoformat(timespec="seconds"))
    app_version: str = ""
    notes: tuple[str, ...] = field(default=())

    @property
    def is_empty(self) -> bool:
        return not self.measurements

    @property
    def npu_measurements(self) -> tuple[Measurement, ...]:
        """Rows verified to have run on an NPU."""
        return tuple(m for m in self.measurements if m.target.npu_execution_claimed)

    def by_target(self) -> dict[str, tuple[Measurement, ...]]:
        """Group measurements by execution target kind."""
        grouped: dict[str, list[Measurement]] = {}
        for measurement in self.measurements:
            grouped.setdefault(measurement.target.kind.value, []).append(measurement)
        return {key: tuple(value) for key, value in grouped.items()}

    def to_markdown(self) -> str:
        """Render as Markdown. An empty report says so explicitly."""
        lines = [
            f"# Benchmark report {self.report_id}",
            "",
            f"- Host: {self.host}",
            f"- Runtime: {self.runtime}",
            f"- App version: {self.app_version or 'unknown'}",
            f"- Created: {self.created_at}",
            "",
        ]
        if self.is_empty:
            lines.append("No measurements were taken in this run.")
            return "\n".join(lines) + "\n"
        lines += ["| Metric | Value | Execution | Method |", "| --- | --- | --- | --- |"]
        lines += ["| " + " | ".join(m.as_row()) + " |" for m in self.measurements]
        if self.notes:
            lines += ["", "## Notes", *[f"- {note}" for note in self.notes]]
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict[str, Any]:
        """Serialisable representation for JSON export."""
        return {
            "report_id": self.report_id,
            "created_at": self.created_at,
            "host": self.host,
            "runtime": self.runtime,
            "app_version": self.app_version,
            "notes": list(self.notes),
            "measurements": [
                {
                    "name": m.name,
                    "unit": m.unit,
                    "value": m.value,
                    "samples": list(m.samples),
                    "iterations": m.iteration_count,
                    "target": {
                        "kind": m.target.kind.value,
                        "provider": m.target.provider,
                        "verification": m.target.verification.state.value,
                        "method": m.target.verification.method,
                    },
                    "provenance": {
                        "method": m.provenance.method,
                        "host": m.provenance.host,
                        "runtime": m.provenance.runtime,
                        "model_id": m.provenance.model_id,
                        "artifact_sha256": m.provenance.artifact_sha256,
                        "measured_at": m.provenance.measured_at,
                    },
                }
                for m in self.measurements
            ],
        }


class BenchmarkRunner(Component):
    """Runs benchmark cases and produces a measured report."""

    kind = "benchmark"

    @abstractmethod
    def run(self, cases: Sequence[BenchmarkCase]) -> BenchmarkReport:
        """Measure every case and return a report.

        A case that raises is recorded as a failure rather than silently
        dropped, so a partial report can never look like a complete one.
        """

    def preferred_capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.BENCHMARK_MEASUREMENT})

