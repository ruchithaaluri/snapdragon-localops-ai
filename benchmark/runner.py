"""The local benchmark runner: time real work, record real samples.

How a measurement is produced here, in order:

1. run ``warmup_iterations`` of the case and throw the timings away, so the
   first-call cost of importing or lazily building something is not reported as
   the steady-state cost;
2. run ``iterations`` timed iterations with :func:`time.perf_counter_ns`, a
   monotonic clock, collecting one sample each;
3. compute statistics from *those samples* - never from a single number typed in
   somewhere, and never estimated.

The execution target is supplied by the caller and is recorded verbatim on every
row. When it is an NPU it must be a verified one;
:class:`~core.contracts.benchmark.Measurement` raises otherwise, so a CPU run
can never be filed under "NPU" even by mistake.
"""

from __future__ import annotations

import logging
import math
import platform
import statistics as std
from collections.abc import Sequence

from core.contracts.base import Capability, ProviderHealth, ProviderInfo
from core.contracts.benchmark import (
    BenchmarkCase,
    BenchmarkReport,
    BenchmarkRunner,
    Measurement,
    MeasurementProvenance,
    Statistic,
)
from core.execution import ExecutionTarget
from core.ids import new_id
from core.observability import get_logger, log_event
from core.timing import Stopwatch
from core.version import __version__

__all__ = ["LocalBenchmarkRunner", "host_description"]

_log = get_logger(__name__)


def host_description() -> str:
    """A stable, human-readable identifier for this machine."""
    node = platform.node() or "unknown-host"
    return f"{node} ({platform.system()} {platform.machine()}, py{platform.python_version()})"


def _statistics(samples: Sequence[float], min_samples: int) -> dict[Statistic, float]:
    """Aggregates computed from ``samples``; unsupported ones are omitted.

    Omitting a statistic is better than inventing one: a standard deviation from
    two samples is noise presented as precision.
    """
    values = list(samples)
    result: dict[Statistic, float] = {
        Statistic.MEAN: std.fmean(values),
        Statistic.MEDIAN: std.median(values),
        Statistic.MIN: min(values),
        Statistic.MAX: max(values),
    }
    if len(values) < min_samples:
        # Too few samples to say anything about spread or the tail. Reporting a
        # p95 from three samples would be precision the data cannot support.
        return result
    result[Statistic.STDEV] = std.stdev(values)
    # Nearest-rank p95: the smallest sample at or above 95% of the data.
    rank = max(1, math.ceil(0.95 * len(values)))
    ordered = sorted(values)
    result[Statistic.P95] = ordered[min(rank, len(ordered)) - 1]
    return result


class LocalBenchmarkRunner(BenchmarkRunner):
    """Times benchmark cases on this machine.

    Parameters
    ----------
    target:
        Where the work ran. Defaults to a verified CPU run, which is the honest
        description of timing pure-Python work on this process.
    runtime:
        Name recorded in the report's provenance, e.g. ``"onnxruntime"`` when a
        model-backed case is timed.
    min_samples_for_statistics:
        Below this many samples, the noisier aggregates are left out.
    model_id / artifact_sha256:
        Optional identification of the artifact a case exercised, recorded on
        every row so a number can be traced to the exact model that produced it.
    """

    kind = "benchmark"

    def __init__(
        self,
        target: ExecutionTarget | None = None,
        *,
        runtime: str = "python",
        min_samples_for_statistics: int = 5,
        model_id: str = "",
        artifact_sha256: str = "",
    ) -> None:
        if min_samples_for_statistics < 1:
            raise ValueError("min_samples_for_statistics must be >= 1")
        self._target = target or ExecutionTarget.cpu(
            "python", detail="benchmark harness running in this process"
        )
        self._runtime = runtime
        self._min_samples = min_samples_for_statistics
        self._model_id = model_id
        self._sha256 = artifact_sha256

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="local-benchmark",
            kind=self.kind,
            version=__version__,
            capabilities=frozenset({Capability.BENCHMARK_MEASUREMENT}),
            detail=f"monotonic timing via {self._target.describe()}",
        )

    def health(self) -> ProviderHealth:
        return ProviderHealth.ready("measurement only; no state is modified")

    def run(self, cases: Sequence[BenchmarkCase]) -> BenchmarkReport:
        """Measure every case. A case that raises is noted, never dropped."""
        notes: list[str] = []
        measurements: list[Measurement] = []

        for case in cases:
            try:
                samples = self._time(case)
            except Exception as exc:  # noqa: BLE001 - a partial run must be visible
                detail = f"{case.name}: {type(exc).__name__}: {exc}"
                notes.append(f"NOT MEASURED - {detail}")
                log_event(_log, logging.ERROR, "benchmark.case_failed", case=case.name,
                          error=type(exc).__name__)
                continue
            if not samples:
                notes.append(f"NOT MEASURED - {case.name}: produced no samples")
                continue
            measurements.append(self._measurement(case, samples))

        report = BenchmarkReport(
            report_id=new_id("bench"),
            measurements=tuple(measurements),
            host=host_description(),
            runtime=self._runtime,
            app_version=__version__,
            notes=tuple(notes),
        )
        log_event(
            _log,
            logging.INFO,
            "benchmark.completed",
            cases=len(cases),
            measured=len(measurements),
            failed=len(notes),
        )
        return report

    # -- internals --------------------------------------------------------

    def _time(self, case: BenchmarkCase) -> list[float]:
        """Warm up, then time each iteration with a monotonic clock."""
        for _ in range(case.warmup_iterations):
            case.operation()

        samples: list[float] = []
        for _ in range(case.iterations):
            watch = Stopwatch().start()
            case.operation()
            samples.append(watch.stop().milliseconds)
        return samples

    def _measurement(self, case: BenchmarkCase, samples: Sequence[float]) -> Measurement:
        """Turn real samples into a row, with full provenance attached."""
        provenance = MeasurementProvenance(
            method="monotonic wall-clock timing (perf_counter_ns)",
            host=host_description(),
            runtime=self._runtime,
            iterations=len(samples),
            warmup_iterations=case.warmup_iterations,
            model_id=self._model_id,
            artifact_sha256=self._sha256,
        )
        return Measurement(
            name=case.name,
            unit=case.unit,
            value=min(samples),
            samples=tuple(samples),
            provenance=provenance,
            target=self._target,
            statistics=_statistics(samples, self._min_samples),
        )
