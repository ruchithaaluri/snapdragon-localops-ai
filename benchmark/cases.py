"""The default benchmark suite: what this machine actually does, measured.

Deliberately small. Four cases, ten iterations each, over a few hundred bytes of
text. The whole run takes well under a second and touches no model, no network,
and nothing on disk. A benchmark that is expensive to run is a benchmark that
does not get run before a demo, and one that is not run produces no numbers at
all.

What is measured is the work the application genuinely performs: extracting
evidence, running the diagnostic engine, searching the retrieval index, and the
end-to-end pipeline. Those are the latencies a user actually waits for.

What is **not** here: any NPU figure. This machine is not Snapdragon hardware,
so every row carries a verified CPU target and the report says so. Running the
same command on the Snapdragon target is how an NPU row becomes real;
:mod:`ai.backends.snapdragon` documents what has to happen first.
"""

from __future__ import annotations

import logging
import os
import platform
from collections.abc import Callable

from core.contracts.benchmark import BenchmarkCase, BenchmarkReport
from core.execution import ExecutionTarget
from core.observability import get_logger, log_event
from core.version import __version__

from benchmark.runner import LocalBenchmarkRunner, host_description

__all__ = [
    "SAMPLE_ERROR",
    "build_default_cases",
    "device_description",
    "run_default_benchmark",
]

_log = get_logger(__name__)

#: The text every case runs against: small, fixed, and the same failure the demo
#: produces, so a benchmark number describes the workload the demo performs.
SAMPLE_ERROR = (
    "Traceback (most recent call last):\n"
    '  File "sample_app.py", line 3, in <module>\n'
    "    import localops_sample\n"
    "ModuleNotFoundError: No module named 'localops_sample'\n"
)


def device_description() -> str:
    """Device facts recorded alongside every number.

    Machine, architecture, processor and Python version. On the Snapdragon
    target the processor line is what makes a measurement attributable to that
    device rather than to "a laptop".

    Every field is read defensively: ``platform.processor()`` returns an empty
    string or ``None`` on some Windows builds, and a benchmark that fails while
    describing its own host has produced no number at all.
    """
    return " | ".join(
        (
            f"{platform.system()} {platform.release()} {platform.machine()}".strip(),
            f"cpu={platform.processor() or _cpu_fallback()}",
            f"py{platform.python_version()} {platform.python_implementation()}",
        )
    )


def _cpu_fallback() -> str:
    """CPU name from the environment, for when :mod:`platform` will not say."""
    for variable in ("PROCESSOR_IDENTIFIER", "PROCESSOR_ARCHITECTURE"):
        value = os.environ.get(variable, "").strip()
        if value:
            return value
    return "unknown"


def build_default_cases(
    *, warmup: int = 1, iterations: int = 10
) -> tuple[BenchmarkCase, ...]:
    """The four cases the default report contains.

    The iteration counts are deliberately modest: enough for a stable median and
    a p95, small enough that the whole report is produced in under a second.
    """
    from diagnostics import RuleBasedDiagnosticEngine, build_query
    from diagnostics.evidence import extract_evidence

    query = build_query("why does the app fail?", SAMPLE_ERROR, source="sample.log")
    engine = RuleBasedDiagnosticEngine()

    cases: tuple[tuple[str, str, Callable[[], object]], ...] = (
        (
            "evidence.extract",
            "evidence extraction",
            lambda: extract_evidence(SAMPLE_ERROR, source="sample.log"),
        ),
        (
            "diagnosis.analyze",
            "rule-based diagnosis",
            lambda: engine.analyze(query),
        ),
        (
            "retrieval.search",
            "retrieval search",
            _retrieval_probe(query.question),
        ),
        (
            "pipeline.end_to_end",
            "end-to-end pipeline (no history)",
            _pipeline_probe(query),
        ),
    )
    return tuple(
        BenchmarkCase(
            case_id=case_id,
            name=name,
            operation=operation,
            unit="ms",
            warmup_iterations=warmup,
            iterations=iterations,
            tags=("local", "deterministic"),
        )
        for case_id, name, operation in cases
    )


def _retrieval_probe(question: str) -> Callable[[], object]:
    """A closure that searches a small in-memory index built from the demo KB."""
    from core.contracts.retrieval import SearchQuery
    from ai.backends.mock import MockEmbeddingBackend
    from retrieval import InMemoryVectorIndex, LocalRetriever

    from demo.scenario import KNOWLEDGE_BASE

    retriever = LocalRetriever(MockEmbeddingBackend(), InMemoryVectorIndex(dimension=64))
    retriever.index_documents(list(KNOWLEDGE_BASE))

    def search() -> object:
        return retriever.retrieve(SearchQuery(text=question, top_k=4))

    return search


def _pipeline_probe(query) -> Callable[[], object]:  # noqa: ANN001 - DiagnosticQuery
    """A closure that runs the full pipeline with persistence switched off.

    History is disabled so the measurement times the pipeline rather than
    SQLite, and so the benchmark writes nothing to the user's data directory.
    """
    from app.pipeline import build_pipeline
    from core.config.schema import AppSettings

    # Built through ``model_validate`` rather than attribute assignment: the
    # settings model validates on assignment, and mutating it afterwards leaves
    # the pipeline holding something other than a valid settings object.
    settings = AppSettings.model_validate({"storage": {"enabled": False}})
    pipeline = build_pipeline(settings, open_history=False)

    def run_pipeline() -> object:
        return pipeline.run(query)

    return run_pipeline

def run_default_benchmark(
    settings=None,  # noqa: ANN001 - AppSettings, optional
    *,
    warmup: int | None = None,
    iterations: int | None = None,
) -> BenchmarkReport:
    """Measure the default suite on this machine and return a real report.

    The execution target is a verified CPU run, because that is the truth about
    the machine the numbers came from. Nothing here can produce an NPU row: the
    target is constructed as CPU, and
    :class:`~core.contracts.benchmark.Measurement` would reject an unverified NPU
    claim even if it somehow arrived.
    """
    policy = settings.benchmark if settings is not None else None
    warm = warmup if warmup is not None else (policy.warmup_iterations if policy else 1)
    count = (
        iterations
        if iterations is not None
        else (policy.measured_iterations if policy else 10)
    )

    runner = LocalBenchmarkRunner(
        # Verified CPU: the harness runs real work in this process, and that is
        # exactly what the number describes.
        ExecutionTarget.cpu(
            "python",
            device=device_description(),
            runtime_version=__version__,
            detail="deterministic pipeline stages timed in-process",
        ),
        runtime=f"python {platform.python_version()} (app {__version__})",
    )
    report = runner.run(build_default_cases(warmup=warm, iterations=count))
    enriched = BenchmarkReport(
        report_id=report.report_id,
        measurements=report.measurements,
        host=host_description(),
        runtime=report.runtime,
        created_at=report.created_at,
        app_version=__version__,
        notes=report.notes + _context_notes(),
    )
    log_event(
        _log,
        logging.INFO,
        "benchmark.default_completed",
        measured=len(enriched.measurements),
        notes=len(enriched.notes),
    )
    return enriched


def _context_notes() -> tuple[str, ...]:
    """The honesty notes that travel with every default report."""
    from ai.backends.snapdragon import SNAPDRAGON_VALIDATION_STATUS

    return (
        f"Device: {device_description()}",
        "Execution target on every row: verified CPU run. This machine is not "
        "Snapdragon hardware, so no NPU figure is reported.",
        SNAPDRAGON_VALIDATION_STATUS,
    )

    settings = AppSettings.model_validate({})
    settings.storage.enabled = False
    pipeline = build_pipeline(settings, open_history=False)

    def run_pipeline() -> object:
        return pipeline.run(query)

    return run_pipeline

    return (
        f"{platform.system()} {platform.release()} {platform.machine()} | "
        f"cpu={platform.processor() or 'unknown'} | "
        f"py{platform.python_version()} {platform.python_implementation()}"
    )
