"""The main window: input on the left, stage results in tabs on the right.

The window owns no diagnostic logic. It collects input, asks
:class:`~app.pipeline.LocalDiagnosticPipeline` to do the work on a worker
thread, and renders whatever comes back - including "I do not know", which is a
result the engine is allowed to return.

Running the pipeline off the GUI thread is not an optimisation: a run reads
files and runs subprocesses, and doing that on the UI thread freezes the window
for as long as it takes.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.pipeline import LocalDiagnosticPipeline, build_pipeline
from app.ui.panels import (
    BenchmarkPanel,
    DemoPanel,
    DiagnosisPanel,
    EnvironmentPanel,
    EvidencePanel,
    InputPanel,
    RemediationPanel,
    VerificationPanel,
)
from app.ui.theme import Theme, stylesheet_for
from app.ui.widgets import StatusStrip, app_status
from core.contracts.base import ProviderInfo
from core.contracts.diagnostics import DiagnosticQuery
from core.contracts.pipeline import PipelineRun
from core.errors import LocalOpsError
from core.observability import get_logger, log_event
from core.safety import ApprovalDecision
from core.version import __version__
from diagnostics import build_query
from verification import checks_for_plan

__all__ = ["MainWindow", "run_gui"]

_log = get_logger(__name__)


class _AnalysisWorker(QObject):
    """Runs the pipeline on a worker thread and reports back by signal."""

    finished = Signal(object)   # PipelineRun
    failed = Signal(str)        # human-readable reason

    def __init__(self, pipeline: LocalDiagnosticPipeline) -> None:
        super().__init__()
        self._pipeline = pipeline

    def run(self, query: DiagnosticQuery) -> None:
        try:
            self.finished.emit(self._pipeline.run(query))
        except LocalOpsError as exc:
            log_event(_log, logging.ERROR, "ui.analysis_failed", error=str(exc))
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - the UI must not hang on a bug
            log_event(
                _log,
                logging.ERROR,
                "ui.analysis_crashed",
                error=f"{type(exc).__name__}: {exc}",
            )
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class _BenchmarkWorker(QObject):
    """Runs the measurement suite off the UI thread.

    The suite is cheap but not instant, and it starts Python subprocesses via
    the end-to-end case, so it must not run on the UI thread.
    """

    finished = Signal(object)   # BenchmarkReport
    failed = Signal(str)

    def __init__(self, settings) -> None:  # noqa: ANN001 - AppSettings
        super().__init__()
        self._settings = settings

    def run(self) -> None:
        from benchmark import run_default_benchmark

        try:
            self.finished.emit(run_default_benchmark(self._settings))
        except Exception as exc:  # noqa: BLE001 - a failed measurement is not a crash
            log_event(
                _log,
                logging.ERROR,
                "ui.benchmark_failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class MainWindow(QMainWindow):
    """The application window."""

    def __init__(self, settings) -> None:  # noqa: ANN001 - AppSettings
        super().__init__()
        self._settings = settings
        self._theme = Theme(settings.ui.theme)  # type: ignore[arg-type]
        self._pipeline = build_pipeline(settings)
        self._last_run: PipelineRun | None = None
        self._thread: QThread | None = None
        self._worker: _AnalysisWorker | None = None
        self._bench_thread: QThread | None = None
        self._bench_worker: "_BenchmarkWorker | None" = None

        # The AI backend is resolved once, here, so the status strip can report
        # it without re-instantiating it. A backend that cannot be built (a
        # typo in configuration) must not stop the window from opening.
        try:
            from ai.registry import build_ai_backend

            backend = build_ai_backend(settings)
            self._ai_info, self._ai_is_mock = backend.info, backend.is_mock()
        except Exception as exc:  # noqa: BLE001 - a bad backend must not block start-up
            log_event(
                _log, logging.WARNING, "ui.ai_backend_unavailable", error=str(exc)
            )
            self._ai_info = ProviderInfo(
                name="unavailable", kind="ai_backend", detail=str(exc)
            )
            self._ai_is_mock = True

        self.setWindowTitle(f"Snapdragon LocalOps AI  -  v{__version__}")
        self.resize(settings.ui.window_width, settings.ui.window_height)
        self._build()
        self._connect()
        self.environment.refresh(settings, self._pipeline)
        self.statusBar().showMessage("Ready. Nothing has been run yet.")

    # -- construction -----------------------------------------------------

    def _build(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)

        # The status strip sits above everything: LOCAL MODE, MOCK AI / REAL AI,
        # and the accelerator, each as a word rather than a colour alone.
        self._status_strip = QHBoxLayout()
        self._status_strip.setSpacing(6)
        layout.addLayout(self._status_strip)

        self.input = InputPanel()
        self.evidence = EvidencePanel()
        self.diagnosis = DiagnosisPanel(self._theme)
        self.remediation = RemediationPanel(self._theme)
        self.verification = VerificationPanel(self._theme)
        self.environment = EnvironmentPanel(self._theme)
        self.demo = DemoPanel(self._theme, self._settings)
        self.benchmark = BenchmarkPanel(self._theme)

        self._tabs = QTabWidget()
        self._tabs.addTab(self.diagnosis, "Diagnosis")
        self._tabs.addTab(self.evidence, "Evidence")
        self._tabs.addTab(self.remediation, "Suggested fixes")
        self._tabs.addTab(self.verification, "Verification")
        self._tabs.addTab(self.demo, "Demo")
        self._tabs.addTab(self.benchmark, "Benchmark")
        self._tabs.addTab(self.environment, "This machine")

        self._stage_bar = QLabel()
        self._stage_bar.setWordWrap(True)
        self._stage_bar.setStyleSheet(f"color: {self._theme.text_dim};")

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.input)
        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 900])

        body = QHBoxLayout()
        body.addWidget(splitter)
        layout.addLayout(body, 1)
        layout.addWidget(self._stage_bar)
        self.setCentralWidget(central)
        self._refresh_status_strip()

    def _refresh_status_strip(self, target=None) -> None:  # noqa: ANN001 - ExecutionTarget
        """Rebuild the indicator pills from live component state.

        ``MOCK AI`` / ``REAL AI`` is read from the backend, and the accelerator
        word is only ``NPU`` for a verified run. Both are resolved by
        :mod:`app.ui.widgets.status_strip`, which has no Qt dependency and is
        therefore testable without a display.
        """
        while self._status_strip.count():
            item = self._status_strip.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        status = app_status(self._ai_info, self._ai_is_mock, target)
        for pill in StatusStrip.pills(self._theme, status):
            self._status_strip.addWidget(pill)
        self._status_strip.addStretch(1)
        self._backend_label = QLabel(f"backend: {status.backend}")
        self._backend_label.setStyleSheet(f"color: {self._theme.text_dim};")
        self._status_strip.addWidget(self._backend_label)

    def _connect(self) -> None:
        self.input.analyze_requested.connect(self.start_analysis)
        self.remediation.approved.connect(self._on_approved)
        self.remediation.denied.connect(self._on_denied)
        self.remediation.apply_requested.connect(self._on_apply)
        self.verification.run_requested.connect(self._on_run_checks)
        self.benchmark.measure_requested.connect(self._on_measure)

    # -- benchmark --------------------------------------------------------

    def _on_measure(self) -> None:
        """Run the measurement suite on a worker thread."""
        if self._bench_thread is not None and self._bench_thread.isRunning():
            return
        self.benchmark.set_busy(True)
        self.statusBar().showMessage("Measuring this machine...")

        self._bench_thread = QThread(self)
        self._bench_worker = _BenchmarkWorker(self._settings)
        self._bench_worker.moveToThread(self._bench_thread)
        self._bench_thread.started.connect(self._bench_worker.run)
        self._bench_worker.finished.connect(self._on_benchmark_finished)
        self._bench_worker.failed.connect(self._on_benchmark_failed)
        self._bench_thread.start()

    def _on_benchmark_finished(self, report) -> None:  # noqa: ANN001 - BenchmarkReport
        self._teardown_bench()
        self.benchmark.show_report(report)
        self.statusBar().showMessage(
            f"Benchmark: {len(report.measurements)} measurement(s) on {report.host}"
        )

    def _on_benchmark_failed(self, reason: str) -> None:
        self._teardown_bench()
        self.statusBar().showMessage("Benchmark failed")
        QMessageBox.warning(self, "Benchmark failed", reason)

    def _teardown_bench(self) -> None:
        if self._bench_thread is not None:
            self._bench_thread.quit()
            self._bench_thread.wait(3000)
        self._bench_thread = None
        self._bench_worker = None
        self.benchmark.set_busy(False)

    # -- analysis ---------------------------------------------------------

    def start_analysis(self) -> None:
        """Kick off a run on a worker thread."""
        if self._thread is not None and self._thread.isRunning():
            return
        query = build_query(
            self.input.question(),
            self.input.text(),
            source=self.input.source(),
        )
        self.input.set_busy(True)
        self.statusBar().showMessage("Analysing...")
        self._stage_bar.setText(
            "Running capture -> evidence -> diagnosis -> remediation..."
        )

        self._thread = QThread(self)
        self._worker = _AnalysisWorker(self._pipeline)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(lambda: self._worker.run(query))
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    def _on_finished(self, run: PipelineRun) -> None:
        self._last_run = run
        self.input.set_busy(False)
        self.evidence.show_evidence(run.query.evidence)
        self.diagnosis.show_diagnosis(run.diagnosis)
        self.remediation.show_plan(run.plan)
        self.verification.clear()
        if run.diagnosis and run.diagnosis.provenance:
            self.environment.show_target(run.diagnosis.provenance.execution)
            # The accelerator indicator follows the run that actually happened,
            # so it reads CPU until a verified NPU run has been recorded.
            self._refresh_status_strip(run.diagnosis.provenance.execution)

        self._stage_bar.setText(
            "  |  ".join(f"{stage}: {status}" for stage, status, _ in run.as_rows())
        )
        self.statusBar().showMessage(
            f"Run {run.run_id} - {run.outcome.value.replace('_', ' ')}"
        )
        if run.is_awaiting_approval:
            self._tabs.setCurrentWidget(self.remediation)
        if run.remediation_results:
            self.remediation.show_results(run.remediation_results)
        self._teardown_thread()

    def _on_failed(self, reason: str) -> None:
        self.input.set_busy(False)
        self._stage_bar.setText("The run did not complete.")
        self.statusBar().showMessage("Analysis failed")
        QMessageBox.critical(self, "Analysis failed", reason)
        self._teardown_thread()

    def _teardown_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
        self._thread = None
        self._worker = None

    # -- approvals --------------------------------------------------------

    def _on_approved(self, action_id: str) -> None:
        """Record the user's decision. This runs nothing by itself."""
        self._pipeline.gate.decide(
            _action_for(self._last_run, action_id),
            ApprovalDecision.APPROVED,
            note="approved in the desktop shell",
        )
        self.statusBar().showMessage(
            f"Action {action_id} approved. Nothing has run yet - use "
            "'Run the approved actions'."
        )

    def _on_denied(self, action_id: str) -> None:
        self._pipeline.gate.decide(
            _action_for(self._last_run, action_id),
            ApprovalDecision.DENIED,
            note="denied in the desktop shell",
        )
        self.statusBar().showMessage(f"Action {action_id} denied.")

    def _on_apply(self) -> None:
        """Apply the approvals recorded against the plan already on screen.

        The plan is *not* regenerated: its action ids are what the user
        approved, so the same actions are what run.
        """
        run = self._last_run
        if run is None or run.plan is None:
            return
        approvals = tuple(
            record
            for action in run.plan.actions
            if (record := self._pipeline.gate.get(action.action_id)) is not None
        )
        if not approvals:
            self.statusBar().showMessage("Nothing has been approved yet.")
            return

        results = self._pipeline.apply_plan(run.plan, approvals)
        self.remediation.show_results(results)
        applied = sum(1 for r in results if r.succeeded)
        self.statusBar().showMessage(
            f"{applied} of {len(results)} approved action(s) applied."
        )

        # Re-ask the questions the diagnosis raised, on the actions that ran.
        applied_actions = [
            action
            for action in run.plan.actions
            if any(r.action_id == action.action_id and r.succeeded for r in results)
        ]
        checks = checks_for_plan(applied_actions, run.query.evidence)
        if checks:
            self._pending_checks = checks
            self.verification.show_checks(checks)
            self._tabs.setCurrentWidget(self.verification)
        else:
            self._pending_checks = ()
            self.verification.clear(
                "No check could be derived from this run.\n\n"
                "Verification re-reads a file or re-runs a command. When the "
                "evidence was pasted rather than loaded from disk, and no action "
                "carried a command, there is nothing honest to re-test."
            )

    def _on_run_checks(self) -> None:
        """Approve the displayed checks, run them, and show the report."""
        checks = self.verification.checks()
        if not checks:
            return
        for check in checks:
            # The gate is asked about each check by id, so approving here is a
            # per-check decision, not a blanket "run everything".
            self._pipeline.gate.approve(_check_as_action(check), note="approved in the shell")
        report = self._pipeline.verify(self._last_run, checks)
        self.verification.show_report(report)
        self.statusBar().showMessage(f"Verification: {report.summary()}")

    def closeEvent(self, event) -> None:  # noqa: ANN001 - Qt event
        self._teardown_thread()
        self._teardown_bench()
        super().closeEvent(event)


def _check_as_action(check):
    """View a check as an action so the approval gate can be asked about it.

    A check is never executed as a remediation action; this mirrors
    :func:`verification.runner._as_action`.
    """
    from core.safety import RemediationAction

    return RemediationAction(
        action_id=check.check_id,
        title=check.description,
        description=check.description,
        risk=check.risk,
        command=check.parameters.get("command"),
    )


def _action_for(run: PipelineRun | None, action_id: str):
    """Find an action in the last plan, or fail loudly rather than guess."""
    if run is None or run.plan is None:
        raise LocalOpsError("no plan is loaded, so there is nothing to decide about")
    return run.plan.action(action_id)


def run_gui(settings) -> int:
    """Start the Qt event loop. Returns the process exit code."""
    app = QApplication.instance() or QApplication([])
    theme = Theme(settings.ui.theme)  # type: ignore[arg-type]
    app.setStyleSheet(stylesheet_for(theme))

    window = MainWindow(settings)
    window.show()
    return app.exec()
