"""The demo panel: one button, and the whole workflow on screen.

The panel shows the demo's own result rather than a scripted animation. Every
row is a value the run actually produced, and the two that matter - BEFORE and
AFTER - are read from a subprocess exit code, so the display cannot disagree
with what happened.

The button runs the scenario on a worker thread for the same reason the main
analysis does: the demo starts a real Python process twice, and doing that on
the UI thread would freeze the window.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.ui.theme import Theme
from app.ui.widgets import StatusPill
from core.observability import get_logger, log_event

__all__ = ["DemoPanel"]

_log = get_logger(__name__)

_HEADERS = ("Step", "Result")

#: Rows whose value is a pass/fail verdict and gets coloured accordingly.
_VERDICT_ROWS = frozenset({"BEFORE", "AFTER"})


class _DemoWorker(QObject):
    """Runs the demo off the UI thread and reports back by signal."""

    finished = Signal(object)   # DemoResult
    failed = Signal(str)

    def __init__(self, settings) -> None:  # noqa: ANN001 - AppSettings
        super().__init__()
        self._settings = settings

    def run(self) -> None:
        from demo import run_demo

        try:
            self.finished.emit(run_demo(self._settings, approve=True))
        except Exception as exc:  # noqa: BLE001 - a failed demo must not crash the app
            log_event(
                _log, logging.ERROR, "ui.demo_failed", error=f"{type(exc).__name__}: {exc}"
            )
            self.failed.emit(f"{type(exc).__name__}: {exc}")



class DemoPanel(QWidget):
    """Runs the built-in scenario and reports BEFORE/AFTER."""

    def __init__(self, theme: Theme, settings, parent=None) -> None:  # noqa: ANN001
        super().__init__(parent)
        self._theme = theme
        self._settings = settings
        self._thread: QThread | None = None
        self._worker: _DemoWorker | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        header = QHBoxLayout()
        self._summary = QLabel(
            "One deterministic scenario, offline, repeatable.\n\n"
            "A sample app ships with a wrong import. It genuinely fails. The app "
            "captures the real traceback, diagnoses it, proposes one whitelisted "
            "fix, and waits for approval before applying anything."
        )
        self._summary.setWordWrap(True)
        self._outcome = StatusPill(theme)
        self._outcome.set_status("not run", "muted")
        header.addWidget(self._summary, 1)
        header.addWidget(self._outcome, 0)
        layout.addLayout(header)

        self._table = QTableWidget(0, len(_HEADERS))
        self._table.setHorizontalHeaderLabels(_HEADERS)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table, 1)

        footer = QHBoxLayout()
        self._note = QLabel(
            "The diagnosis is rule-based and the fix is a predefined action. "
            "Both are shown as such."
        )
        self._note.setWordWrap(True)
        self._note.setStyleSheet(f"color: {self._theme.text_dim};")
        footer.addWidget(self._note, 1)
        self._run = QPushButton("Run the demo")
        self._run.setProperty("primary", True)
        self._run.clicked.connect(self._on_run)
        footer.addWidget(self._run)
        layout.addLayout(footer)

    # -- behaviour --------------------------------------------------------

    def _on_run(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self._run.setEnabled(False)
        self._outcome.set_status("running", "warn", "The scenario is executing.")
        self._thread = QThread(self)
        self._worker = _DemoWorker(self._settings)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    def _on_finished(self, result) -> None:  # noqa: ANN001 - DemoResult
        self._teardown()
        self.show_result(result)

    def _on_failed(self, reason: str) -> None:
        self._teardown()
        self._outcome.set_status("failed", "danger", reason)
        self._summary.setText("The demo did not complete. Nothing was changed.")
        QMessageBox.warning(self, "Demo failed", reason)

    def _teardown(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
        self._thread = None
        self._worker = None
        self._run.setEnabled(True)


    # -- rendering --------------------------------------------------------

    def show_result(self, result) -> None:  # noqa: ANN001 - DemoResult
        """Render a completed demo run."""
        rows = result.as_rows()
        self._table.setRowCount(len(rows))
        for row, (label, value) in enumerate(rows):
            for column, text in enumerate((label, str(value))):
                cell = QTableWidgetItem(text)
                if label in _VERDICT_ROWS:
                    cell.setForeground(
                        QBrush(
                            QColor(
                                self._theme.status(
                                    "ok" if text == "PASSED" else "danger"
                                )
                            )
                        )
                    )
                self._table.setItem(row, column, cell)
        self._table.resizeColumnsToContents()
        self._table.horizontalHeader().setStretchLastSection(True)

        if result.worked:
            self._outcome.set_status(
                "BEFORE FAILED -> AFTER PASSED",
                "ok",
                "The fix was approved, applied, and verified by re-running the app.",
            )
            self._summary.setText(
                "The scenario worked: the app failed, the fix was approved, and "
                "the same check now passes."
            )
        else:
            self._outcome.set_status(
                "inconclusive", "warn", "See the rows below for what happened."
            )
            self._summary.setText(
                "The scenario did not reach the expected result. The rows below "
                "show what was actually observed."
            )
        self._note.setText(
            f"Applied: {result.notes[0]}" if result.notes else "Nothing was applied."
        )
