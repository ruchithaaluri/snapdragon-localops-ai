"""The benchmark panel: measured numbers, on this machine, or none at all.

The panel deliberately shows almost nothing before a run. An empty table that
says "not measured yet" is the honest starting state; a table of plausible
figures would not be.

After a run, every row carries the execution target it was measured on. The
``NPU`` word cannot appear here unless a verified NPU run produced the row -
:class:`~core.contracts.benchmark.Measurement` raises otherwise, so the panel
cannot show an NPU number that was not produced.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from app.ui.theme import Theme
from app.ui.widgets import StatusPill
from core.observability import get_logger, log_event

__all__ = ["BenchmarkPanel"]

_log = get_logger(__name__)

_NOT_RUN = (
    "No benchmark has run yet.\n\n"
    "Press Measure to time this machine. The numbers below are produced on the "
    "host named in the report, or they are not shown at all."
)


class BenchmarkPanel(QWidget):
    """Runs the measurement suite and renders the report."""

    measure_requested = Signal()

    def __init__(self, theme: Theme, parent=None) -> None:  # noqa: ANN001
        super().__init__(parent)
        self._theme = theme

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        header = QHBoxLayout()
        self._summary = QLabel(_NOT_RUN)
        self._summary.setWordWrap(True)
        self._status = StatusPill(theme)
        self._status.set_status("not measured", "muted")
        header.addWidget(self._summary, 1)
        header.addWidget(self._status, 0)
        layout.addLayout(header)

        self._report = QTextBrowser()
        self._report.setOpenExternalLinks(False)
        layout.addWidget(self._report, 1)

        footer = QHBoxLayout()
        self._note = QLabel(
            "CPU figures are measured on this machine. NPU figures require a "
            "verified run on Snapdragon hardware, which has not been performed."
        )
        self._note.setWordWrap(True)
        self._note.setStyleSheet(f"color: {self._theme.text_dim};")
        footer.addWidget(self._note, 1)
        self._run = QPushButton("Measure this machine")
        self._run.setProperty("primary", True)
        self._run.clicked.connect(self.measure_requested.emit)
        footer.addWidget(self._run)
        layout.addLayout(footer)

    def set_busy(self, busy: bool) -> None:
        self._run.setEnabled(not busy)
        if busy:
            self._status.set_status("measuring", "warn", "Timing real work.")

    def show_report(self, report) -> None:  # noqa: ANN001 - BenchmarkReport
        """Render a measured report, or explain why there is none."""
        if report is None or report.is_empty:
            self._status.set_status(
                "not measured", "muted", "The suite produced no measurements."
            )
            self._summary.setText(
                "The suite produced no measurements. The notes below say why."
            )
            self._report.setPlainText("\n".join(getattr(report, "notes", ())))
            return

        self._status.set_status(
            "VERIFIED", "ok", "Every row below was measured on this machine."
        )
        self._summary.setText(
            f"{len(report.measurements)} measurement(s) on {report.host}. "
            f"Each row is a verified CPU run."
        )
        self._report.setMarkdown(report.to_markdown())
        log_event(
            _log, logging.INFO, "ui.benchmark_rendered", measured=len(report.measurements)
        )
