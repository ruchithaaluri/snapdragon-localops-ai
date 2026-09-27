"""Verification: did the suggested change actually help?

Each row shows the expectation and the value that was *observed*, so a pass is
auditable. A bare "OK" would not tell the user what was checked.

Checks are offered pending approval and run only once the user has agreed to
them. The panel never runs anything itself - it emits
:attr:`VerificationPanel.run_requested` and the window performs the run.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.contracts.verification import VerificationReport
from app.ui.theme import Theme
from app.ui.widgets import StatusPill

__all__ = ["VerificationPanel"]

_HEADERS = ("Check", "Outcome", "Expected", "Observed")
_STATUS_FOR = {
    "passed": "ok",
    "failed": "danger",
    "skipped": "muted",
    "error": "warn",
}

_NOT_RUN = (
    "No verification has run yet.\n\n"
    "Checks are read-only by default and are approved individually."
)
_NO_CHECKS = (
    "No check could be derived from this run.\n\n"
    "Verification re-reads a file or re-runs a command. When the evidence was "
    "pasted rather than loaded from disk, and no action carried a command, there "
    "is nothing honest to re-test."
)


class VerificationPanel(QWidget):
    """Offers derived checks, then renders the resulting report."""

    run_requested = Signal()

    def __init__(self, theme: Theme, parent=None) -> None:  # noqa: ANN001 - Qt parent
        super().__init__(parent)
        self._theme = theme
        self._checks: tuple = ()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QHBoxLayout()
        self._summary = QLabel(_NOT_RUN)
        self._summary.setWordWrap(True)
        self._resolved = StatusPill(theme)
        self._resolved.set_status("not run", "muted")
        header.addWidget(self._summary, 1)
        header.addWidget(self._resolved, 0)
        layout.addLayout(header)

        self._table = QTableWidget(0, len(_HEADERS))
        self._table.setHorizontalHeaderLabels(_HEADERS)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table, 1)

        footer = QHBoxLayout()
        self._note = QLabel("")
        self._note.setWordWrap(True)
        self._note.setStyleSheet(f"color: {self._theme.text_dim};")
        footer.addWidget(self._note, 1)
        self._run = QPushButton("Approve and run these checks")
        self._run.setProperty("primary", True)
        self._run.setEnabled(False)
        self._run.setToolTip(
            "Runs the checks listed above. Each re-reads a file or re-runs a "
            "command; nothing is modified."
        )
        self._run.clicked.connect(self._on_run)
        footer.addWidget(self._run)
        layout.addLayout(footer)

    # -- rendering --------------------------------------------------------

    def clear(self, message: str | None = None) -> None:
        self._checks = ()
        self._table.setRowCount(0)
        self._run.setEnabled(False)
        self._resolved.set_status("not run", "muted")
        self._summary.setText(message or _NOT_RUN)
        self._note.setText("")

    def show_checks(self, checks) -> None:
        """Offer derived checks. Nothing runs until the user approves them."""
        self._checks = tuple(checks)
        if not self._checks:
            self.clear(_NO_CHECKS)
            return

        self._table.setRowCount(len(self._checks))
        for row, check in enumerate(self._checks):
            for column, text in enumerate(
                (check.description, "pending", check.expectation, "-")
            ):
                cell = QTableWidgetItem(text)
                if column == 1:
                    cell.setForeground(QBrush(QColor(self._theme.status("warn"))))
                self._table.setItem(row, column, cell)
        self._table.resizeColumnsToContents()
        self._table.horizontalHeader().setStretchLastSection(True)

        self._summary.setText(
            f"{len(self._checks)} check(s) ready. Nothing has been run yet."
        )
        self._resolved.set_status("pending", "warn", "No check has been run.")
        self._note.setText(
            "Running a check re-reads a file or re-runs a command. Nothing is modified."
        )
        self._run.setEnabled(True)

    def show_report(self, report: VerificationReport | None) -> None:
        """Render a report, or explain why there is none."""
        if report is None or not len(report):
            self.clear(
                "No verification ran for this run.\n\n"
                "Verification needs checks to run. Nothing is checked, and no "
                "result is claimed, until you have reviewed them."
            )
            return

        self._checks = ()
        self._run.setEnabled(False)
        self._summary.setText(report.summary())
        self._resolved.set_status(
            "resolved" if report.resolved else "not resolved",
            "ok" if report.resolved else "warn",
            "At least one check passed and none failed."
            if report.resolved
            else "At least one check failed, or none passed.",
        )

        self._table.setRowCount(len(report))
        for row, result in enumerate(report):
            for column, text in enumerate(
                (
                    result.description or result.check_id,
                    result.outcome.value,
                    result.expected,
                    result.observed,
                )
            ):
                cell = QTableWidgetItem(text)
                if column == 1:
                    color = self._theme.status(
                        _STATUS_FOR.get(result.outcome.value, "muted")
                    )
                    cell.setForeground(QBrush(QColor(color)))
                if column == 3 and result.detail:
                    cell.setToolTip(result.detail)
                self._table.setItem(row, column, cell)
        self._table.resizeColumnsToContents()
        self._table.horizontalHeader().setStretchLastSection(True)
        self._note.setText("")

    # -- accessors --------------------------------------------------------

    def checks(self):
        """The checks currently offered for approval."""
        return self._checks

    def _on_run(self) -> None:
        self._run.setEnabled(False)
        self.run_requested.emit()
