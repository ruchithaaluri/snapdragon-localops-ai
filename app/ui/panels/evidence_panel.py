"""Evidence table: every extracted fact, with the line it came from.

The evidence id is shown because the diagnosis cites it. Without a visible
mapping from id to source line, a citation would be an unverifiable claim.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from core.contracts.diagnostics import Evidence, EvidenceKind

__all__ = ["EvidencePanel"]

_HEADERS = ("Kind", "Source", "Line", "Confidence", "Content")


class EvidencePanel(QWidget):
    """Lists the evidence a diagnosis is allowed to rely on."""

    def __init__(self, parent=None) -> None:  # noqa: ANN001 - Qt parent
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._summary = QLabel("No evidence yet.")
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        self._table = QTableWidget(0, len(_HEADERS))
        self._table.setHorizontalHeaderLabels(_HEADERS)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table, 1)

    def clear(self, message: str = "No evidence yet.") -> None:
        self._table.setRowCount(0)
        self._summary.setText(message)

    def show_evidence(self, evidence: tuple[Evidence, ...]) -> None:
        """Render ``evidence``, newest extraction last."""
        if not evidence:
            self.clear(
                "No recognised error pattern was found in the text. Paste more of "
                "the output, including the lines just above the error."
            )
            return

        self._table.setRowCount(len(evidence))
        for row, item in enumerate(evidence):
            cells = (
                item.kind.value,
                item.source or "-",
                item.locator or "-",
                f"{item.confidence:.2f}",
                item.excerpt(400),
            )
            for column, text in enumerate(cells):
                cell = QTableWidgetItem(text)
                if column in (0, 2, 3):
                    cell.setTextAlignment(Qt.AlignCenter)
                # The id is the row's tooltip and data role: the diagnosis cites
                # it, so selecting a row can show exactly what was cited.
                cell.setData(Qt.UserRole, item.evidence_id)
                if column == 4:
                    cell.setToolTip(item.content)
                self._table.setItem(row, column, cell)

        faults = sum(1 for e in evidence if e.kind is not EvidenceKind.USER_INPUT)
        self._summary.setText(
            f"{len(evidence)} item(s) extracted; {faults} describe a fault. "
            "Select a row to see the full text."
        )
        self._table.resizeColumnsToContents()
        self._table.horizontalHeader().setStretchLastSection(True)
