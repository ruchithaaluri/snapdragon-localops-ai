"""Proposed actions, each with its own explicit approve/deny decision.

Nothing in this panel runs an action directly. Approving records a decision in
the pipeline's :class:`~core.safety.InMemoryApprovalGate`; the executor still
re-checks the safety policy *and* the action id, so a defect here could at worst
approve the wrong thing, never execute something the user did not see.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.safety import RemediationAction, RemediationPlan
from app.ui.theme import Theme
from app.ui.widgets import StatusPill

__all__ = ["RemediationPanel"]

_NO_ACTIONS = (
    "No actions have been proposed yet.\n\n"
    "Nothing is ever run without an explicit decision on each action below."
)


class RemediationPanel(QWidget):
    """Lists proposed actions and collects per-action approval decisions."""

    approved = Signal(str)   # action id
    denied = Signal(str)     # action id
    apply_requested = Signal()

    def __init__(self, theme: Theme, parent=None) -> None:  # noqa: ANN001 - Qt parent
        super().__init__(parent)
        self._theme = theme
        self._cards: dict[str, QWidget] = {}
        self._plan: RemediationPlan | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        self._summary = QLabel(_NO_ACTIONS)
        self._summary.setWordWrap(True)
        outer.addWidget(self._summary)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._container = QWidget()
        self._cards_layout = QVBoxLayout(self._container)
        self._cards_layout.setAlignment(Qt.AlignTop)
        self._scroll.setWidget(self._container)
        self._scroll.setVisible(False)
        outer.addWidget(self._scroll, 1)

        self._apply = QPushButton("Run the approved actions")
        self._apply.setProperty("primary", True)
        self._apply.setEnabled(False)
        self._apply.clicked.connect(self._on_apply_clicked)
        outer.addWidget(self._apply)

    # -- rendering --------------------------------------------------------

    def clear(self, message: str | None = None) -> None:
        self._plan = None
        self._cards.clear()
        self._apply.setEnabled(False)
        self._scroll.setVisible(False)
        self._summary.setText(message or _NO_ACTIONS)

    def show_plan(self, plan: RemediationPlan | None) -> None:
        """Render a plan, with every action starting in the pending state."""
        self.clear()
        self._plan = plan
        if plan is None or not plan.actions:
            self._summary.setText(
                "No automatic check is available for this evidence.\n\n"
                "That is a real answer, not a failure: the planner only suggests a "
                "check when it has an honest one to offer. Follow the citations in "
                "the Diagnosis tab."
            )
            return

        self._summary.setText(plan.summary)
        self._scroll.setVisible(True)
        for action in plan.actions:
            card = self._build_card(action)
            self._cards[action.action_id] = card
            self._cards_layout.addWidget(card)

    def mark_decision(self, action_id: str, approved: bool) -> None:
        """Reflect a recorded decision on the matching card."""
        card = self._cards.get(action_id)
        if card is None:
            return
        pill = card.findChild(StatusPill)
        if pill is not None:
            pill.set_status(
                "approved" if approved else "denied", "ok" if approved else "danger"
            )
        for button in card.findChildren(QPushButton):
            button.setEnabled(False)
        self._refresh_apply()

    def show_results(self, results) -> None:  # noqa: ANN001 - tuple[RemediationResult, ...]
        """Report what actually happened for each action."""
        for result in results:
            card = self._cards.get(result.action_id)
            if card is None:
                continue
            output = card.findChild(QWidget, "output")
            if output is not None:
                output.setText(_format_output(result))
        self._refresh_apply()

    # -- internals --------------------------------------------------------

    def _build_card(self, action: RemediationAction) -> QWidget:
        card = QGroupBox(action.title)
        card_layout = QVBoxLayout(card)

        description = QLabel(action.description)
        description.setWordWrap(True)
        card_layout.addWidget(description)

        if action.command:
            command = QLabel(f"$ {action.command}")
            command.setWordWrap(True)
            command.setTextInteractionFlags(Qt.TextSelectableByMouse)
            command.setStyleSheet("font-family: Consolas, monospace;")
            card_layout.addWidget(command)
        else:
            note = QLabel("No command - this is advice to read, not to run.")
            note.setStyleSheet("font-style: italic;")
            card_layout.addWidget(note)

        if action.rationale:
            why = QLabel(action.rationale)
            why.setWordWrap(True)
            why.setStyleSheet(f"color: {self._theme.text_dim};")
            card_layout.addWidget(why)

        row = QHBoxLayout()
        pill = StatusPill(self._theme)
        pill.set_status("pending", "warn", "No decision has been made yet.")
        approve = QPushButton("Approve")
        approve.setToolTip(
            "Records your approval for this action only. The safety policy is "
            "still applied afterwards."
        )
        deny = QPushButton("Deny")
        approve.clicked.connect(lambda _=False, a=action: self._on_decide(a, True))
        deny.clicked.connect(lambda _=False, a=action: self._on_decide(a, False))
        row.addWidget(pill)
        row.addStretch(1)
        row.addWidget(deny)
        row.addWidget(approve)
        card_layout.addLayout(row)

        output = QLabel()
        output.setObjectName("output")
        output.setWordWrap(True)
        output.setTextInteractionFlags(Qt.TextSelectableByMouse)
        output.setStyleSheet(f"color: {self._theme.text_dim};")
        card_layout.addWidget(output)

        risk = QLabel(f"risk: {action.risk.value}  -  always requires approval")
        risk.setStyleSheet(f"color: {self._theme.text_dim}; font-size: 11px;")
        card_layout.addWidget(risk)
        return card

    def _on_decide(self, action: RemediationAction, approved: bool) -> None:
        self.mark_decision(action.action_id, approved)
        (self.approved if approved else self.denied).emit(action.action_id)

    def _on_apply_clicked(self) -> None:
        approved = self._approved_ids()
        if not approved:
            QMessageBox.information(
                self, "Nothing approved", "Approve at least one action first."
            )
            return
        confirm = QMessageBox.question(
            self,
            "Run the approved actions?",
            f"{len(approved)} action(s) will run on this machine.\n\n"
            "Each was approved individually. Continue?",
        )
        if confirm is QMessageBox.StandardButton.Yes:
            self.apply_requested.emit()

    def _approved_ids(self) -> list[str]:
        return [
            action_id
            for action_id, card in self._cards.items()
            if (pill := card.findChild(StatusPill)) is not None
            and pill.text() == "approved"
        ]

    def _refresh_apply(self) -> None:
        self._apply.setEnabled(bool(self._approved_ids()))


def _format_output(result) -> str:  # noqa: ANN001 - RemediationResult
    """Render one result as text: outcome, detail, then captured output."""
    lines = [result.outcome.value + (f" - {result.detail}" if result.detail else "")]
    for stream in (result.stdout, result.stderr):
        text = stream.strip()
        if text:
            lines.append(text)
    return "\n".join(lines)
