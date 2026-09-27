"""Safety kernel: risk classification, approval records, and the approval gate.

Hard rules encoded here:

* A remediation action **cannot** opt out of approval. ``requires_approval``
  exists so plans can be explicit, but it cannot be set to ``False``.
* Approval is per-action, not per-plan, and is bound to a specific action id.
* A deny-by-default gate is always available, so a missing UI or a headless run
  can never accidentally approve anything.
* Command text is screened against a deny-list before it can be shown as an
  executable suggestion.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from core.errors import ActionDeniedError, ApprovalRequiredError
from core.ids import new_id, utc_now

__all__ = [
    "DENIED_COMMAND_PATTERNS",
    "ApprovalDecision",
    "ApprovalGate",
    "ApprovalRecord",
    "ApprovalStatus",
    "DenyAllApprovalGate",
    "InMemoryApprovalGate",
    "RemediationAction",
    "RemediationPlan",
    "RiskLevel",
    "SafetyPolicy",
]


class RiskLevel(str, Enum):
    """How much damage an action could do if it is wrong."""

    READ_ONLY = "read_only"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    DESTRUCTIVE = "destructive"

    @property
    def rank(self) -> int:
        """Ordinal for comparisons (``RiskLevel.HIGH.rank > RiskLevel.LOW.rank``)."""
        return _RISK_ORDER[self]


_RISK_ORDER = {
    RiskLevel.READ_ONLY: 0,
    RiskLevel.LOW: 1,
    RiskLevel.MEDIUM: 2,
    RiskLevel.HIGH: 3,
    RiskLevel.DESTRUCTIVE: 4,
}


class ApprovalDecision(str, Enum):
    """Outcome of a human decision about one action."""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"

    @property
    def is_approved(self) -> bool:
        return self is ApprovalDecision.APPROVED


class ApprovalStatus(str, Enum):
    """State of a plan as a whole, for the UI."""

    PENDING = "pending"
    PARTIALLY_APPROVED = "partially_approved"
    APPROVED = "approved"
    DENIED = "denied"


@dataclass(frozen=True, slots=True)
class RemediationAction:
    """A single proposed change, with its risk and rollback.

    The application never invents commands. ``command`` is only ever populated
    from something the user or a verified artefact supplied, and it is screened
    by :meth:`SafetyPolicy.check_command` before display.
    """

    action_id: str
    title: str
    description: str
    risk: RiskLevel = RiskLevel.READ_ONLY
    command: str | None = None
    requires_approval: bool = True
    reversible: bool = True
    rollback_command: str | None = None
    working_directory: str | None = None
    supporting_evidence_ids: tuple[str, ...] = field(default=())
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("remediation action requires an action_id")
        if not self.title:
            raise ValueError("remediation action requires a title")
        object.__setattr__(self, "requires_approval", True)
        if self.risk is RiskLevel.READ_ONLY and self.rollback_command is None:
            object.__setattr__(self, "reversible", True)

    @property
    def is_mutating(self) -> bool:
        """True when the action can change system or project state."""
        return self.risk is not RiskLevel.READ_ONLY

    def describe(self) -> str:
        """One-line, log-safe description. Never includes secrets."""
        return f"[{self.risk.value}] {self.title} ({self.action_id})"


@dataclass(frozen=True, slots=True)
class RemediationPlan:
    """An ordered set of actions proposed for one diagnostic run."""

    run_id: str
    summary: str
    actions: tuple[RemediationAction, ...] = field(default=())
    plan_id: str = field(default_factory=lambda: new_id("plan"))
    created_at: str = field(default_factory=lambda: utc_now().isoformat(timespec="seconds"))
    generator: str = "unknown"
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("plan confidence must be within [0, 1]")

    def action(self, action_id: str) -> RemediationAction:
        for action in self.actions:
            if action.action_id == action_id:
                return action
        raise KeyError(action_id)

    @property
    def highest_risk(self) -> RiskLevel:
        return max((a.risk for a in self.actions), key=lambda r: r.rank, default=RiskLevel.READ_ONLY)

    def mutating_actions(self) -> tuple[RemediationAction, ...]:
        return tuple(a for a in self.actions if a.is_mutating)


@dataclass(frozen=True, slots=True)
class ApprovalRecord:
    """An explicit human decision about one action."""

    action_id: str
    decision: ApprovalDecision
    decided_at: str = field(default_factory=lambda: utc_now().isoformat(timespec="seconds"))
    decided_by: str = "local-user"
    note: str = ""

    @classmethod
    def approve(cls, action_id: str, *, decided_by: str = "local-user", note: str = "") -> "ApprovalRecord":
        return cls(action_id=action_id, decision=ApprovalDecision.APPROVED, decided_by=decided_by, note=note)

    @classmethod
    def deny(cls, action_id: str, *, decided_by: str = "local-user", note: str = "") -> "ApprovalRecord":
        return cls(action_id=action_id, decision=ApprovalDecision.DENIED, decided_by=decided_by, note=note)

    @property
    def is_approved(self) -> bool:
        return self.decision.is_approved


@runtime_checkable
class ApprovalGate(Protocol):
    """Decides whether a proposed action may be executed.

    Implementations must default to *not* approved. A gate that approves by
    default is a bug, not a configuration choice.
    """

    def request(self, action: RemediationAction) -> ApprovalRecord:
        """Ask the user (or the policy) about one action and return the record."""
        ...

    def status(self, action: RemediationAction) -> ApprovalStatus:
        """Summarise the current decision state for the UI."""
        ...


class DenyAllApprovalGate:
    """Gate that refuses everything. The safe default for headless runs."""

    def request(self, action: RemediationAction) -> ApprovalRecord:
        return ApprovalRecord.deny(action.action_id, decided_by="deny-all-gate")

    def status(self, action: RemediationAction) -> ApprovalStatus:
        return ApprovalStatus.DENIED

    def assert_approved(self, action: RemediationAction) -> None:
        """Always raises. Used to prove the gate blocks execution."""
        raise ApprovalRequiredError(
            "approval denied by the default deny-all gate",
            action_id=action.action_id,
        )


class InMemoryApprovalGate:
    """Gate that records explicit decisions made through the UI.

    Nothing is approved until :meth:`decide` is called with
    :class:`ApprovalDecision.APPROVED`. Decisions are per action id, and an
    unknown action id is a :class:`KeyError` rather than a silent approval.
    """

    def __init__(self, *, decided_by: str = "local-user") -> None:
        self._records: dict[str, ApprovalRecord] = {}
        self._decided_by = decided_by

    def decide(
        self,
        action: RemediationAction,
        decision: ApprovalDecision,
        *,
        note: str = "",
    ) -> ApprovalRecord:
        """Record the user's decision for one action."""
        if decision is ApprovalDecision.PENDING:
            raise ValueError("a decision must be APPROVED or DENIED, not PENDING")
        record = ApprovalRecord(
            action_id=action.action_id,
            decision=decision,
            decided_by=self._decided_by,
            note=note,
        )
        self._records[action.action_id] = record
        return record

    def approve(self, action: RemediationAction, *, note: str = "") -> ApprovalRecord:
        return self.decide(action, ApprovalDecision.APPROVED, note=note)

    def deny(self, action: RemediationAction, *, note: str = "") -> ApprovalRecord:
        return self.decide(action, ApprovalDecision.DENIED, note=note)

    def revoke(self, action: RemediationAction) -> None:
        """Forget a prior decision, returning the action to pending."""
        self._records.pop(action.action_id, None)

    def get(self, action_id: str) -> ApprovalRecord | None:
        return self._records.get(action_id)

    def request(self, action: RemediationAction) -> ApprovalRecord:
        """Return the existing record, or a pending placeholder."""
        return self._records.get(action.action_id) or ApprovalRecord(
            action_id=action.action_id, decision=ApprovalDecision.PENDING
        )

    def status(self, action: RemediationAction) -> ApprovalStatus:
        record = self.request(action)
        if record.decision is ApprovalDecision.PENDING:
            return ApprovalStatus.PENDING
        return ApprovalStatus.APPROVED if record.is_approved else ApprovalStatus.DENIED

    def assert_approved(self, action: RemediationAction) -> ApprovalRecord:
        """Raise unless this exact action was approved."""
        record = self.request(action)
        if not record.is_approved:
            raise ApprovalRequiredError(
                "action has not been approved by the user",
                action_id=action.action_id,
                decision=record.decision.value,
            )
        return record



#: Command fragments that are never offered as a suggested fix, regardless of
#: the rest of the plan. This is a blunt guard rail, not a security boundary.
DENIED_COMMAND_PATTERNS: tuple[str, ...] = (
    "rm -rf /",
    "format c:",
    "format /",
    "diskpart",
    "reg delete",
    "remove-item -recurse -force",
    "del /f /q",
    "takeown /f c:",
    "cipher /w",
    "shutdown /s",
    "bcdedit",
)


@dataclass(frozen=True, slots=True)
class SafetyPolicy:
    """Limits what the application is allowed to propose or execute.

    ``allow_mutating_actions`` is the master switch for remediation. It is
    ``False`` by default, so a fresh install can only suggest and verify
    read-only checks. ``max_risk`` applies per action and is checked against
    :attr:`RemediationAction.risk`.
    """

    allow_mutating_actions: bool = False
    max_risk: RiskLevel = RiskLevel.MEDIUM
    require_approval: bool = True
    denied_command_patterns: tuple[str, ...] = field(default=DENIED_COMMAND_PATTERNS)
    allow_network_actions: bool = False
    require_reversible: bool = True

    def __post_init__(self) -> None:
        if not self.require_approval:
            raise ValueError("user approval cannot be disabled")

    def check_action(self, action: RemediationAction) -> None:
        """Validate one action against the policy, or raise :class:`ActionDeniedError`."""
        if action.risk.rank > self.max_risk.rank:
            raise ActionDeniedError(
                "action risk exceeds the configured maximum",
                action_id=action.action_id,
                risk=action.risk.value,
                max_risk=self.max_risk.value,
            )
        if action.is_mutating and not self.allow_mutating_actions:
            raise ActionDeniedError(
                "mutating remediation is disabled by policy",
                action_id=action.action_id,
                risk=action.risk.value,
            )
        if self.require_reversible and action.is_mutating and not action.reversible:
            raise ActionDeniedError(
                "irreversible action rejected by policy",
                action_id=action.action_id,
            )
        if action.command:
            self.check_command(action.command, action_id=action.action_id)
        if action.rollback_command:
            self.check_command(action.rollback_command, action_id=action.action_id)

    def check_command(self, command: str, *, action_id: str = "") -> None:
        """Reject commands that match the deny-list."""
        lowered = command.lower()
        for pattern in self.denied_command_patterns:
            if pattern in lowered:
                raise ActionDeniedError(
                    "command matches a denied pattern and will not be offered",
                    action_id=action_id,
                    pattern=pattern,
                )

    def check_plan(self, actions: Sequence[RemediationAction]) -> None:
        """Validate every action in a plan."""
        for action in actions:
            self.check_action(action)

    @property
    def permits_remediation(self) -> bool:
        """Whether mutating remediation may be proposed at all."""
        return self.allow_mutating_actions

