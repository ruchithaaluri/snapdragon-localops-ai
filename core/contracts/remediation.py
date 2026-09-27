"""Remediation contracts: proposing a fix, and applying it under approval.

The split matters. :class:`RemediationPlanner` decides *what* might help;
:class:`RemediationExecutor` decides whether it may *run*, and only ever after
an explicit user decision recorded in an
:class:`~core.safety.ApprovalRecord`.

Executors are held to these rules:

* no autonomous action: :meth:`RemediationExecutor.execute` must refuse to run
  anything without an approval record for that exact action id,
* policy is checked first, approval second, execution last,
* every execution is logged and returns a structured result, including failures,
* read-only actions are still approved, because "run this command" is not the
  user's own decision by default.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from core.contracts.base import Capability, Component, Provenance
from core.ids import new_id, utc_now
from core.safety import ApprovalRecord, RemediationAction, RemediationPlan, SafetyPolicy

__all__ = [
    "ExecutionOutcome",
    "RemediationExecutor",
    "RemediationPlanner",
    "RemediationResult",
]


class ExecutionOutcome(str, Enum):
    """Result state of one applied action."""

    APPLIED = "applied"
    FAILED = "failed"
    SKIPPED = "skipped"
    DENIED = "denied"
    NOT_APPROVED = "not_approved"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True, slots=True)
class RemediationResult:
    """What happened when one action was applied."""

    action_id: str
    outcome: ExecutionOutcome
    detail: str = ""
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    executed_at: str = field(default_factory=lambda: utc_now().isoformat(timespec="seconds"))
    duration_ms: float = 0.0
    provenance: Provenance | None = None

    @property
    def succeeded(self) -> bool:
        return self.outcome is ExecutionOutcome.APPLIED

    @classmethod
    def not_approved(cls, action: RemediationAction) -> "RemediationResult":
        return cls(action_id=action.action_id, outcome=ExecutionOutcome.NOT_APPROVED)

    @classmethod
    def denied(cls, action: RemediationAction, reason: str) -> "RemediationResult":
        return cls(action_id=action.action_id, outcome=ExecutionOutcome.DENIED, detail=reason)


class RemediationPlanner(Component):
    """Turns a diagnosis into proposed actions. Proposes; never executes."""

    kind = "remediation_planner"

    @abstractmethod
    def plan(self, diagnosis: object, *, context: dict[str, str] | None = None) -> RemediationPlan:
        """Propose actions for a diagnosis.

        ``diagnosis`` is typed as ``object`` to keep this module free of a
        dependency on the diagnostics package; implementations narrow it.
        """


class RemediationExecutor(Component):
    """Applies approved actions, and only approved actions."""

    kind = "remediation_executor"

    def __init__(self, policy: SafetyPolicy) -> None:
        self._policy = policy

    @property
    def policy(self) -> SafetyPolicy:
        """The safety policy this executor enforces."""
        return self._policy

    @abstractmethod
    def execute(
        self,
        action: RemediationAction,
        approval: ApprovalRecord | None,
    ) -> RemediationResult:
        """Apply one action if, and only if, it is approved and permitted."""

    def execute_plan(
        self,
        plan: RemediationPlan,
        approvals: Sequence[ApprovalRecord],
    ) -> tuple[RemediationResult, ...]:
        """Apply every approved action of a plan, skipping the rest."""
        by_id = {record.action_id: record for record in approvals}
        return tuple(self.execute(action, by_id.get(action.action_id)) for action in plan.actions)

    def new_action(
        self,
        title: str,
        description: str,
        *,
        risk: object = None,
        **kwargs: object,
    ) -> RemediationAction:
        """Helper for planners building actions with a fresh id."""
        from core.safety import RiskLevel

        return RemediationAction(
            action_id=new_id("act"),
            title=title,
            description=description,
            risk=risk if risk is not None else RiskLevel.READ_ONLY,  # type: ignore[arg-type]
            **kwargs,  # type: ignore[arg-type]
        )

    def preferred_capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.REMEDIATION_PLANNING})
