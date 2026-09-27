"""Pipeline contract: the end-to-end diagnostic run.

    capture -> evidence -> diagnosis -> remediation -> approval -> verification

Phase 1 implements this as a *coordinated* pipeline: the deterministic stages
run for real, the AI stages are filled by
:class:`~core.contracts.ai.AIBackend` (the mock backend in this phase), and the
approval gate is a real object that starts in the ``PENDING`` state. Nothing is
skipped silently: :class:`PipelineRun` records the outcome of every stage,
including the ones that did not run and why.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.contracts.base import Component
from core.contracts.diagnostics import Diagnosis, DiagnosticQuery
from core.contracts.remediation import RemediationResult
from core.contracts.verification import VerificationCheck, VerificationReport
from core.ids import new_id, utc_now
from core.safety import ApprovalRecord, RemediationPlan

__all__ = [
    "Pipeline",
    "PipelineRun",
    "PipelineStage",
    "StageOutcome",
    "StageRecord",
    "StageStatus",
]


class PipelineStage(str, Enum):
    """The stages of one diagnostic run, in order."""

    CAPTURE = "capture"
    EVIDENCE = "evidence"
    RETRIEVAL = "retrieval"
    DIAGNOSIS = "diagnosis"
    REMEDIATION = "remediation"
    APPROVAL = "approval"
    VERIFICATION = "verification"


class StageStatus(str, Enum):
    """State of a stage inside a run."""

    PENDING = "pending"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class StageOutcome(str, Enum):
    """Terminal outcome of a run."""

    DIAGNOSED = "diagnosed"
    AWAITING_APPROVAL = "awaiting_approval"
    REMEDIATED = "remediated"
    VERIFIED = "verified"
    UNRESOLVED = "unresolved"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class StageRecord:
    """What happened in one stage."""

    stage: PipelineStage
    status: StageStatus
    detail: str = ""
    duration_ms: float = 0.0
    started_at: str = field(default_factory=lambda: utc_now().isoformat(timespec="seconds"))

    @property
    def is_complete(self) -> bool:
        return self.status is StageStatus.COMPLETED

    def as_row(self) -> tuple[str, str, str]:
        return (self.stage.value, self.status.value, self.detail)


@dataclass(frozen=True, slots=True)
class PipelineRun:
    """One end-to-end run: its inputs, its stages, and its artefacts."""

    query: DiagnosticQuery
    stages: tuple[StageRecord, ...] = field(default=())
    diagnosis: Diagnosis | None = None
    plan: RemediationPlan | None = None
    approvals: tuple[ApprovalRecord, ...] = field(default=())
    remediation_results: tuple[RemediationResult, ...] = field(default=())
    verification: VerificationReport | None = None
    run_id: str = field(default_factory=lambda: new_id("run"))
    outcome: StageOutcome = StageOutcome.UNRESOLVED
    created_at: str = field(default_factory=lambda: utc_now().isoformat(timespec="seconds"))

    def stage(self, stage: PipelineStage) -> StageRecord | None:
        for record in self.stages:
            if record.stage is stage:
                return record
        return None

    def status_of(self, stage: PipelineStage) -> StageStatus:
        record = self.stage(stage)
        return record.status if record else StageStatus.PENDING

    @property
    def is_awaiting_approval(self) -> bool:
        return self.outcome is StageOutcome.AWAITING_APPROVAL

    def as_rows(self) -> tuple[tuple[str, str, str], ...]:
        """Stage rows for the UI's progress column."""
        return tuple(record.as_row() for record in self.stages)


class Pipeline(Component):
    """Coordinates the stages of a diagnostic run."""

    kind = "pipeline"

    @abstractmethod
    def run(self, query: DiagnosticQuery, *, auto_approve: bool = False) -> PipelineRun:
        """Run the pipeline up to (but never past) the approval gate.

        ``auto_approve`` must stay ``False`` in every production path. It exists
        so tests and the demo can exercise the post-approval stages; the
        executor still requires a real :class:`ApprovalRecord`.
        """

    def verify(
        self,
        run: PipelineRun,
        checks: Sequence[VerificationCheck],
    ) -> VerificationReport:
        """Run verification checks against an existing run."""
        raise NotImplementedError(f"{self.info.name} does not run verification")

    def to_dict(self) -> dict[str, Any]:
        """Serialise the run for logs and the SQLite store."""
        return {
            "run_id": self.run_id,
            "outcome": self.outcome.value,
            "created_at": self.created_at,
            "question": self.query.question,
            "stages": [record.as_row() for record in self.stages],
        }
