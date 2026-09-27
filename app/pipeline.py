"""End-to-end orchestration: capture -> evidence -> diagnosis -> remediation.

This module wires the layer packages together. It contains no Qt imports, so the
whole flow can be exercised headlessly from a test or the CLI.

Two properties are load-bearing:

* **Every stage is recorded, including the ones that did not run.** A stage that
  was skipped says why in :class:`~core.contracts.pipeline.StageRecord.detail`.
  A pipeline that silently omits a stage cannot be told apart from one where
  the stage found nothing.
* **The run stops at the approval gate and no further.** The pipeline proposes
  actions and hands back a pending plan. It never applies anything unless the
  caller supplies a real :class:`~core.safety.ApprovalRecord` for a specific
  action, which the UI obtains from the user.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from core.config.schema import AppSettings
from core.contracts.base import Capability, ProviderHealth, ProviderInfo
from core.contracts.pipeline import (
    Pipeline,
    PipelineRun,
    PipelineStage,
    StageOutcome,
    StageRecord,
    StageStatus,
)
from core.contracts.retrieval import Retriever, SearchQuery
from core.contracts.remediation import RemediationResult
from core.contracts.verification import VerificationCheck, VerificationReport
from core.errors import InsufficientEvidenceError
from core.ids import new_run_id
from core.observability import get_logger, log_event
from core.safety import ApprovalRecord, InMemoryApprovalGate, SafetyPolicy
from core.timing import Stopwatch
from core.version import __version__
from diagnostics import RuleBasedDiagnosticEngine
from diagnostics.remediation import (
    LocalRemediationExecutor,
    RuleBasedRemediationPlanner,
)
from verification import LocalVerificationRunner

__all__ = ["LocalDiagnosticPipeline", "build_pipeline"]

_log = get_logger(__name__)


class LocalDiagnosticPipeline(Pipeline):
    """Runs the deterministic stages for real and stops at the approval gate.

    Parameters
    ----------
    engine:
        The diagnostic engine. Produces cited hypotheses, or raises
        :class:`~core.errors.InsufficientEvidenceError`.
    executor:
        Applies approved actions. Consulted only after approval.
    verifier:
        Runs read-only checks. Consulted only after remediation.
    gate:
        Approval gate. Defaults to an empty
        :class:`~core.safety.InMemoryApprovalGate`, which approves nothing.
    retriever:
        Optional. When supplied and its index is non-empty, retrieved passages
        are attached to the query as *context* - never as evidence.
    policy:
        The safety policy the executor enforces.
    """

    kind = "pipeline"

    def __init__(
        self,
        engine: RuleBasedDiagnosticEngine,
        executor: LocalRemediationExecutor,
        verifier: LocalVerificationRunner,
        *,
        gate: InMemoryApprovalGate | None = None,
        retriever: Retriever | None = None,
        policy: SafetyPolicy | None = None,
    ) -> None:
        self._engine = engine
        self._executor = executor
        self._verifier = verifier
        self._gate = gate or InMemoryApprovalGate()
        self._retriever = retriever
        self._policy = policy or executor.policy

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="local-pipeline",
            kind=self.kind,
            version=__version__,
            capabilities=frozenset({Capability.ROOT_CAUSE_RANKING}),
            detail="deterministic stages; stops at the approval gate",
        )

    def health(self) -> ProviderHealth:
        return ProviderHealth.ready("all stages available")

    @property
    def gate(self) -> InMemoryApprovalGate:
        """The gate the UI records the user's decisions into."""
        return self._gate

    def run(self, query: object, *, auto_approve: bool = False) -> PipelineRun:
        """Run the pipeline, stopping at the approval gate unless already approved."""
        run_id = new_run_id()
        stages: list[StageRecord] = []

        # -- capture --------------------------------------------------------
        watch = Stopwatch().start()
        stages.append(
            StageRecord(
                stage=PipelineStage.CAPTURE,
                status=StageStatus.COMPLETED,
                detail=f"{len(getattr(query, 'evidence', ()))} item(s) captured",
                duration_ms=watch.stop().milliseconds,
            )
        )

        # -- evidence -------------------------------------------------------
        evidence = tuple(getattr(query, "evidence", ()))
        stages.append(
            StageRecord(
                stage=PipelineStage.EVIDENCE,
                status=StageStatus.COMPLETED if evidence else StageStatus.SKIPPED,
                detail=(
                    f"{len(evidence)} evidence item(s) extracted"
                    if evidence
                    else "no evidence was supplied with the question"
                ),
            )
        )

        # -- retrieval ------------------------------------------------------
        enriched = self._with_retrieval(query, run_id)
        if enriched is not query:
            query = enriched
            stages.append(
                StageRecord(
                    stage=PipelineStage.RETRIEVAL,
                    status=StageStatus.COMPLETED,
                    detail="project context attached as context, not as evidence",
                )
            )
        else:
            stages.append(
                StageRecord(
                    stage=PipelineStage.RETRIEVAL,
                    status=StageStatus.SKIPPED,
                    detail="no retriever configured, or its index is empty",
                )
            )

        # -- diagnosis ------------------------------------------------------
        watch = Stopwatch().start()
        try:
            diagnosis = self._engine.analyze(query)
        except InsufficientEvidenceError as exc:
            elapsed = watch.stop().milliseconds
            stages.append(
                StageRecord(
                    stage=PipelineStage.DIAGNOSIS,
                    status=StageStatus.FAILED,
                    detail=str(exc),
                    duration_ms=elapsed,
                )
            )
            log_event(
                _log,
                logging.WARNING,
                "pipeline.insufficient_evidence",
                run_id=run_id,
                detail=str(exc),
            )
            return PipelineRun(
                query=query,  # type: ignore[arg-type]
                stages=tuple(stages),
                run_id=run_id,
                outcome=StageOutcome.UNRESOLVED,
            )
        stages.append(
            StageRecord(
                stage=PipelineStage.DIAGNOSIS,
                status=StageStatus.COMPLETED,
                detail=(
                    f"{len(diagnosis.hypotheses)} hypothesis(es), top confidence "
                    f"{diagnosis.confidence:.2f}, {diagnosis.reasoning_mode.value}"
                ),
                duration_ms=watch.stop().milliseconds,
            )
        )

        # -- remediation ----------------------------------------------------
        watch = Stopwatch().start()
        plan = RuleBasedRemediationPlanner(evidence).plan(diagnosis)
        stages.append(
            StageRecord(
                stage=PipelineStage.REMEDIATION,
                status=StageStatus.COMPLETED if plan.actions else StageStatus.SKIPPED,
                detail=plan.summary,
                duration_ms=watch.stop().milliseconds,
            )
        )

        # -- approval -------------------------------------------------------
        # The gate. Nothing after this line runs without a real ApprovalRecord.
        approvals = tuple(
            record
            for action in plan.actions
            if (record := self._gate.get(action.action_id)) is not None
        )
        stages.append(
            StageRecord(
                stage=PipelineStage.APPROVAL,
                status=StageStatus.COMPLETED if approvals else StageStatus.PENDING,
                detail=(
                    f"{len(approvals)}/{len(plan.actions)} action(s) approved by the user"
                    if plan.actions
                    else "no actions to approve"
                ),
            )
        )

        results = self._executor.execute_plan(plan, approvals) if approvals else ()
        stages.append(
            StageRecord(
                stage=PipelineStage.VERIFICATION,
                status=StageStatus.SKIPPED,
                detail=(
                    "no verification checks were supplied for this run"
                    if results
                    else "verification runs only after an applied action"
                ),
            )
        )

        outcome = self._outcome(plan, results, auto_approve)
        run = PipelineRun(
            query=query,  # type: ignore[arg-type]
            stages=tuple(stages),
            diagnosis=diagnosis,
            plan=plan,
            approvals=approvals,
            remediation_results=results,
            run_id=run_id,
            outcome=outcome,
        )
        log_event(
            _log,
            logging.INFO,
            "pipeline.run",
            run_id=run_id,
            outcome=outcome.value,
            stages=len(stages),
        )
        return run

    def verify(
        self,
        run: PipelineRun,
        checks: Sequence[VerificationCheck],
    ) -> VerificationReport:
        """Run verification checks against a completed run."""
        report = self._verifier.run(checks)
        log_event(
            _log,
            logging.INFO,
            "pipeline.verified",
            # The run is only needed for its id here, so a report can also be
            # produced for checks a caller prepared on its own.
            run_id=getattr(run, "run_id", ""),
            summary=report.summary(),
            resolved=report.resolved,
        )
        return report

    def apply_plan(
        self,
        plan: object,
        approvals: Sequence[ApprovalRecord],
    ) -> tuple[RemediationResult, ...]:
        """Apply a *previously shown* plan, using approvals already recorded.

        This deliberately does not re-run the pipeline. A plan's action ids are
        minted when the plan is built, so regenerating the plan would mint new
        ids and the approvals the user just gave would silently apply to nothing.
        The user approved these exact actions, so these exact actions are what
        runs.
        """
        return self._executor.execute_plan(plan, approvals)

    # -- internals --------------------------------------------------------

    def _with_retrieval(self, query: object, run_id: str) -> object:
        """Attach retrieved passages as *context*, never as evidence.

        Returning the original object when there is nothing to add keeps the
        ``enriched is not query`` check in :meth:`run` a cheap "did this stage
        do anything?" test.
        """
        if self._retriever is None:
            return query
        index = self._retriever.index()
        if index is None or index.count() == 0:
            return query
        result = self._retriever.retrieve(SearchQuery(text=query.question))  # type: ignore[attr-defined]
        if not result.hits:
            return query
        return type(query)(  # type: ignore[call-arg]
            question=query.question,  # type: ignore[attr-defined]
            evidence=query.evidence,  # type: ignore[attr-defined]
            context=query.context,  # type: ignore[attr-defined]
            project_root=query.project_root,  # type: ignore[attr-defined]
            run_id=run_id,
            max_hypotheses=query.max_hypotheses,  # type: ignore[attr-defined]
            retrieved_context=result.as_context(),
            metadata=dict(query.metadata),  # type: ignore[attr-defined]
        )

    @staticmethod
    def _outcome(plan: object, results: Sequence[object], auto_approve: bool) -> StageOutcome:
        """Decide how the run ended, from what actually happened.

        ``auto_approve`` only ever changes the label of a run that had no
        actions to approve. It cannot make the executor run anything: that still
        requires a real :class:`~core.safety.ApprovalRecord`.
        """
        if not getattr(plan, "actions", ()):
            return StageOutcome.DIAGNOSED
        if not results:
            return StageOutcome.AWAITING_APPROVAL
        return StageOutcome.REMEDIATED


def build_pipeline(settings: AppSettings) -> LocalDiagnosticPipeline:
    """Assemble a pipeline from validated configuration."""
    policy = settings.safety.to_policy()
    engine = RuleBasedDiagnosticEngine(
        require_evidence=settings.diagnostics.require_evidence,
        confidence_threshold=settings.diagnostics.confidence_threshold,
        max_hypotheses=settings.diagnostics.max_candidates,
    )
    executor = LocalRemediationExecutor(
        policy, timeout_seconds=settings.verification.timeout_seconds
    )
    gate = InMemoryApprovalGate()
    verifier = LocalVerificationRunner(
        gate, stop_on_first_failure=settings.verification.stop_on_first_failure
    )
    return LocalDiagnosticPipeline(engine, executor, verifier, gate=gate, policy=policy)
