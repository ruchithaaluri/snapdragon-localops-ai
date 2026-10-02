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
import threading
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
from core.errors import InsufficientEvidenceError, LocalOpsError
from core.ids import new_run_id
from core.observability import get_logger, log_event
from core.safety import ApprovalRecord, InMemoryApprovalGate, SafetyPolicy
from core.storage import SqliteStore, open_store
from core.timing import Stopwatch
from core.version import __version__
from diagnostics import RuleBasedDiagnosticEngine
from diagnostics.remediation import (
    LocalRemediationExecutor,
    RuleBasedRemediationPlanner,
)
from verification import LocalVerificationRunner

__all__ = [
    "LocalDiagnosticPipeline",
    "build_pipeline",
    "build_retriever",
    "build_store",
]

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
    store:
        Optional. When supplied, each run is appended to this local database.
        Persistence is best effort: a storage failure is logged and the run is
        still returned, because the diagnosis is the product and the history is
        only a record of it.
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
        store: SqliteStore | None = None,
        retrieval_top_k: int = 8,
        retrieval_min_score: float = 0.0,
    ) -> None:
        self._engine = engine
        self._executor = executor
        self._verifier = verifier
        self._gate = gate or InMemoryApprovalGate()
        self._retriever = retriever
        self._policy = policy or executor.policy
        self._store = store
        self._retrieval_top_k = retrieval_top_k
        self._retrieval_min_score = retrieval_min_score

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

    @property
    def store(self) -> SqliteStore | None:
        """The local run history, when persistence is enabled."""
        return self._store

    @property
    def retriever(self) -> Retriever | None:
        """The retriever supplying project context, when one is configured."""
        return self._retriever

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
                    detail=self._retrieval_reason(),
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
            unresolved = PipelineRun(
                query=query,  # type: ignore[arg-type]
                stages=tuple(stages),
                run_id=run_id,
                outcome=StageOutcome.UNRESOLVED,
            )
            # An unresolved run is still a run: the user asked, the evidence was
            # not enough, and that fact is worth remembering.
            self._persist(unresolved)
            return unresolved
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
        self._persist(run)
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

    def _persist(self, run: PipelineRun) -> None:
        """Append the run to local history. Best effort, and never fatal.

        The diagnosis is the product; the database is only a record of it. A
        locked file, a full disk or a read-only profile must cost the user their
        history and nothing else, so every failure is caught and logged rather
        than raised. It is never silent either.
        """
        if self._store is None:
            return
        try:
            self._store.save_run(
                run_id=run.run_id,
                question=run.query.question,
                outcome=run.outcome.value,
                created_at=run.created_at,
                payload=_run_payload(run),
            )
            if run.query.evidence:
                self._store.save_evidence(run.run_id, run.query.evidence)
            for record in run.approvals:
                self._store.save_approval(run.run_id, record)
        except Exception as exc:  # noqa: BLE001 - history is optional, the run is not
            log_event(
                _log,
                logging.WARNING,
                "pipeline.persist_failed",
                run_id=run.run_id,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _retrieval_reason(self) -> str:
        """Why retrieval will add nothing, stated precisely rather than vaguely."""
        if self._retriever is None:
            return "no retriever configured"
        index = self._retriever.index()
        if index is None:
            return "the configured retriever exposes no index"
        return f"the local index holds no chunks ({index.count()})"

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
        result = self._retriever.retrieve(
            SearchQuery(
                text=query.question,  # type: ignore[attr-defined]
                top_k=self._retrieval_top_k,
                min_score=self._retrieval_min_score,
            )
        )
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


def build_store(settings: AppSettings) -> SqliteStore | None:
    """Open the local run history, or return ``None`` if it is unavailable.

    History is a convenience, not a prerequisite. A locked file, a read-only
    profile or a missing directory costs the user their run history and nothing
    else, so the failure is logged and the application carries on.
    """
    if not settings.storage.enabled:
        log_event(
            _log, logging.INFO, "storage.disabled", detail="run history is switched off"
        )
        return None
    try:
        return open_store(settings.database_path)
    except Exception as exc:  # noqa: BLE001 - history is optional
        log_event(
            _log,
            logging.WARNING,
            "storage.unavailable",
            path=str(settings.database_path),
            error=f"{type(exc).__name__}: {exc}",
        )
        return None


def build_retriever(settings: AppSettings, *, ai_backend=None) -> Retriever | None:
    """A local retriever over a restored (usually empty) index, or ``None``.

    Nothing is indexed automatically. The corpus stays empty until the user
    indexes something, so the retrieval stage reports itself as skipped with an
    accurate reason instead of returning passages that do not exist. Embeddings
    come from the configured AI backend; in Phase 1 that is the deterministic
    mock, which matches words rather than meaning, and says so in its own
    provenance.
    """
    from ai.registry import build_ai_backend
    from retrieval import InMemoryVectorIndex, LocalRetriever

    try:
        backend = ai_backend if ai_backend is not None else build_ai_backend(settings)
    except LocalOpsError as exc:
        log_event(_log, logging.WARNING, "retrieval.ai_unavailable", error=str(exc))
        return None

    embeddings = backend.embeddings
    if embeddings is None:
        log_event(
            _log,
            logging.INFO,
            "retrieval.unavailable",
            backend=backend.info.name,
            detail="the configured backend provides no embeddings",
        )
        return None

    index = InMemoryVectorIndex(dimension=embeddings.dimension)
    snapshot = settings.index_path
    if settings.retrieval.persist_index and snapshot.is_file():
        try:
            index.load(str(snapshot))
        except Exception as exc:  # noqa: BLE001 - a bad index must not stop start-up
            log_event(
                _log,
                logging.WARNING,
                "retrieval.index_unreadable",
                path=str(snapshot),
                error=f"{type(exc).__name__}: {exc}",
            )
            return None
    return LocalRetriever(embeddings, index)


def build_pipeline(
    settings: AppSettings,
    *,
    store: SqliteStore | None = None,
    retriever: Retriever | None = None,
    ai_backend=None,
    open_history: bool = True,
) -> LocalDiagnosticPipeline:
    """Assemble a pipeline from validated configuration.

    Every collaborator is injectable, so a test can point the store at
    ``tmp_path`` and a caller that genuinely wants no history can pass
    ``open_history=False`` rather than leaving ``store`` ambiguous.
    """
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
    if retriever is None:
        retriever = build_retriever(settings, ai_backend=ai_backend)
    if store is None and open_history:
        store = build_store(settings)
    return LocalDiagnosticPipeline(
        engine,
        executor,
        verifier,
        gate=gate,
        retriever=retriever,
        policy=policy,
        store=store,
        retrieval_top_k=settings.retrieval.top_k,
        retrieval_min_score=settings.retrieval.min_score,
    )


def _run_payload(run: PipelineRun) -> dict[str, object]:
    """The JSON-safe summary written to the run history.

    Deliberately a summary rather than a full copy of the run: the database is a
    searchable record of what happened, not a second source of truth that could
    drift from the live objects.
    """
    diagnosis = run.diagnosis
    return {
        "run_id": run.run_id,
        "created_at": run.created_at,
        "question": run.query.question,
        "outcome": run.outcome.value,
        "stages": [record.as_row() for record in run.stages],
        "reasoning_mode": diagnosis.reasoning_mode.value if diagnosis else "",
        "confidence": diagnosis.confidence if diagnosis else 0.0,
        "evidence_ids": [item.evidence_id for item in run.query.evidence],
        "evidence_count": len(run.query.evidence),
        "hypotheses": [
            {
                "id": hypothesis.hypothesis_id,
                "status": hypothesis.status.value,
                "confidence": hypothesis.confidence,
                "statement": hypothesis.statement,
            }
            for hypothesis in (diagnosis.ranked() if diagnosis else ())
        ],
        "actions": [
            {
                "action_id": action.action_id,
                "title": action.title,
                "risk": action.risk.value,
            }
            for action in (run.plan.actions if run.plan else ())
        ],
    }
