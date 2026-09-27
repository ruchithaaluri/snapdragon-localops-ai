"""The diagnostic engine, the planner, and the pipeline that joins them."""

from __future__ import annotations

import pytest

from core.contracts.diagnostics import (
    DiagnosticQuery,
    Evidence,
    EvidenceKind,
    ReasoningMode,
)
from core.contracts.pipeline import PipelineStage, StageOutcome, StageStatus
from core.errors import InsufficientEvidenceError
from app.pipeline import build_pipeline
from diagnostics import RuleBasedDiagnosticEngine, build_query
from diagnostics.remediation import RuleBasedRemediationPlanner

from .conftest import SAMPLE_LOG


# -- engine --------------------------------------------------------------


def test_a_diagnosis_cites_its_evidence(engine, query):
    diagnosis = engine.analyze(query)
    assert diagnosis.evidence_ids
    for hypothesis in diagnosis.hypotheses:
        assert hypothesis.evidence_ids, "a hypothesis with no citation cannot exist"
        assert set(hypothesis.evidence_ids) <= set(diagnosis.evidence_ids)


def test_cited_ids_really_exist_in_the_query(engine, query):
    diagnosis = engine.analyze(query)
    known = set(query.evidence_by_id())
    assert set(diagnosis.evidence_ids) <= known


def test_reasoning_mode_is_rules_only_and_provenance_says_cpu(engine, query):
    diagnosis = engine.analyze(query)
    assert diagnosis.reasoning_mode is ReasoningMode.RULES_ONLY
    assert diagnosis.provenance is not None
    assert diagnosis.provenance.execution.kind.value == "cpu"
    assert not diagnosis.provenance.is_npu
    assert "no model inference" in diagnosis.provenance.execution.verification.detail


def test_a_rules_engine_never_claims_high_confidence(engine, query):
    diagnosis = engine.analyze(query)
    assert diagnosis.confidence <= 0.95
    assert all(h.confidence <= 0.95 for h in diagnosis.hypotheses)


def test_several_categories_are_reported_separately(engine, query):
    statements = " ".join(h.statement for h in engine.analyze(query).hypotheses)
    assert "Python" in statements
    assert "Node / TypeScript" in statements
    assert "Windows" in statements


def test_hypotheses_come_back_ranked(engine, query):
    ranked = engine.analyze(query).ranked()
    assert [h.confidence for h in ranked] == sorted(
        (h.confidence for h in ranked), reverse=True
    )


def test_insufficient_evidence_raises_instead_of_guessing(engine):
    query = build_query("Why is it slow?", "everything looks fine to me")
    with pytest.raises(InsufficientEvidenceError) as excinfo:
        engine.analyze(query)
    assert excinfo.value.code == "insufficient_evidence"
    assert "hint" in excinfo.value.context


def test_a_tolerant_engine_reports_the_absence_instead_of_raising():
    tolerant = RuleBasedDiagnosticEngine(require_evidence=False)
    diagnosis = tolerant.analyze(
        build_query("Why is it slow?", "everything looks fine to me")
    )
    assert diagnosis.gaps
    assert "No recognised error pattern" in diagnosis.summary


def test_user_input_alone_is_never_treated_as_a_fault(engine):
    """Context is not proof."""
    query = DiagnosticQuery(
        question="what now?",
        evidence=(Evidence(kind=EvidenceKind.USER_INPUT, content="my code is broken"),),
    )
    with pytest.raises(InsufficientEvidenceError):
        engine.analyze(query)


def test_a_low_confidence_lead_is_marked_inconclusive():
    engine = RuleBasedDiagnosticEngine(confidence_threshold=0.99)
    diagnosis = engine.analyze(build_query("why?", "Error: something vague happened\n"))
    assert all(h.status.value == "inconclusive" for h in diagnosis.hypotheses)


def test_a_generic_catch_all_is_capped_as_a_lead():
    diagnosis = RuleBasedDiagnosticEngine().analyze(
        build_query("why?", "Error: something vague happened\n")
    )
    assert diagnosis.confidence <= 0.55


def test_engine_rejects_nonsense_configuration():
    with pytest.raises(ValueError):
        RuleBasedDiagnosticEngine(confidence_threshold=1.5)
    with pytest.raises(ValueError):
        RuleBasedDiagnosticEngine(max_hypotheses=0)


# -- planner -------------------------------------------------------------


def test_the_planner_only_proposes_read_only_actions(engine, query):
    plan = RuleBasedRemediationPlanner(query.evidence).plan(engine.analyze(query))
    assert plan.actions
    for action in plan.actions:
        assert action.risk.value == "read_only"
        assert action.requires_approval is True


def test_every_proposed_command_is_quoted_from_the_table(engine, query):
    from diagnostics.remediation.planner import _TEMPLATES

    allowed = {t.command for t in _TEMPLATES if t.command}
    plan = RuleBasedRemediationPlanner(query.evidence).plan(engine.analyze(query))
    for action in plan.actions:
        assert action.command is None or action.command in allowed


def test_the_planner_proposes_but_never_executes(engine, query):
    planner = RuleBasedRemediationPlanner(query.evidence)
    assert planner.plan(engine.analyze(query)).actions
    assert "runs nothing" in planner.health().detail


def test_the_planner_rejects_a_non_diagnosis():
    with pytest.raises(TypeError):
        RuleBasedRemediationPlanner().plan("not a diagnosis")


def test_evidence_without_a_suggestion_yields_no_action(engine):
    query = DiagnosticQuery(
        question="why?",
        evidence=(
            Evidence(
                kind=EvidenceKind.LOG_FILE,
                content="Access is denied.",
                metadata={
                    "signature": "unknown_future_signature",
                    "category": "windows",
                },
            ),
        ),
    )
    plan = RuleBasedRemediationPlanner(query.evidence).plan(engine.analyze(query))
    assert plan.actions == ()
    assert "No automatic check" in plan.summary


# -- pipeline ------------------------------------------------------------


def test_the_pipeline_records_every_stage(settings):
    run = build_pipeline(settings).run(
        build_query("why?", SAMPLE_LOG, source="app.log")
    )
    assert {record.stage.value for record in run.stages} == {
        "capture",
        "evidence",
        "retrieval",
        "diagnosis",
        "remediation",
        "approval",
        "verification",
    }


def test_a_skipped_stage_says_why(settings):
    run = build_pipeline(settings).run(
        build_query("why?", SAMPLE_LOG, source="app.log")
    )
    retrieval = run.stage(PipelineStage.RETRIEVAL)
    assert retrieval is not None
    assert retrieval.status is StageStatus.SKIPPED
    assert retrieval.detail, "a skipped stage must explain itself"


def test_the_pipeline_stops_at_the_approval_gate(settings):
    run = build_pipeline(settings).run(
        build_query("why?", SAMPLE_LOG, source="app.log")
    )
    assert run.outcome is StageOutcome.AWAITING_APPROVAL
    assert run.remediation_results == ()
    assert run.plan and run.plan.actions


def test_auto_approve_cannot_bypass_the_gate(settings):
    """``auto_approve`` is a label, never a permission."""
    run = build_pipeline(settings).run(
        build_query("why?", SAMPLE_LOG, source="app.log"), auto_approve=True
    )
    assert run.remediation_results == ()
    assert run.outcome is StageOutcome.AWAITING_APPROVAL


def test_applying_the_plan_runs_only_approved_actions(settings):
    pipeline = build_pipeline(settings)
    run = pipeline.run(build_query("why?", SAMPLE_LOG, source="app.log"))
    assert run.plan is not None
    first = run.plan.actions[0]
    pipeline.gate.approve(first)
    results = pipeline.apply_plan(run.plan, [pipeline.gate.get(first.action_id)])
    assert results[0].outcome.value == "applied"
    assert all(
        r.outcome.value == "not_approved"
        for r in results
        if r.action_id != first.action_id
    )


def test_a_run_with_no_evidence_is_unresolved_not_failed(settings):
    run = build_pipeline(settings).run(build_query("why?", "nothing interesting here"))
    assert run.outcome is StageOutcome.UNRESOLVED
    assert run.diagnosis is None
    stage = run.stage(PipelineStage.DIAGNOSIS)
    assert stage.status is StageStatus.FAILED
    assert "not enough evidence" in stage.detail


def test_the_pipeline_never_claims_an_npu(settings):
    run = build_pipeline(settings).run(
        build_query("why?", SAMPLE_LOG, source="app.log")
    )
    assert run.diagnosis is not None
    assert run.diagnosis.provenance is not None
    assert not run.diagnosis.provenance.is_npu
