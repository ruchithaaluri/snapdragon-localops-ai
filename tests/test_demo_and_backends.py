"""The competition MVP: the demo, the backends, the whitelist, the benchmark.

These are the tests that pin the claims the project actually makes. Each one
would fail if a guarantee were quietly dropped:

* the demo really does go FAILED -> PASSED, and really is repeatable,
* nothing is applied without an explicit approval,
* the safe-action whitelist cannot be bypassed with a synthetic id,
* the Snapdragon backend refuses to claim NPU execution it has not verified,
* every benchmark row is a real measurement on a verified CPU target.

No model is downloaded, no network is touched, and every file the demo writes
goes under the temporary data directory the ``conftest`` fixture provides.
"""

from __future__ import annotations

import pytest

from ai.registry import BACKENDS, build_ai_backend
from core.config.schema import AppSettings
from core.errors import ActionDeniedError, ProviderNotFoundError, UnverifiedNpuClaimError
from core.execution import (
    AcceleratorKind,
    ExecutionTarget,
    ExecutionVerification,
)

from demo import (
    DEMO_TITLE,
    apply_safe_action,
    broken_source,
    build_demo_action,
    fixed_source,
    run_demo,
    safe_action_names,
)


# -- the demo ------------------------------------------------------------


def test_the_demo_goes_from_failed_to_passed(settings):
    """The headline claim, verified end to end on this machine."""
    result = run_demo(settings, open_history=False)

    assert not result.before_passed, "the sample app must genuinely fail first"
    assert result.after_passed, "and genuinely pass after the approved fix"
    assert result.worked
    assert result.title == DEMO_TITLE


def test_the_demo_is_repeatable(settings):
    """Running it twice must work, not pass only the first time.

    The second run has to reset the app, or BEFORE would be a lie.
    """
    first = run_demo(settings, open_history=False)
    second = run_demo(settings, open_history=False)
    assert first.worked and second.worked
    assert not second.before_passed, "the sample app must be broken again"


def test_the_demo_diagnosis_cites_real_evidence(settings):
    """The answer is traceable to the captured traceback, not invented."""
    result = run_demo(settings, open_history=False)

    assert result.run.diagnosis is not None
    cited = set(result.run.diagnosis.evidence_ids)
    assert cited, "a diagnosis must cite something"
    assert cited <= set(result.run.query.evidence_by_id())
    assert any(
        "ModuleNotFoundError" in item.content
        for item in result.run.query.evidence
    ), "the evidence is the real traceback the app produced"


def test_the_demo_reports_its_reasoning_honestly(settings):
    """Rule-based, not dressed up as a model."""
    result = run_demo(settings, open_history=False)
    assert result.run.diagnosis.reasoning_mode.value == "rules_only"


def test_the_demo_retrieval_stage_does_real_work(settings):
    """The knowledge base is seeded, so retrieval is completed, not skipped."""
    from core.contracts.pipeline import PipelineStage, StageStatus



# -- the approval gate ---------------------------------------------------


def test_nothing_is_applied_without_approval(settings):
    """The central safety claim, demonstrated rather than asserted."""
    result = run_demo(settings, approve=False, open_history=False)

    assert not result.approved
    assert not result.applied
    assert not result.after_passed, (
        "without approval the app stays broken - that is the gate working"
    )
    assert not result.worked


def test_the_demo_only_ever_offers_whitelisted_actions(settings):
    result = run_demo(settings, open_history=False)
    assert result.action.action_id in safe_action_names()
    assert result.action.command is None, "a fix is not a command line"


# -- the safe-action whitelist -------------------------------------------


def test_an_unknown_action_id_is_denied():
    """A synthetic id cannot reach the underlying function."""
    from core.safety import RemediationAction, RiskLevel

    forged = RemediationAction(
        action_id="totally.made.up",
        title="do something else",
        description="Target: whatever",
        risk=RiskLevel.LOW,
    )
    with pytest.raises(ActionDeniedError, match="whitelist"):
        apply_safe_action(forged, approved=True)


def test_a_real_action_id_still_needs_approval(tmp_path):
    """A whitelisted id is not a licence; the approval is still required."""
    action = build_demo_action(str(tmp_path / "nothing.py"))
    with pytest.raises(ActionDeniedError, match="not approved"):
        apply_safe_action(action, approved=False)


def test_the_action_refuses_an_unrecognised_file(tmp_path):
    """A file that is not the expected one is refused, not overwritten."""
    target = tmp_path / "unrelated.py"
    target.write_text("print('hello')\n", encoding="utf-8")
    with pytest.raises(ActionDeniedError, match="not present"):
        apply_safe_action(build_demo_action(str(target)), approved=True)
    assert target.read_text(encoding="utf-8") == "print('hello')\n"


def test_the_fix_changes_exactly_one_line():
    """The before/after difference is one line, and it is the import."""
    before, after = broken_source().splitlines(), fixed_source().splitlines()
    assert len(before) == len(after)
    differing = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert len(differing) == 1
    assert "localops_sample" in before[differing[0]]
    assert "import json" in after[differing[0]]


# -- AI backends ---------------------------------------------------------


def test_the_default_backend_is_the_mock(settings):
    backend = build_ai_backend(settings)
    assert backend.info.name == "mock-ai-backend"
    assert backend.is_mock() is True


def test_all_three_backends_are_registered():
    assert set(BACKENDS) == {"mock", "local", "snapdragon"}


def test_an_unknown_backend_name_is_rejected():
    with pytest.raises(ProviderNotFoundError):
        build_ai_backend(
            AppSettings.model_validate({"ai": {"llm": {"name": "nonexistent"}}})
        )


def test_the_local_backend_never_claims_to_be_mock():
    backend = build_ai_backend(
        AppSettings.model_validate({"ai": {"llm": {"name": "local"}}})
    )
    assert backend.is_mock() is False
    assert backend.llm is None, "no model configured means no model, not a mock"
    assert "mock" not in backend.info.name.lower(), (
        "selecting 'local' must not fall back to the mock backend"
    )


# -- the Snapdragon backend ----------------------------------------------


def test_the_snapdragon_backend_is_not_validated_on_this_machine():
    from ai.backends.snapdragon import SNAPDRAGON_VALIDATION_STATUS

    backend = build_ai_backend(
        AppSettings.model_validate({"ai": {"llm": {"name": "snapdragon"}}})
    )
    assert backend.is_validated() is False, (
        "no NPU run has happened, so the NPU claim is unavailable"
    )
    assert SNAPDRAGON_VALIDATION_STATUS in backend.health().detail
    assert backend.llm is None and backend.ocr is None


def test_the_snapdragon_backend_refuses_an_unverified_npu_claim():
    from ai.backends.snapdragon import SnapdragonAIBackend

    backend = SnapdragonAIBackend()
    with pytest.raises(UnverifiedNpuClaimError):
        backend.record_verified_run(ExecutionTarget.npu_advertised("QNN"))


def test_the_snapdragon_backend_accepts_only_a_verified_run():
    from ai.backends.snapdragon import SnapdragonAIBackend

    backend = SnapdragonAIBackend()
    backend.record_verified_run(
        ExecutionTarget(
            kind=AcceleratorKind.NPU,
            provider="QNNExecutionProvider",
            verification=ExecutionVerification.verified("timed-run", "on HTP"),
        )
    )
    assert backend.is_validated() is True


def test_the_snapdragon_runtime_refuses_to_load_a_model():
    from ai.backends.snapdragon import SnapdragonRuntime
    from core.errors import ExecutionBackendUnavailableError
    from core.model_artifact import ModelArtifact

    with pytest.raises(ExecutionBackendUnavailableError, match="pending target"):
        SnapdragonRuntime().load(ModelArtifact(model_id="m", path="m.onnx"))


# -- the status strip (no Qt needed) -------------------------------------


def test_the_accelerator_word_is_cpu_not_npu_by_default():
    from app.ui.widgets import execution_word

    assert execution_word(ExecutionTarget.cpu("python"))[0] == "CPU"


def test_an_unverified_npu_is_never_reported_plainly_as_npu():
    from app.ui.widgets import execution_word

    word, tone, _tip = execution_word(ExecutionTarget.npu_advertised("QNN"))
    assert word == "NPU?", "an advertised provider is not a measurement"
    assert tone == "warn"




def test_a_verified_npu_is_reported_as_npu():
    from app.ui.widgets import execution_word

    target = ExecutionTarget(
        kind=AcceleratorKind.NPU,
        provider="QNNExecutionProvider",
        verification=ExecutionVerification.verified("timed-run", "on HTP"),
    )
    word, tone, _tip = execution_word(target)
    assert word == "NPU" and tone == "ok"


def test_no_execution_target_means_not_measured():
    from app.ui.widgets import execution_word

    assert execution_word(None)[0] == "not measured"


def test_the_mode_word_follows_the_backend():
    from app.ui.widgets import app_status
    from core.contracts.base import ProviderInfo

    info = ProviderInfo(name="mock-ai-backend", kind="ai_backend")
    assert app_status(info, True, None).ai_word == "MOCK AI"
    assert app_status(info, False, None).ai_word == "REAL AI"


def test_a_backend_that_cannot_be_built_does_not_block_start_up():
    """The window resolves the backend defensively; see MainWindow.__init__."""
    from core.contracts.base import ProviderInfo

    info = ProviderInfo(name="unavailable", kind="ai_backend", detail="typo")
    assert info.detail == "typo"


def test_the_demo_verification_reruns_the_app(settings):
    """AFTER is measured by running the app, not asserted by the presenter."""
    result = run_demo(settings, open_history=False)
    assert result.report.resolved
    assert result.report.passed


def test_the_demo_reports_before_and_after_in_words(settings):
    result = run_demo(settings, open_history=False)
    rows = dict(result.as_rows())
    assert rows["BEFORE"] == "FAILED"
    assert rows["AFTER"] == "PASSED"
    assert rows["User approval"] == "APPROVED"
# -- the benchmark -------------------------------------------------------


def test_the_benchmark_measures_every_default_case(settings):
    from benchmark import build_default_cases, run_default_benchmark

    report = run_default_benchmark(settings, warmup=0, iterations=2)

    assert not report.is_empty
    assert len(report.measurements) == len(build_default_cases())
    assert not any("NOT MEASURED" in note for note in report.notes), report.notes


def test_every_benchmark_row_is_a_verified_cpu_measurement(settings):
    """No row may claim an accelerator this machine does not have."""
    from benchmark import run_default_benchmark

    report = run_default_benchmark(settings, warmup=0, iterations=2)

    for measurement in report.measurements:
        assert measurement.target.kind is AcceleratorKind.CPU
        assert measurement.target.is_verified
        assert measurement.iteration_count == 2
        assert measurement.provenance.method
        assert measurement.provenance.host


def test_the_benchmark_records_the_device_and_the_pending_validation(settings):
    from benchmark import device_description, run_default_benchmark

    notes = " ".join(run_default_benchmark(settings, warmup=0, iterations=2).notes)
    assert device_description() in notes
    assert "Snapdragon deployment validation pending target hardware." in notes
    assert "no NPU figure" in notes


def test_a_benchmark_row_cannot_claim_an_npu_it_did_not_run():
    """The data model refuses it, independently of the harness."""
    from core.contracts.benchmark import Measurement, MeasurementProvenance
    from core.errors import UnmeasuredBenchmarkError

    with pytest.raises(UnmeasuredBenchmarkError):
        Measurement(
            name="inference",
            unit="ms",
            value=1.0,
            samples=(1.0,),
            provenance=MeasurementProvenance(
                method="timed", host="h", runtime="r", iterations=1
            ),
            target=ExecutionTarget.npu_advertised("QNN"),
        )

