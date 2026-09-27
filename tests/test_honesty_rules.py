"""The project's central honesty rules, tested directly.

These are the tests that matter most. Each one pins a behaviour that would be
easy to "fix" into something dishonest by accident.
"""

from __future__ import annotations

import pytest

from core.contracts.base import Provenance
from core.contracts.benchmark import BenchmarkCase, Measurement, MeasurementProvenance
from core.errors import UnmeasuredBenchmarkError, UnverifiedNpuClaimError
from core.execution import (
    AcceleratorKind,
    ExecutionPreferences,
    ExecutionTarget,
    ExecutionVerification,
    ExecutionVerificationState,
    FallbackPolicy,
    TargetStatus,
    require_npu_claim_allowed,
    resolve_target,
)

from benchmark import LocalBenchmarkRunner


# -- execution targets ---------------------------------------------------


def test_an_advertised_npu_is_never_a_verified_npu():
    target = ExecutionTarget.npu_advertised("QNNExecutionProvider")
    assert target.kind is AcceleratorKind.NPU
    assert not target.is_verified
    assert not target.npu_execution_claimed
    assert target.status is TargetStatus.ADVERTISED_NPU


def test_a_verified_npu_must_name_its_method():
    with pytest.raises(ValueError):
        ExecutionVerification(state=ExecutionVerificationState.VERIFIED)
    assert ExecutionVerification.verified("timed-run", "forward pass").is_verified


def test_resolving_a_target_refuses_to_select_an_unproven_npu():
    """With a verified CPU present, the unproven NPU is skipped, not used."""
    decision = resolve_target(
        [
            ExecutionTarget.npu_advertised("QNNExecutionProvider"),
            ExecutionTarget.cpu("CPUExecutionProvider"),
        ],
        [AcceleratorKind.NPU],
        provider="onnxruntime",
        policy=FallbackPolicy.ALLOW,
    )
    assert decision.fell_back, "an unproven NPU must never be selected"
    assert decision.target.kind is AcceleratorKind.CPU
    assert decision.target.status is TargetStatus.FALLBACK_CPU
    assert "unproven" in decision.reason


def test_resolving_with_no_usable_target_raises():
    """An advertised NPU alone is not a usable target: there is nothing to run on."""
    with pytest.raises(Exception):
        resolve_target(
            [ExecutionTarget.npu_advertised("QNN")],
            [AcceleratorKind.NPU],
            provider="onnxruntime",
        )


def test_deny_policy_refuses_a_cpu_fallback():
    with pytest.raises(Exception):
        resolve_target(
            [ExecutionTarget.npu_advertised("QNN"), ExecutionTarget.cpu("python")],
            [AcceleratorKind.NPU],
            provider="onnxruntime",
            policy=FallbackPolicy.DENY,
        )


def test_an_npu_preference_cannot_disable_verification():
    with pytest.raises(ValueError):
        ExecutionPreferences(
            preferred=(AcceleratorKind.NPU,), require_verified_npu=False
        )


def test_require_npu_claim_allowed_catches_a_planted_claim():
    with pytest.raises(UnverifiedNpuClaimError):
        require_npu_claim_allowed([ExecutionTarget.npu_advertised("QNN")])


# -- provenance ----------------------------------------------------------


def test_provenance_refuses_an_unverified_npu_claim():
    with pytest.raises(UnverifiedNpuClaimError):
        Provenance(
            provider="fake", execution=ExecutionTarget.npu_advertised("QNN")
        )


def test_provenance_allows_a_verified_npu_run():
    provenance = Provenance(
        provider="snapdragon-backend",
        execution=ExecutionTarget(
            kind=AcceleratorKind.NPU,
            provider="QNNExecutionProvider",
            verification=ExecutionVerification.verified("timed-run", "5 runs on HTP"),
        ),
    )
    assert provenance.is_npu


def test_provenance_defaults_to_unknown_not_verified():
    target = ExecutionTarget(kind=AcceleratorKind.CPU, provider="python")
    assert not target.is_verified
    assert target.status is TargetStatus.UNKNOWN


# -- measurements -------------------------------------------------------


def _provenance() -> MeasurementProvenance:
    return MeasurementProvenance(
        method="timed run", host="test", runtime="python", iterations=1
    )


def test_a_measurement_needs_real_samples():
    with pytest.raises(UnmeasuredBenchmarkError):
        Measurement(
            name="inference",
            unit="ms",
            value=1.0,
            samples=(),
            provenance=_provenance(),
            target=ExecutionTarget.cpu("python"),
        )


def test_a_measurement_needs_a_method():
    """A blank method is rejected by the provenance that carries it."""
    with pytest.raises(ValueError, match="method"):
        MeasurementProvenance(
            method="  ", host="h", runtime="r", iterations=1
        )


def test_provenance_needs_at_least_one_iteration():
    with pytest.raises(ValueError):
        MeasurementProvenance(method="timed", host="h", runtime="r", iterations=0)


def test_an_unverified_npu_row_cannot_be_recorded():
    with pytest.raises(UnmeasuredBenchmarkError):
        Measurement(
            name="inference",
            unit="ms",
            value=1.0,
            samples=(1.0,),
            provenance=_provenance(),
            target=ExecutionTarget.npu_advertised("QNN"),
        )


def test_the_benchmark_runner_refuses_an_npu_target_it_cannot_verify():
    runner = LocalBenchmarkRunner(ExecutionTarget.npu_advertised("QNN"))
    with pytest.raises(UnmeasuredBenchmarkError):
        runner.run(
            [
                BenchmarkCase(
                    "c", "work", lambda: None, iterations=1, warmup_iterations=0
                )
            ]
        )


def test_a_failed_case_is_noted_rather_than_dropped():
    def boom() -> None:
        raise RuntimeError("no model")

    report = LocalBenchmarkRunner().run(
        [BenchmarkCase("c", "boom", boom, iterations=1, warmup_iterations=0)]
    )
    assert report.is_empty
    assert report.notes and "NOT MEASURED" in report.notes[0]
    assert "No measurements were taken" in report.to_markdown()


def test_statistics_need_enough_samples():
    from core.contracts.benchmark import Statistic

    few = LocalBenchmarkRunner().run(
        [
            BenchmarkCase(
                "c", "x", lambda: None, iterations=2, warmup_iterations=0
            )
        ]
    ).measurements[0]
    assert Statistic.MEAN in few.statistics
    assert Statistic.STDEV not in few.statistics, "two samples cannot support a stdev"


def test_report_serialises_to_plain_data():
    import json

    report = LocalBenchmarkRunner().run(
        [
            BenchmarkCase(
                "c", "x", lambda: None, iterations=1, warmup_iterations=0
            )
        ]
    )
    payload = report.to_dict()
    json.dumps(payload)  # must not raise
    assert payload["measurements"][0]["target"]["kind"] == "cpu"
    assert payload["measurements"][0]["target"]["verification"] == "verified"
