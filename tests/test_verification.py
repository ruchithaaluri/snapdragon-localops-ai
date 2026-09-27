"""The verification runner: failures are data, approval is required."""

from __future__ import annotations

import pytest

from core.contracts.verification import CheckKind, CheckOutcome, VerificationCheck
from core.ids import new_id
from core.safety import RemediationAction, RiskLevel
from verification import LocalVerificationRunner, checks_for_plan

from .conftest import SAMPLE_LOG


def approve_all(gate, checks) -> None:
    """Record an approval for each check, mirroring the gate's protocol."""
    for check in checks:
        gate.approve(
            RemediationAction(
                action_id=check.check_id,
                title=check.description,
                description=check.description,
                risk=check.risk,
            )
        )


def test_the_default_gate_verifies_nothing():
    """A headless run must not check anything until someone has agreed."""
    report = LocalVerificationRunner().run(
        [VerificationCheck(CheckKind.FILE_EXISTS, "x", parameters={"path": __file__})]
    )
    assert report.results[0].outcome is CheckOutcome.SKIPPED
    # The default gate is deny-all, so the record is an explicit denial.
    assert "denied" in report.results[0].detail


def test_file_checks_read_the_disk(verifier, gate, log_file):
    checks = [
        VerificationCheck(
            CheckKind.FILE_EXISTS, "the log exists", parameters={"path": str(log_file)}
        ),
        VerificationCheck(
            CheckKind.FILE_EXISTS, "nope", parameters={"path": str(log_file) + ".missing"}
        ),
        VerificationCheck(
            CheckKind.FILE_CONTAINS,
            "contains the error",
            parameters={"path": str(log_file), "pattern": "ModuleNotFoundError"},
        ),
        VerificationCheck(
            CheckKind.LOG_PATTERN_ABSENT,
            "error gone from a clean file",
            parameters={"path": str(log_file), "pattern": "ZZZ-never-present"},
        ),
        VerificationCheck(
            CheckKind.LOG_PATTERN_ABSENT,
            "error gone from the real file (it is not)",
            parameters={"path": str(log_file), "pattern": "ModuleNotFoundError"},
        ),
    ]
    approve_all(gate, checks)
    report = verifier.run(checks)
    got = {r.description: r.outcome for r in report}
    assert got["the log exists"] is CheckOutcome.PASSED
    assert got["nope"] is CheckOutcome.FAILED
    assert got["contains the error"] is CheckOutcome.PASSED
    assert got["error gone from a clean file"] is CheckOutcome.PASSED
    assert got["error gone from the real file (it is not)"] is CheckOutcome.FAILED
    assert not report.resolved


def test_command_checks_use_the_exit_code(verifier, gate):
    checks = [
        VerificationCheck(
            CheckKind.COMMAND_SUCCEEDS,
            "exits 0",
            parameters={"command": 'python -c "print(1)"'},
        ),
        VerificationCheck(
            CheckKind.COMMAND_SUCCEEDS,
            "exits 3",
            parameters={"command": 'python -c "import sys; sys.exit(3)"', "exit_code": "3"},
        ),
        VerificationCheck(
            CheckKind.COMMAND_SUCCEEDS,
            "missing binary",
            parameters={"command": "definitely-not-a-real-binary-xyz"},
        ),
    ]
    approve_all(gate, checks)
    got = {r.description: r.outcome for r in verifier.run(checks)}
    assert got["exits 0"] is CheckOutcome.PASSED
    assert got["exits 3"] is CheckOutcome.PASSED
    assert got["missing binary"] is CheckOutcome.ERROR


def test_a_metric_without_a_measurement_is_skipped_not_guessed(verifier, gate):
    check = VerificationCheck(
        CheckKind.METRIC_BELOW, "latency", parameters={"threshold": "100"}
    )
    approve_all(gate, [check])
    assert verifier.run([check]).results[0].outcome is CheckOutcome.SKIPPED


def test_a_broken_check_does_not_end_the_run(verifier, gate, log_file):
    """One bad check must not prevent the rest from being reported."""
    checks = [
        VerificationCheck(CheckKind.FILE_EXISTS, "missing parameter", parameters={}),
        VerificationCheck(
            CheckKind.FILE_CONTAINS,
            "bad regex",
            parameters={"path": str(log_file), "pattern": "([unclosed"},
        ),
        VerificationCheck(
            CheckKind.FILE_EXISTS, "still runs", parameters={"path": str(log_file)}
        ),
    ]
    approve_all(gate, checks)
    report = verifier.run(checks)
    assert len(report) == len(checks), "every check produces exactly one row"
    got = {r.description: r.outcome for r in report}
    assert got["missing parameter"] is CheckOutcome.ERROR
    assert got["bad regex"] is CheckOutcome.ERROR
    assert got["still runs"] is CheckOutcome.PASSED


def test_stop_on_first_failure_marks_the_rest_skipped(gate, log_file):
    runner = LocalVerificationRunner(gate, stop_on_first_failure=True)
    checks = [
        VerificationCheck(
            CheckKind.FILE_EXISTS, "missing", parameters={"path": str(log_file) + ".nope"}
        ),
        VerificationCheck(
            CheckKind.FILE_EXISTS, "present", parameters={"path": str(log_file)}
        ),
    ]
    approve_all(gate, checks)
    report = runner.run(checks)
    assert report.results[0].outcome is CheckOutcome.FAILED
    assert report.results[1].outcome is CheckOutcome.SKIPPED
    assert "stopped" in report.results[1].detail


def test_a_mutating_check_cannot_opt_out_of_approval():
    with pytest.raises(ValueError, match="require approval"):
        VerificationCheck(
            CheckKind.COMMAND_SUCCEEDS,
            "mutating",
            risk=RiskLevel.LOW,
            requires_approval=False,
        )


def test_report_summary_counts_everything(verifier, gate, log_file):
    checks = [
        VerificationCheck(CheckKind.FILE_EXISTS, "ok", parameters={"path": str(log_file)}),
        VerificationCheck(
            CheckKind.FILE_EXISTS, "no", parameters={"path": str(log_file) + ".nope"}
        ),
    ]
    approve_all(gate, checks)
    report = verifier.run(checks)
    assert report.summary() == "1 passed, 1 failed, 0 skipped"
    assert not report.resolved


# -- derived checks ------------------------------------------------------


def test_checks_are_derived_for_a_file_backed_signature(log_file):
    from diagnostics import build_query

    query = build_query("why?", SAMPLE_LOG, source=str(log_file))
    offered = checks_for_plan((), query.evidence)
    assert offered, "a file-backed error must yield a re-readable check"
    assert all(c.kind is CheckKind.LOG_PATTERN_ABSENT for c in offered)
    assert all(c.requires_approval for c in offered)


def test_pasted_text_yields_no_file_check():
    """Pasted text has no file to re-read, so no check is invented."""
    from diagnostics import build_query

    query = build_query("why?", SAMPLE_LOG, source="pasted text")
    assert checks_for_plan((), query.evidence) == ()


def test_an_action_with_a_command_yields_a_command_check():
    action = RemediationAction(
        action_id=new_id("act"),
        title="check",
        description="d",
        command='python -c "print(1)"',
    )
    offered = checks_for_plan([action], ())
    assert len(offered) == 1
    assert offered[0].kind is CheckKind.COMMAND_SUCCEEDS
    assert offered[0].parameters["command"] == action.command


def test_an_action_without_a_command_yields_nothing():
    advice = RemediationAction(
        action_id=new_id("act"), title="t", description="d", command=None
    )
    assert checks_for_plan([advice], ()) == ()
