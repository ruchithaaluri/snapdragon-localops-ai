"""Approval and policy: the rules that stop the app acting on its own."""

from __future__ import annotations

import pytest

from core.contracts.remediation import ExecutionOutcome
from core.errors import ActionDeniedError, ApprovalRequiredError
from core.ids import new_id
from core.safety import (
    ApprovalDecision,
    ApprovalRecord,
    ApprovalStatus,
    DenyAllApprovalGate,
    InMemoryApprovalGate,
    RemediationAction,
    RemediationPlan,
    RiskLevel,
    SafetyPolicy,
)
from diagnostics.remediation import LocalRemediationExecutor


def action(**kwargs) -> RemediationAction:
    """A minimal read-only action, overridable per test."""
    defaults = {
        "action_id": new_id("act"),
        "title": "check the interpreter",
        "description": "print the executable",
        "command": 'python -c "print(1)"',
    }
    return RemediationAction(**{**defaults, **kwargs})


# -- construction --------------------------------------------------------


def test_requires_approval_cannot_be_switched_off():
    """Even a caller asking for no approval gets one."""
    assert action(requires_approval=False).requires_approval is True


def test_an_action_needs_an_id_and_a_title():
    with pytest.raises(ValueError):
        RemediationAction(action_id="", title="t", description="d")
    with pytest.raises(ValueError):
        RemediationAction(action_id=new_id("act"), title="", description="d")


def test_policy_refuses_to_disable_approval():
    with pytest.raises(ValueError, match="approval cannot be disabled"):
        SafetyPolicy(require_approval=False)


# -- gates ---------------------------------------------------------------


def test_deny_all_gate_denies_everything():
    gate = DenyAllApprovalGate()
    target = action()
    assert not gate.request(target).is_approved
    assert gate.status(target) is ApprovalStatus.DENIED
    with pytest.raises(ApprovalRequiredError):
        gate.assert_approved(target)


def test_in_memory_gate_starts_pending():
    gate = InMemoryApprovalGate()
    target = action()
    assert gate.status(target) is ApprovalStatus.PENDING
    assert gate.request(target).decision is ApprovalDecision.PENDING


def test_in_memory_gate_records_an_explicit_decision():
    gate = InMemoryApprovalGate()
    target = action()
    gate.approve(target, note="looks fine")
    assert gate.status(target) is ApprovalStatus.APPROVED
    assert gate.get(target.action_id).note == "looks fine"
    gate.revoke(target)
    assert gate.status(target) is ApprovalStatus.PENDING


def test_in_memory_gate_rejects_a_pending_decision():
    with pytest.raises(ValueError):
        InMemoryApprovalGate().decide(action(), ApprovalDecision.PENDING)


def test_assert_approved_raises_for_an_unknown_action():
    with pytest.raises(ApprovalRequiredError):
        InMemoryApprovalGate().assert_approved(action())


# -- executor ------------------------------------------------------------


def test_nothing_runs_without_an_approval(executor):
    assert (
        executor.execute(action(), None).outcome is ExecutionOutcome.NOT_APPROVED
    )


def test_a_denied_action_does_not_run(executor, gate):
    target = action()
    assert (
        executor.execute(target, gate.deny(target)).outcome
        is ExecutionOutcome.NOT_APPROVED
    )


def test_approval_for_another_action_does_not_authorise_this_one(executor):
    """Regression: approving one action once unlocked any other."""
    result = executor.execute(action(), ApprovalRecord.approve("act_someone_else"))
    assert result.outcome is ExecutionOutcome.NOT_APPROVED
    assert "different action" in result.detail


def test_an_approved_action_runs_and_reports_output(executor, gate):
    target = action()
    result = executor.execute(target, gate.approve(target))
    assert result.outcome is ExecutionOutcome.APPLIED
    assert result.exit_code == 0
    assert "1" in result.stdout
    assert result.provenance is not None
    assert not result.provenance.is_npu, "a subprocess is never an NPU run"


def test_policy_denial_beats_approval(executor):
    dangerous = action(
        action_id=new_id("act"), title="nuke", command="format c:", risk=RiskLevel.LOW
    )
    result = executor.execute(dangerous, ApprovalRecord.approve(dangerous.action_id))
    assert result.outcome is ExecutionOutcome.DENIED


def test_mutating_actions_are_blocked_by_default_policy(executor):
    mutating = action(
        action_id=new_id("act"),
        title="delete",
        command="del /f /q build.tmp",
        risk=RiskLevel.LOW,
    )
    result = executor.execute(mutating, ApprovalRecord.approve(mutating.action_id))
    assert result.outcome is ExecutionOutcome.DENIED


def test_risk_above_the_configured_maximum_is_denied():
    executor = LocalRemediationExecutor(SafetyPolicy(max_risk=RiskLevel.READ_ONLY))
    risky = action(action_id=new_id("act"), risk=RiskLevel.HIGH)
    result = executor.execute(risky, ApprovalRecord.approve(risky.action_id))
    assert result.outcome is ExecutionOutcome.DENIED


def test_an_informational_action_is_skipped_not_silently_applied(executor, gate):
    advice = action(command=None)
    result = executor.execute(advice, gate.approve(advice))
    assert result.outcome is ExecutionOutcome.SKIPPED
    assert "no command" in result.detail


def test_executor_reports_its_degraded_health_when_mutation_is_blocked(executor):
    assert executor.health().status.value == "degraded"
    assert "read-only" in executor.health().detail


def test_a_failing_command_is_data_not_an_exception(executor, gate):
    failing = action(command='python -c "import sys; sys.exit(3)"')
    result = executor.execute(failing, gate.approve(failing))
    assert result.outcome is ExecutionOutcome.FAILED
    assert result.exit_code == 3
    assert "exit code 3" in result.detail


def test_a_missing_command_is_reported_not_raised(executor, gate):
    missing = action(command="definitely-not-a-real-binary-xyz")
    result = executor.execute(missing, gate.approve(missing))
    assert result.outcome is ExecutionOutcome.FAILED
    assert "not found" in result.detail


def test_shell_metacharacters_are_not_interpreted(executor, gate, tmp_path):
    """``>`` must reach the program as an argument, not become a redirection."""
    target = tmp_path / "pwned.txt"
    sneaky = action(command=f'python -c "print(1)" > "{target}"')
    executor.execute(sneaky, gate.approve(sneaky))
    assert not target.exists(), "a shell redirect must never be honoured"


def test_execute_plan_only_runs_approved_actions(executor, gate):
    a, b = action(action_id="act_one"), action(action_id="act_two")
    plan = RemediationPlan(run_id="run_x", summary="s", actions=(a, b))
    gate.approve(a)
    results = executor.execute_plan(plan, [gate.get("act_one")])
    outcomes = {r.action_id: r.outcome for r in results}
    assert outcomes["act_one"] is ExecutionOutcome.APPLIED
    assert outcomes["act_two"] is ExecutionOutcome.NOT_APPROVED


# -- policy --------------------------------------------------------------


def test_policy_check_rejects_a_denied_command_directly():
    with pytest.raises(ActionDeniedError):
        SafetyPolicy().check_command("reg delete HKLM\\Software")


def test_deny_list_covers_the_obvious_landmines():
    for command in ("rm -rf /", "format c:", "diskpart", "bcdedit", "cipher /w"):
        with pytest.raises(ActionDeniedError):
            SafetyPolicy().check_command(command)
