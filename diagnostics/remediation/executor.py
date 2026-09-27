"""Applying an approved action - and refusing, safely, to do otherwise.

The order of checks in :meth:`LocalRemediationExecutor.execute` is the whole
point of this module, and it never varies:

1. **Policy first.** A command the safety policy forbids is denied even if the
   user approved it. Approval is permission from the user, not a way around the
   configuration.
2. **Approval second.** No approval record for this exact action id means
   nothing runs - not "probably" nothing, nothing.
3. **Execution last**, and only for the specific action that was approved.

Commands run through :mod:`shlex` and are executed **without a shell**. That
removes a class of quoting and injection problems outright, and it means a
command containing shell metacharacters fails loudly instead of being
silently reinterpreted.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path

from core.contracts.base import (
    Capability,
    ProviderHealth,
    ProviderInfo,
    ProviderStatus,
    Provenance,
)
from core.contracts.remediation import (
    ExecutionOutcome,
    RemediationExecutor,
    RemediationResult,
)
from core.errors import ActionDeniedError
from core.execution import ExecutionTarget
from core.observability import get_logger, log_event
from core.safety import ApprovalRecord, RemediationAction, SafetyPolicy
from core.timing import measure
from core.version import __version__

__all__ = ["LocalRemediationExecutor"]

_log = get_logger(__name__)

#: How much of a subprocess's output is kept. Enough to diagnose, bounded so a
#: runaway command cannot exhaust memory.
_MAX_CAPTURED_CHARS = 20_000


def _parse_argv(command: str) -> list[str] | None:
    """Split a command into argv, or return ``None`` when it cannot be split.

    Returning ``None`` is a failure, not a reason to fall back to a shell: the
    caller records the action as failed instead of reinterpreting the string.
    """
    try:
        argv = shlex.split(command, posix=False)
    except ValueError:
        return None
    # posix=False keeps the surrounding quotes, which would otherwise reach the
    # child process as literal characters.
    return [part.strip('"') for part in argv] or None


class LocalRemediationExecutor(RemediationExecutor):
    """Runs approved, policy-permitted commands in a subprocess.

    Parameters
    ----------
    policy:
        The safety policy, consulted before every execution.
    timeout_seconds:
        Upper bound on a single command. A command that hangs is reported as
        failed rather than blocking the UI indefinitely.
    working_directory:
        Directory commands run in. Defaults to the action's own
        ``working_directory``, then to the process's current directory.
    """

    kind = "remediation_executor"

    def __init__(
        self,
        policy: SafetyPolicy,
        *,
        timeout_seconds: float = 30.0,
        working_directory: str | Path | None = None,
    ) -> None:
        super().__init__(policy)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._timeout = timeout_seconds
        self._working_directory = Path(working_directory) if working_directory else None

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="local-executor",
            kind=self.kind,
            version=__version__,
            capabilities=frozenset({Capability.REMEDIATION_PLANNING}),
            detail=(
                f"policy max risk {self.policy.max_risk.value}; mutating "
                f"{'allowed' if self.policy.permits_remediation else 'blocked'}"
            ),
        )

    def health(self) -> ProviderHealth:
        if not self.policy.permits_remediation:
            return ProviderHealth(
                ProviderStatus.DEGRADED,
                "read-only actions only; mutating remediation is disabled by policy",
            )
        return ProviderHealth.ready("remediation permitted up to the configured risk")

    def execute(
        self,
        action: RemediationAction,
        approval: ApprovalRecord | None,
    ) -> RemediationResult:
        """Apply one action if, and only if, it is approved and permitted."""
        # 1. Policy. Checked before approval, and never skipped.
        try:
            self.policy.check_action(action)
        except ActionDeniedError as exc:
            log_event(
                _log,
                logging.WARNING,
                "remediation.denied",
                action_id=action.action_id,
                reason=str(exc),
            )
            return RemediationResult.denied(action, str(exc))

        # 2. Approval, for this exact action. The action id must match:
        #    approving one action must never unlock another.
        if approval is not None and approval.is_approved:
            if approval.action_id != action.action_id:
                log_event(
                    _log,
                    logging.WARNING,
                    "remediation.approval_mismatch",
                    action_id=action.action_id,
                    approved_id=approval.action_id,
                )
                return RemediationResult(
                    action_id=action.action_id,
                    outcome=ExecutionOutcome.NOT_APPROVED,
                    detail=(
                        "the approval refers to a different action "
                        f"({approval.action_id}) and does not apply here"
                    ),
                )
        else:
            log_event(
                _log,
                logging.INFO,
                "remediation.not_approved",
                action_id=action.action_id,
                decision=approval.decision.value if approval else "absent",
            )
            return RemediationResult.not_approved(action)

        # 3. Execution.
        if not action.command:
            return RemediationResult(
                action_id=action.action_id,
                outcome=ExecutionOutcome.SKIPPED,
                detail=(
                    "this action carries no command; it is informational and was "
                    "approved for reading only"
                ),
                provenance=self._provenance(0.0),
            )
        return self._run(action)


    # -- internals --------------------------------------------------------

    def _run(self, action: RemediationAction) -> RemediationResult:
        """Run one approved command and turn its outcome into a result."""
        argv = _parse_argv(action.command or "")
        if argv is None:
            return RemediationResult(
                action_id=action.action_id,
                outcome=ExecutionOutcome.FAILED,
                detail="command could not be parsed; refusing to guess its meaning",
                stderr=f"unparsable command: {action.command}",
            )

        cwd = (
            Path(action.working_directory)
            if action.working_directory
            else self._working_directory
        )
        if cwd is not None and not cwd.is_dir():
            return RemediationResult(
                action_id=action.action_id,
                outcome=ExecutionOutcome.FAILED,
                detail=f"working directory does not exist: {cwd}",
            )

        log_event(
            _log,
            logging.INFO,
            "remediation.started",
            action_id=action.action_id,
            risk=action.risk.value,
            program=argv[0] if argv else "",
        )
        with measure() as watch:
            try:
                completed = subprocess.run(  # noqa: S603 - argv list, shell=False
                    argv,
                    cwd=str(cwd) if cwd else None,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    check=False,
                    shell=False,
                )
            except FileNotFoundError as exc:
                return self._failed(
                    action,
                    watch.milliseconds,
                    detail=f"command not found: {argv[0]}",
                    stderr=str(exc),
                )
            except subprocess.TimeoutExpired:
                return self._failed(
                    action,
                    watch.milliseconds,
                    detail=f"timed out after {self._timeout:.0f}s",
                )
            except OSError as exc:  # pragma: no cover - platform dependent
                return self._failed(action, watch.milliseconds, detail=str(exc))

        succeeded = completed.returncode == 0
        log_event(
            _log,
            logging.INFO if succeeded else logging.WARNING,
            "remediation.completed",
            action_id=action.action_id,
            exit_code=completed.returncode,
            duration_ms=round(watch.milliseconds, 1),
        )
        return RemediationResult(
            action_id=action.action_id,
            outcome=ExecutionOutcome.APPLIED if succeeded else ExecutionOutcome.FAILED,
            detail="" if succeeded else f"exit code {completed.returncode}",
            exit_code=completed.returncode,
            stdout=completed.stdout[:_MAX_CAPTURED_CHARS],
            stderr=completed.stderr[:_MAX_CAPTURED_CHARS],
            duration_ms=watch.milliseconds,
            provenance=self._provenance(watch.milliseconds),
        )

    def _failed(
        self,
        action: RemediationAction,
        elapsed_ms: float,
        *,
        detail: str,
        stderr: str = "",
    ) -> RemediationResult:
        """Record a failure as data and log it. A failure is never an exception."""
        log_event(
            _log,
            logging.WARNING,
            "remediation.failed",
            action_id=action.action_id,
            detail=detail,
        )
        return RemediationResult(
            action_id=action.action_id,
            outcome=ExecutionOutcome.FAILED,
            detail=detail,
            stderr=stderr,
            duration_ms=elapsed_ms,
            provenance=self._provenance(elapsed_ms),
        )

    def _provenance(self, elapsed_ms: float) -> Provenance:
        """Where the command ran, always reported as a local process."""
        return Provenance(
            provider="local-executor",
            execution=ExecutionTarget.cpu(
                "subprocess", detail="approved command run on this machine"
            ),
            elapsed_ms=elapsed_ms,
            notes=("no shell", "approval recorded"),
        )

