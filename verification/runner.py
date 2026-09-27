"""The local verification runner: execute checks, report what was observed.

Each :class:`~core.contracts.verification.CheckKind` maps to a small read-only
function here. The runner's job is bookkeeping, and it is deliberately boring:

* run every check, in order, unless ``stop_on_first_failure`` is set,
* ask the approval gate first - an unapproved check is reported
  :attr:`~core.contracts.verification.CheckOutcome.SKIPPED`, never silently run,
* record the observed value, so a pass is auditable rather than a bare "ok",
* turn an unexpected exception in one check into an
  :attr:`~core.contracts.verification.CheckOutcome.ERROR` row so the rest of the
  run still completes.

Commands run through :mod:`shlex` with ``shell=False``, for the same reason the
remediation executor does: a command string must never be reinterpreted by a
shell.
"""

from __future__ import annotations

import logging
import re
import shlex
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from core.contracts.base import Capability, ProviderHealth, ProviderInfo
from core.contracts.verification import (
    CheckKind,
    CheckOutcome,
    CheckResult,
    VerificationCheck,
    VerificationReport,
    VerificationRunner,
)
from core.errors import LocalOpsError
from core.observability import get_logger, log_event
from core.safety import ApprovalGate, DenyAllApprovalGate
from core.timing import measure
from core.version import __version__

__all__ = ["LocalVerificationRunner"]

_log = get_logger(__name__)

#: How much file text a single text check will read.
_MAX_READ_BYTES = 4 * 1024 * 1024


class LocalVerificationRunner(VerificationRunner):
    """Runs read-only verification checks on this machine.

    Parameters
    ----------
    approval_gate:
        Consulted for every check whose ``requires_approval`` is set. Defaults to
        :class:`~core.safety.DenyAllApprovalGate`, so nothing is verified until
        the user has agreed to the specific checks on screen.
    stop_on_first_failure:
        Stop the run at the first failure instead of checking everything. Later
        checks are reported as skipped.
    """

    kind = "verification"

    def __init__(
        self,
        approval_gate: ApprovalGate | None = None,
        *,
        stop_on_first_failure: bool = False,
    ) -> None:
        self._gate: ApprovalGate = approval_gate or DenyAllApprovalGate()
        self._stop_on_first_failure = stop_on_first_failure

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="local-verifier",
            kind=self.kind,
            version=__version__,
            capabilities=frozenset({Capability.VERIFICATION_EXECUTION}),
            detail="read-only checks: file, log pattern, exit code, metric, process",
        )

    def health(self) -> ProviderHealth:
        return ProviderHealth.ready("read-only checks; no state is modified")

    def run(self, checks: Sequence[VerificationCheck]) -> VerificationReport:
        """Run ``checks`` and return a report. Individual failures are not raised."""
        results: list[CheckResult] = []
        stopped = False

        for check in checks:
            if stopped:
                results.append(
                    CheckResult(
                        check_id=check.check_id,
                        kind=check.kind,
                        outcome=CheckOutcome.SKIPPED,
                        description=check.description,
                        detail="run stopped after an earlier failure",
                    )
                )
                continue

            result = self._run_one(check)
            results.append(result)
            if self._stop_on_first_failure and not result.passed:
                stopped = True

        report = VerificationReport(results=tuple(results))
        log_event(
            _log,
            logging.INFO,
            "verification.completed",
            summary=report.summary(),
            resolved=report.resolved,
        )
        return report

    # -- internals --------------------------------------------------------

    def _run_one(self, check: VerificationCheck) -> CheckResult:
        """Approval, then dispatch, with every failure captured as a row."""
        if check.requires_approval:
            record = self._gate.request(_as_action(check))
            if not record.is_approved:
                return CheckResult(
                    check_id=check.check_id,
                    kind=check.kind,
                    outcome=CheckOutcome.SKIPPED,
                    description=check.description,
                    expected=check.expectation,
                    detail=(
                        "not approved yet"
                        if record.decision.value == "pending"
                        else f"approval was {record.decision.value}"
                    ),
                )

        handler = _HANDLERS.get(check.kind)
        if handler is None:
            return CheckResult(
                check_id=check.check_id,
                kind=check.kind,
                outcome=CheckOutcome.ERROR,
                description=check.description,
                detail=f"no implementation for {check.kind.value}",
            )

        with measure() as watch:
            try:
                outcome, observed, detail = handler(self, check)
            except LocalOpsError as exc:
                outcome, observed, detail = CheckOutcome.ERROR, "", str(exc)
            except Exception as exc:  # noqa: BLE001 - one bad check must not end the run
                outcome, observed, detail = CheckOutcome.ERROR, "", f"{type(exc).__name__}: {exc}"

        return CheckResult(
            check_id=check.check_id,
            kind=check.kind,
            outcome=outcome,
            description=check.description,
            expected=check.expectation,
            observed=observed,
            detail=detail,
            duration_ms=watch.milliseconds,
        )

    # -- individual checks ------------------------------------------------

    def _check_file_exists(self, check: VerificationCheck) -> tuple[CheckOutcome, str, str]:
        path = _require_path(check, "path")
        exists = path.is_file()
        return (
            CheckOutcome.PASSED if exists else CheckOutcome.FAILED,
            "present" if exists else "absent",
            str(path),
        )

    def _check_file_contains(self, check: VerificationCheck) -> tuple[CheckOutcome, str, str]:
        path = _require_path(check, "path")
        regex = _compile(_require(check, "pattern"))
        found = _search(path, regex)
        return (
            CheckOutcome.PASSED if found is not None else CheckOutcome.FAILED,
            f"line {found}" if found is not None else "not found",
            path.name if path.is_file() else f"{path} does not exist",
        )

    def _check_log_pattern_absent(
        self, check: VerificationCheck
    ) -> tuple[CheckOutcome, str, str]:
        path = _require_path(check, "path")
        regex = _compile(_require(check, "pattern"))
        if not path.is_file():
            # An absent log is the desired end state for this check.
            return CheckOutcome.PASSED, "no such file", str(path)
        found = _search(path, regex)
        return (
            CheckOutcome.PASSED if found is None else CheckOutcome.FAILED,
            "absent" if found is None else f"present at line {found}",
            str(path),
        )

    def _check_command_succeeds(
        self, check: VerificationCheck
    ) -> tuple[CheckOutcome, str, str]:
        command = _require(check, "command")
        expected_code = int(check.parameters.get("exit_code", "0"))
        argv = [part.strip('"') for part in shlex.split(command, posix=False)]
        if not argv:
            return CheckOutcome.ERROR, "", "command is empty"

        try:
            completed = subprocess.run(  # noqa: S603 - argv list, shell=False
                argv,
                capture_output=True,
                text=True,
                timeout=check.timeout_seconds,
                check=False,
                shell=False,
            )
        except FileNotFoundError as exc:
            return CheckOutcome.ERROR, "", f"command not found: {argv[0]} ({exc})"
        except subprocess.TimeoutExpired:
            return CheckOutcome.ERROR, "", f"timed out after {check.timeout_seconds:.0f}s"

        ok = completed.returncode == expected_code
        return (
            CheckOutcome.PASSED if ok else CheckOutcome.FAILED,
            f"exit {completed.returncode}",
            (completed.stdout or completed.stderr).strip()[:400],
        )

    def _check_metric_below(self, check: VerificationCheck) -> tuple[CheckOutcome, str, str]:
        raw = check.parameters.get("value", "")
        threshold_raw = check.parameters.get("threshold", "")
        if not raw or not threshold_raw:
            return CheckOutcome.SKIPPED, "", "no measurement supplied a value"
        try:
            value, threshold = float(raw), float(threshold_raw)
        except ValueError as exc:
            return CheckOutcome.ERROR, "", f"non-numeric value: {exc}"
        return (
            CheckOutcome.PASSED if value < threshold else CheckOutcome.FAILED,
            f"{value:g} (threshold {threshold:g})",
            "",
        )

    def _check_process_running(
        self, check: VerificationCheck
    ) -> tuple[CheckOutcome, str, str]:
        name = _require(check, "name")
        listing = _process_list()
        if listing is None:
            return CheckOutcome.SKIPPED, "", "process listing is not available here"
        running = name.lower() in listing
        return (
            CheckOutcome.PASSED if running else CheckOutcome.FAILED,
            "running" if running else "not running",
            name,
        )


#: Maps each check kind to its implementation. Every entry is read-only.
_HANDLERS: dict[CheckKind, Callable[..., tuple[CheckOutcome, str, str]]] = {
    CheckKind.FILE_EXISTS: LocalVerificationRunner._check_file_exists,
    CheckKind.FILE_CONTAINS: LocalVerificationRunner._check_file_contains,
    CheckKind.LOG_PATTERN_ABSENT: LocalVerificationRunner._check_log_pattern_absent,
    CheckKind.COMMAND_SUCCEEDS: LocalVerificationRunner._check_command_succeeds,
    CheckKind.METRIC_BELOW: LocalVerificationRunner._check_metric_below,
    CheckKind.PROCESS_RUNNING: LocalVerificationRunner._check_process_running,
}


# -- module-level helpers ------------------------------------------------


def _as_action(check: VerificationCheck) -> "RemediationAction":
    """View a check as an action, purely so the approval gate can be asked.

    A check is not a remediation action and is never executed as one; this
    adapter exists because the gate protocol is defined in terms of
    :class:`~core.safety.RemediationAction`.
    """
    from core.safety import RemediationAction

    return RemediationAction(
        action_id=check.check_id,
        title=check.description,
        description=check.description,
        risk=check.risk,
        command=check.parameters.get("command"),
    )


def _require(check: VerificationCheck, key: str) -> str:
    """Fetch a required parameter, raising a clear error when it is absent."""
    value = check.parameters.get(key, "").strip()
    if not value:
        raise LocalOpsError(
            f"a {check.kind.value} check requires the {key!r} parameter",
            check_id=check.check_id,
        )
    return value


def _require_path(check: VerificationCheck, key: str) -> Path:
    """Fetch a required parameter as a :class:`Path`, expanded for the user."""
    return Path(_require(check, key)).expanduser()


def _compile(pattern: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise LocalOpsError(f"invalid pattern {pattern!r}: {exc}") from exc


def _search(path: Path, regex: re.Pattern[str]) -> int | None:
    """Return the 1-based line number of the first match, or ``None``."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")[:_MAX_READ_BYTES]
    except OSError as exc:
        raise LocalOpsError(f"could not read {path}: {exc}") from exc
    for number, line in enumerate(text.splitlines(), start=1):
        if regex.search(line):
            return number
    return None


def _process_list() -> set[str] | None:
    """Lower-cased names of running processes, or ``None`` when undeterminable."""
    import os

    argv = (
        ["tasklist", "/fo", "csv", "/nh"]
        if os.name == "nt"
        else ["ps", "-eo", "comm="]  # pragma: no cover - the product targets Windows
    )
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, shell=False
            argv,
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - platform dependent
        return None

    names: set[str] = set()
    for line in completed.stdout.splitlines():
        if os.name == "nt":
            if line.startswith('"'):
                names.add(line.split('"', 2)[1].lower())
        else:  # pragma: no cover - the product targets Windows
            names.add(line.strip().lower())
    return names or None

