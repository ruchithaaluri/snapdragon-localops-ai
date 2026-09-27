"""Verification contracts: did the fix actually work?

Verification is the step that makes a diagnosis falsifiable. It re-runs the
failing command, re-reads the log, and re-checks the metric the user cared
about, then reports pass/fail with the observed value.

Everything here is read-only by default. A check that changes state declares
:class:`~core.safety.RiskLevel` above ``READ_ONLY`` and therefore needs the same
explicit approval as a remediation action.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from core.contracts.base import Capability, Component
from core.ids import new_id, utc_now
from core.safety import RiskLevel

__all__ = [
    "CheckKind",
    "CheckOutcome",
    "CheckResult",
    "VerificationCheck",
    "VerificationReport",
    "VerificationRunner",
]


class CheckKind(str, Enum):
    """What a check does. Each maps to a read-only implementation."""

    FILE_EXISTS = "file_exists"
    FILE_CONTAINS = "file_contains"
    LOG_PATTERN_ABSENT = "log_pattern_absent"
    COMMAND_SUCCEEDS = "command_succeeds"
    METRIC_BELOW = "metric_below"
    PROCESS_RUNNING = "process_running"


class CheckOutcome(str, Enum):
    """State of a single check."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class VerificationCheck:
    """A single, declarative check.

    Parameters are a plain mapping so a check can be serialised into a
    benchmark report or a saved run without custom serialisation per kind.
    """

    kind: CheckKind
    description: str
    check_id: str = field(default_factory=lambda: new_id("chk"))
    parameters: dict[str, str] = field(default_factory=dict)
    expectation: str = ""
    risk: RiskLevel = RiskLevel.READ_ONLY
    requires_approval: bool = True
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("a verification check needs a description")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.risk is not RiskLevel.READ_ONLY and not self.requires_approval:
            raise ValueError("a mutating check must still require approval")

    @property
    def is_mutating(self) -> bool:
        return self.risk is not RiskLevel.READ_ONLY


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Outcome of one check, including why it failed."""

    check_id: str
    kind: CheckKind
    outcome: CheckOutcome
    description: str = ""
    expected: str = ""
    observed: str = ""
    detail: str = ""
    duration_ms: float = 0.0
    executed_at: str = field(default_factory=lambda: utc_now().isoformat(timespec="seconds"))

    @property
    def passed(self) -> bool:
        return self.outcome is CheckOutcome.PASSED

    def as_row(self) -> tuple[str, str, str, str]:
        """(check, outcome, expectation, observed) for a table."""
        return (self.description or self.check_id, self.outcome.value, self.expected, self.observed)


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Aggregated result of a verification pass."""

    results: tuple[CheckResult, ...]
    report_id: str = field(default_factory=lambda: new_id("ver"))
    run_id: str = ""
    created_at: str = field(default_factory=lambda: utc_now().isoformat(timespec="seconds"))
    notes: tuple[str, ...] = field(default=())

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    @property
    def passed(self) -> tuple[CheckResult, ...]:
        return tuple(r for r in self.results if r.passed)

    @property
    def failed(self) -> tuple[CheckResult, ...]:
        return tuple(r for r in self.results if r.outcome is CheckOutcome.FAILED)

    @property
    def skipped(self) -> tuple[CheckResult, ...]:
        return tuple(r for r in self.results if r.outcome is CheckOutcome.SKIPPED)

    @property
    def resolved(self) -> bool:
        """True when at least one check passed and none failed."""
        return bool(self.passed) and not self.failed

    def summary(self) -> str:
        """One-line summary, safe for logs and the status bar."""
        return (
            f"{len(self.passed)} passed, {len(self.failed)} failed, "
            f"{len(self.skipped)} skipped"
        )


class VerificationRunner(Component):
    """Executes verification checks and reports the result."""

    kind = "verification"

    @abstractmethod
    def run(self, checks: Sequence[VerificationCheck]) -> VerificationReport:
        """Run the given checks and return a report.

        Implementations must not raise for an individual failing check: a
        failure is data, not an exception. They must raise only when the run
        itself could not be performed.
        """

    def preferred_capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.VERIFICATION_EXECUTION})
