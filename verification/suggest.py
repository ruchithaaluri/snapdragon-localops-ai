"""Deriving verification checks from what a diagnosis actually found.

A check is only worth running if it could fail. These suggestions are built
from the evidence of a real run, so each one re-asks a question the diagnosis
already raised:

* the error that was found is no longer present in the file it came from, and
* the suggested read-only command now succeeds.

Anything that cannot be re-asked honestly - an action with no command, evidence
from text the user pasted rather than a file on disk - produces no check at all.
An unverifiable "check" that always passes is worse than no check.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from core.contracts.diagnostics import Evidence
from core.contracts.verification import CheckKind, VerificationCheck
from core.safety import RiskLevel

from diagnostics.evidence import SIGNATURES

__all__ = ["checks_for_plan"]

#: Refuse to re-read anything larger than this.
_MAX_READ_BYTES = 4 * 1024 * 1024


def checks_for_plan(
    actions: Sequence[object],
    evidence: Sequence[Evidence],
) -> tuple[VerificationCheck, ...]:
    """Build read-only checks that would confirm the evidence is addressed.

    ``actions`` are the proposed actions; only those carrying a command can
    become a command check. ``evidence`` supplies the file and pattern to re-test.
    """
    checks: list[VerificationCheck] = []

    for item in evidence:
        signature = next(
            (s for s in SIGNATURES if s.name == item.metadata.get("signature")), None
        )
        if signature is None:
            continue
        path = Path(item.source) if item.source else None
        if path is None or not path.is_file():
            # Pasted text has no file to re-read, so there is nothing to re-test.
            continue
        if path.stat().st_size > _MAX_READ_BYTES:
            continue
        checks.append(
            VerificationCheck(
                kind=CheckKind.LOG_PATTERN_ABSENT,
                description=(
                    f"{signature.name} no longer appears in {path.name}"
                ),
                parameters={"path": str(path), "pattern": signature.pattern.pattern},
                expectation="the error is gone",
                risk=RiskLevel.READ_ONLY,
                requires_approval=True,
            )
        )

    for action in actions:
        command = getattr(action, "command", None)
        if not command:
            continue
        checks.append(
            VerificationCheck(
                kind=CheckKind.COMMAND_SUCCEEDS,
                description=f"{action.title} now succeeds",
                parameters={"command": command},
                expectation="exit code 0",
                risk=RiskLevel.READ_ONLY,
                requires_approval=True,
            )
        )
    return tuple(checks)
