"""Verification: did the suggested fix actually work?

This is the step that makes a diagnosis falsifiable. Each
:class:`~core.contracts.verification.VerificationCheck` is a small, declarative
assertion - "this file now exists", "this error no longer appears in the log",
"this command now exits 0" - and the runner reports pass or fail with the value
it actually observed.

Two design decisions are worth stating outright:

* **A failing check is data, not an exception.** The contract is explicit that
  :meth:`LocalVerificationRunner.run` returns a report containing failures; it
  only raises when the run itself could not be performed. A tool that crashed on
  the first failed check could not answer the question it was asked.
* **Approval is required, and defaults to denial.** The runner asks an
  :class:`~core.safety.ApprovalGate` about every check that declares
  ``requires_approval``. The default gate is
  :class:`~core.safety.DenyAllApprovalGate`, so a headless run verifies nothing
  until someone has explicitly agreed to the specific checks shown on screen.
"""

from __future__ import annotations

from verification.runner import LocalVerificationRunner
from verification.suggest import checks_for_plan

__all__ = ["LocalVerificationRunner", "checks_for_plan"]
