"""Remediation: turning a diagnosis into proposed, gated actions.

Split in two, on purpose:

``planner``
    Decides *what* might help. It only ever proposes; it never runs anything.
``executor``
    Decides whether an action may *run*, checks the safety policy first, then
    the per-action approval, and only then executes.

Neither module invents a command. Every proposed action either carries a
read-only command taken from a fixed table in the planner, or carries no command
at all and is informational. A fix the application cannot justify is worse than
no fix, because the user cannot check it.
"""

from __future__ import annotations

from diagnostics.remediation.executor import LocalRemediationExecutor
from diagnostics.remediation.planner import (
    SUGGESTION_COUNT,
    ActionTemplate,
    RuleBasedRemediationPlanner,
)

__all__ = [
    "SUGGESTION_COUNT",
    "ActionTemplate",
    "LocalRemediationExecutor",
    "RuleBasedRemediationPlanner",
]
