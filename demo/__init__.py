"""The competition demo scenario and its whitelisted remediation.

One deterministic story, runnable repeatedly, offline:

    a sample app with one wrong import
        -> it genuinely fails to start
        -> the real traceback is captured
        -> evidence, retrieval and diagnosis run on it
        -> a single whitelisted fix is proposed
        -> the user approves (or does not)
        -> the fix is applied in-process, with no shell
        -> the same check is run again: BEFORE FAILED, AFTER PASSED

Entry points: :func:`demo.scenario.run_demo` for the UI and the CLI,
:func:`demo.safe_actions.safe_action_names` for the UI to show what this program
is able to change.
"""

from __future__ import annotations

from demo.safe_actions import (
    SAFE_ACTIONS,
    SafeAction,
    apply_safe_action,
    build_demo_action,
    safe_action_names,
)
from demo.scenario import (
    DEMO_TITLE,
    DemoResult,
    DemoRun,
    broken_source,
    fixed_source,
    run_demo,
)

__all__ = [
    "DEMO_TITLE",
    "SAFE_ACTIONS",
    "DemoResult",
    "DemoRun",
    "SafeAction",
    "apply_safe_action",
    "broken_source",
    "build_demo_action",
    "fixed_source",
    "run_demo",
    "safe_action_names",
]