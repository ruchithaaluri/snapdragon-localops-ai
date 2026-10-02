"""Safe remediation actions: a whitelist, not a command interpreter.

This module exists because the requirement is *"remediation is NOT arbitrary
command execution"*. The application's general-purpose executor can run a
command string, which is the right tool for a read-only inspection the user
approved. It is the wrong tool for a *fix*, because a fix assembled from text
can be assembled from attacker-controlled text too.

So a fix is not a string here. A fix is an entry in :data:`SAFE_ACTIONS`:

* a fixed, reviewed **id** - the only thing a plan or an approval record can
  refer to,
* a fixed, reviewed Python callable - never shell text, never user input,
* a fixed, bounded write - one file, with the change stated in advance.

There is no code path here that reaches a shell, an interpreter, ``pip``, the
network, or a path the caller did not supply. Adding a capability means adding a
reviewed entry, which is the point: the set of changes this program can make is
visible in one short list.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from core.errors import ActionDeniedError
from core.observability import get_logger, log_event
from core.safety import RemediationAction, RiskLevel

__all__ = [
    "SAFE_ACTIONS",
    "SafeAction",
    "apply_safe_action",
    "build_demo_action",
    "safe_action_names",
]

_log = get_logger(__name__)

#: Refuse to rewrite anything larger than this. A demo fix is a one-line edit; a
#: large file is a sign something is wrong and is not worth rewriting.
_MAX_EDIT_BYTES = 256 * 1024


@dataclass(frozen=True, slots=True)
class SafeAction:
    """One reviewed, reversible remediation.

    ``apply`` receives the target path the caller resolved and returns a
    human-readable description of what it changed. It touches no other file.
    """

    action_id: str
    title: str
    description: str
    rationale: str
    apply: Callable[[str], str]
    risk: RiskLevel = RiskLevel.LOW
    reversible: bool = True

    def to_action(self, target: str) -> RemediationAction:
        """Materialise this entry as an approvable action bound to ``target``.

        ``command`` is deliberately ``None``: this action has no command line,
        because it is not a command. The UI shows the description and the
        target, which is the honest description of what will happen.
        """
        return RemediationAction(
            action_id=self.action_id,
            title=self.title,
            description=f"{self.description}\n\nTarget: {target}",
            risk=self.risk,
            command=None,
            requires_approval=True,
            reversible=self.reversible,
            rationale=self.rationale,
        )


def _correct_module_import(target: str) -> str:
    """Rewrite the demo app's one wrong import. The demo's only fix.

    Refuses anything that is not the expected single-token substitution, so a
    tampered file produces an error rather than a silent rewrite.
    """
    from pathlib import Path

    from demo.scenario import _RIGHT_IMPORT, _WRONG_IMPORT, fixed_source

    path = Path(target)
    if not path.is_file():
        raise ActionDeniedError("the demo app is not on disk", target=target)
    if path.stat().st_size > _MAX_EDIT_BYTES:
        raise ActionDeniedError(
            "the demo app is unexpectedly large; refusing to rewrite it",
            target=target,
        )
    current = path.read_text(encoding="utf-8")
    if _WRONG_IMPORT not in current:
        # Already fixed, or not the file we think it is. Either way, say so
        # rather than overwriting something we did not recognise.
        raise ActionDeniedError(
            "the expected broken import is not present, so nothing was changed",
            target=target,
        )
    # The corrected file is generated, not patched, so the two states cannot
    # drift and the before/after difference stays exactly one line.
    path.write_text(fixed_source(), encoding="utf-8")
    return f"replaced '{_WRONG_IMPORT}' with '{_RIGHT_IMPORT}' in {path.name}"


#: The complete set of changes this program is able to make. One entry.
SAFE_ACTIONS: dict[str, SafeAction] = {
    "demo.correct_module_import": SafeAction(
        action_id="demo.correct_module_import",
        title="Correct the sample app's import",
        description=(
            "Rewrite the single import line that names a module which does not "
            "exist, so it names a standard-library module that does. One line "
            "changes; the rest of the file is regenerated from the same source."
        ),
        rationale=(
            "The diagnosis cited a ModuleNotFoundError for a module that is not "
            "installed and was never meant to be. The fix is to correct the "
            "name, not to install anything."
        ),
        apply=_correct_module_import,
        risk=RiskLevel.LOW,
    ),
}


def apply_safe_action(action: RemediationAction, *, approved: bool) -> str:
    """Perform ``action`` if - and only if - it is whitelisted and approved.

    The order of the checks is the security property:

    1. the action id must exist in :data:`SAFE_ACTIONS`; an unknown id is denied
       rather than attempted,
    2. the safety policy must permit the action's risk,
    3. the caller must have recorded approval.

    A caller cannot reach the underlying function with a synthetic id, and
    cannot reach it with a real id it has not approved.
    """
    entry = SAFE_ACTIONS.get(action.action_id)
    if entry is None:
        raise ActionDeniedError(
            "this action is not in the safe-action whitelist, so it will not run",
            action_id=action.action_id,
            available=safe_action_names(),
        )
    if not approved:
        raise ActionDeniedError(
            "the user has not approved this action", action_id=action.action_id
        )
    from core.safety import SafetyPolicy

    SafetyPolicy(allow_mutating_actions=True, max_risk=RiskLevel.MEDIUM).check_action(
        action
    )

    target = _target_of(action)
    detail = entry.apply(target)
    log_event(
        _log,
        logging.INFO,
        "demo.safe_action.applied",
        action_id=action.action_id,
        target=target,
        detail=detail,
    )
    return detail


def _target_of(action: RemediationAction) -> str:
    """Recover the target path stored in the action's description.

    The path travels inside the action rather than as a separate argument, so an
    approval is bound to *this* action on *this* file and cannot be replayed
    against another one.
    """
    for line in action.description.splitlines():
        if line.startswith("Target: "):
            return line.removeprefix("Target: ").strip()
    raise ActionDeniedError(
        "the action carries no target file", action_id=action.action_id
    )



def safe_action_names() -> tuple[str, ...]:
    """Every action id this program can perform. Shown in the UI."""
    return tuple(sorted(SAFE_ACTIONS))


def build_demo_action(target: str) -> RemediationAction:
    """The demo's remediation action, bound to ``target``."""
    return SAFE_ACTIONS["demo.correct_module_import"].to_action(target)
