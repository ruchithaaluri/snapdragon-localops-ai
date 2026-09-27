"""Suggesting what to try next, from a fixed table - never from invention.

The planner maps an error signature to a *check* the user can read, verify, and
run themselves. It deliberately does not try to be clever:

* every command is read-only, and is quoted verbatim from :data:`_TEMPLATES` -
  nothing is assembled from the user's text, so pasted log content can never
  turn into executable shell;
* an action with no safe automatic check carries no command at all and is marked
  informational, so the user reads the description and decides;
* a signature with no entry produces *no action*, rather than a vague one.

The tempting alternative - "install the package that looks missing" - would mean
guessing a package name from a log line, and a wrong guess installs the wrong
thing.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from core.contracts.base import Capability, ProviderHealth, ProviderInfo
from core.contracts.diagnostics import Diagnosis, Evidence
from core.contracts.remediation import RemediationPlanner
from core.ids import new_id
from core.observability import get_logger, log_event
from core.safety import RemediationAction, RemediationPlan, RiskLevel
from core.version import __version__

__all__ = ["SUGGESTION_COUNT", "ActionTemplate", "RuleBasedRemediationPlanner"]

_log = get_logger(__name__)

#: Upper bound on how many actions one plan may contain.
SUGGESTION_COUNT = 4


@dataclass(frozen=True, slots=True)
class ActionTemplate:
    """A ready-made, reviewable suggestion for one error signature.

    ``command`` is a read-only inspection command. ``None`` means there is no
    honest automatic check, and the action is informational only.
    """

    signature: str
    title: str
    description: str
    command: str | None = None
    rationale: str = ""
    risk: RiskLevel = RiskLevel.READ_ONLY

    def build(self, evidence_ids: Sequence[str]) -> RemediationAction:
        """Materialise this template into an approvable action."""
        return RemediationAction(
            action_id=new_id("act"),
            title=self.title,
            description=self.description,
            risk=self.risk,
            command=self.command,
            # Read-only actions are approved too: "run this command" is never
            # assumed to be what the user already wanted.
            requires_approval=True,
            reversible=True,
            supporting_evidence_ids=tuple(evidence_ids),
            rationale=self.rationale,
        )


#: Signature -> suggestion. Every command here is read-only.
_TEMPLATES: tuple[ActionTemplate, ...] = (
    ActionTemplate(
        signature="python_module_not_found",
        title="Check which interpreter is running",
        description=(
            "Confirm the executable and site-packages of the interpreter that "
            "produced the error. A wrong virtual environment is the usual cause."
        ),
        command=(
            "python -c \"import sys, sysconfig; print(sys.executable); "
            "print(sysconfig.get_paths()['purelib'])\""
        ),
        rationale="Locates the interpreter before anything is installed or changed.",
    ),
    ActionTemplate(
        signature="node_module_not_found",
        title="List installed npm packages",
        description=(
            "Show what the project currently resolves, so a missing dependency "
            "is visible before installing anything."
        ),
        command="npm ls --depth=0",
        rationale="Read-only inventory; does not modify node_modules or the lockfile.",
    ),
    ActionTemplate(
        signature="python_type_error",
        title="Surface the raising frame",
        description=(
            "Print the active traceback with developer mode enabled, so the "
            "frame that raises the TypeError is visible."
        ),
        command="python -X dev -c \"import traceback; traceback.print_exc()\"",
        rationale="Reproduces without changing state.",
    ),
    ActionTemplate(
        signature="git_conflict",
        title="Show unmerged paths",
        description="List files still carrying conflict markers.",
        command="git status --short",
        rationale="Read-only; reports which paths need a manual decision.",
    ),
    ActionTemplate(
        signature="git_no_repository",
        title="Confirm the working directory",
        description=(
            "The command ran outside a work tree. Check where the process "
            "actually is before retrying."
        ),
        rationale="No command: the fix depends on which directory was intended.",
    ),
    ActionTemplate(
        signature="dll_load_failure",
        title="Investigate the missing native dependency",
        description=(
            "A DLL could not be loaded. Inspect PATH and the application "
            "directory, and confirm whether the Visual C++ redistributable is "
            "installed."
        ),
        rationale=(
            "No command: the fix depends on which DLL is missing, and guessing "
            "a download would be irresponsible."
        ),
    ),
    ActionTemplate(
        signature="windows_access_denied",
        title="Inspect the file's metadata",
        description=(
            "Access was denied. Check ownership, attributes, and whether another "
            "process is holding the file open."
        ),
        command="python -c \"import os,sys; p=sys.argv[1]; print(os.stat(p))\" %FILE%",
        rationale="Read-only inspection of the file's metadata.",
    ),
    ActionTemplate(
        signature="model_load_failure",
        title="Verify the model artifact",
        description=(
            "A model failed to load. Check that the file exists, that its size "
            "is non-zero, and that its checksum matches the configured value."
        ),
        command=(
            "python -c \"import hashlib,pathlib,sys; p=pathlib.Path(sys.argv[1]); "
            "print(p.exists(), p.stat().st_size if p.exists() else 0)\" %FILE%"
        ),
        rationale="Read-only; distinguishes a missing file from a corrupt one.",
    ),
    ActionTemplate(
        signature="accelerator_unavailable",
        title="Report the available execution providers",
        description=(
            "No accelerator provider is usable. List what the runtime actually "
            "offers so the run can be reported honestly as CPU."
        ),
        rationale=(
            "No command: the provider list must come from the installed "
            "runtime, not from a guess."
        ),
    ),
    ActionTemplate(
        signature="python_recursion_error",
        title="Raise the recursion limit in a throwaway process",
        description=(
            "Deep recursion is usually a missing base case. If a deeper stack is "
            "genuinely needed it can be raised per-process - never in the shell."
        ),
        command=(
            "python -c \"import sys; sys.setrecursionlimit(10000); "
            "print(sys.getrecursionlimit())\""
        ),
        rationale="Affects one process only; the shell and system are untouched.",
    ),
)



class RuleBasedRemediationPlanner(RemediationPlanner):
    """Proposes read-only checks that follow from the evidence in a diagnosis.

    The planner is given the run's evidence so it can map evidence ids back to
    the signatures that produced them. Evidence is passed in explicitly rather
    than re-derived, so a plan always corresponds to the diagnosis the user is
    currently looking at.
    """

    kind = "remediation_planner"

    def __init__(self, evidence: Sequence[Evidence] = ()) -> None:
        self._evidence = {item.evidence_id: item for item in evidence}

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="rules-planner",
            kind=self.kind,
            version=__version__,
            capabilities=frozenset({Capability.REMEDIATION_PLANNING}),
            detail=(
                f"{len(_TEMPLATES)} read-only suggestions; proposes only, never executes"
            ),
        )

    def health(self) -> ProviderHealth:
        return ProviderHealth.ready("proposal-only; this component runs nothing")

    def plan(
        self, diagnosis: object, *, context: dict[str, str] | None = None
    ) -> RemediationPlan:
        """Propose actions for ``diagnosis``.

        ``diagnosis`` is narrowed here rather than in the contract, keeping
        ``core.contracts`` free of a dependency on this package.
        """
        if not isinstance(diagnosis, Diagnosis):
            raise TypeError(f"expected a Diagnosis, got {type(diagnosis).__name__}")

        evidence = self._evidence
        if context and isinstance(context.get("evidence"), Sequence):
            evidence = {item.evidence_id: item for item in context["evidence"]}

        actions = self._actions_for(diagnosis, evidence)
        summary = (
            f"{len(actions)} read-only check(s) suggested from "
            f"{len(diagnosis.evidence_ids)} cited evidence item(s)."
            if actions
            else "No automatic check is available for this evidence. Follow the "
            "citations under each hypothesis."
        )
        log_event(
            _log,
            logging.INFO,
            "remediation.planned",
            actions=len(actions),
            confidence=diagnosis.confidence,
        )
        return RemediationPlan(
            run_id="",
            summary=summary,
            actions=tuple(actions),
            generator=self.info.name,
            confidence=diagnosis.confidence,
        )

    # -- internals --------------------------------------------------------

    def _actions_for(
        self, diagnosis: Diagnosis, evidence: dict[str, Evidence]
    ) -> list[RemediationAction]:
        """Match cited evidence to templates, strongest hypothesis first."""
        by_signature: dict[str, list[str]] = {}
        for evidence_id in diagnosis.evidence_ids:
            item = evidence.get(evidence_id)
            if item is None:
                continue
            signature = item.metadata.get("signature", "")
            if signature:
                by_signature.setdefault(signature, []).append(evidence_id)

        # Rank by the confidence of the hypotheses that cited this evidence, so
        # the strongest finding is suggested first.
        strength: dict[str, float] = {}
        for hypothesis in diagnosis.ranked():
            for evidence_id in hypothesis.evidence_ids:
                item = evidence.get(evidence_id)
                signature = item.metadata.get("signature", "") if item else ""
                if signature:
                    strength[signature] = max(
                        strength.get(signature, 0.0), hypothesis.confidence
                    )

        actions: list[RemediationAction] = []
        used: set[str] = set()
        for signature in sorted(by_signature, key=lambda s: -strength.get(s, 0.0)):
            template = next((t for t in _TEMPLATES if t.signature == signature), None)
            if template is None or template.signature in used:
                continue
            used.add(template.signature)
            actions.append(template.build(by_signature[signature]))
            if len(actions) >= SUGGESTION_COUNT:
                break
        return actions

