"""The competition demo: one deterministic scenario, start to finish.

The story the demo tells, and the one it actually runs:

1. A tiny sample app ships with a **real, deliberate defect**: it imports a
   module that does not exist, so it cannot start.
2. Running it produces a genuine ``ModuleNotFoundError`` - not a fabricated
   one. The error is captured to a log file.
3. That log is fed into the normal pipeline. Evidence extraction, retrieval,
   rule-based diagnosis and the remediation plan all run for real.
4. The plan is shown. **Nothing happens until the user approves.**
5. The one approved action is a whitelisted, in-process, reversible edit to the
   sample app's source. No shell, no ``pip install``, no network.
6. The app is run again. The same check that reported ``FAILED`` now reports
   ``PASSED``.

Every step is offline and repeatable. Running the demo twice in a row resets the
sample app to its broken state first, so the demo cannot pass on the second run
only because the first run left the file fixed.

What this is not: a real fix produced by a model, or a measurement of one. The
diagnosis is rule-based and the remediation is a predefined action. Both facts
are displayed. The point of the demo is the *workflow* and the *safety gate*,
and it is honest about which parts are deterministic and which would be learned.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from app.pipeline import build_pipeline, build_retriever
from core.config.paths import ensure_dir, runtime_dir
from core.contracts.pipeline import PipelineRun
from core.contracts.verification import (
    CheckKind,
    VerificationCheck,
    VerificationReport,
)
from core.observability import get_logger, log_event
from core.safety import ApprovalDecision, RemediationAction, RiskLevel
from diagnostics import build_query

from demo.safe_actions import apply_safe_action, build_demo_action

__all__ = [
    "DEMO_TITLE",
    "DemoResult",
    "DemoRun",
    "broken_source",
    "fixed_source",
    "run_demo",
]

_log = get_logger(__name__)

DEMO_TITLE = "Demo: a sample app that cannot start because of one wrong import"

#: The defect. ``localops_sample`` is a module that has never existed, so the
#: app genuinely fails. The fix is a one-token change to a name that does exist.
_WRONG_IMPORT = "import localops_sample"
_RIGHT_IMPORT = "import json"

_PROGRAM = '''\
"""A sample app with one deliberate defect, used by the LocalOps demo."""

{import_line}


def build_report():
    """Return a small report. Never reached while the import is wrong."""
    return {{"status": "ok", "rows": 3, "serialized": json.dumps({{"rows": 3}})}}


if __name__ == "__main__":
    print(build_report())
'''

#: The question the demo asks, and the evidence the user would paste.
_QUESTION = (
    "Why does the sample app fail to start, and what is the safest fix?"
)

#: A tiny, hand-written knowledge base, so the retrieval stage does real work in
#: the demo instead of reporting "skipped: index is empty".
#:
#: It is deliberately four short passages, not a corpus. Retrieval is
#: demonstrated by retrieving the *right* one out of a handful, which is
#: enough to show the stage functioning and costs nothing to ship or to index.
KNOWLEDGE_BASE: tuple[tuple[str, str], ...] = (
    (
        "module-not-found.md",
        "ModuleNotFoundError means the interpreter could not find the named "
        "module. The usual cause is that the package is not installed in the "
        "environment the process is actually running under, rather than the one "
        "you expected. Check which interpreter is active before installing "
        "anything.",
    ),
    (
        "wrong-module-name.md",
        "A ModuleNotFoundError for a name that was never intended to be a "
        "package is almost always a typo in an import statement, not a missing "
        "install. Compare the name in the traceback with the import line that "
        "produced it, and correct the name instead of installing anything.",
    ),
    (
        "traceback-reading.md",
        "Read a Python traceback from the bottom up. The last line names the "
        "exception; the lines above it are the call chain that led there. The "
        "file and line number in each frame identify the source of the problem.",
    ),
    (
        "virtual-environments.md",
        "A virtual environment isolates installed packages per project. Running "
        "a script with the system interpreter instead of the project "
        "environment produces missing-module errors that look like broken "
        "dependencies. Print sys.executable to see which interpreter is active.",
    ),
)



def broken_source() -> str:
    """The sample app exactly as it ships: it cannot import, so it cannot run."""
    return _PROGRAM.format(import_line=_WRONG_IMPORT)




def fixed_source() -> str:
    """The sample app after the one approved edit."""
    return _PROGRAM.format(import_line=_RIGHT_IMPORT)


@dataclass(frozen=True, slots=True)
class DemoResult:
    """Everything the UI and the CLI need to show the demo.

    ``before_passed`` and ``after_passed`` are the two words the demo exists to
    display. They are booleans derived from a subprocess exit code, not labels
    chosen by the presenter.
    """

    title: str
    workspace: Path
    app_path: Path
    log_path: Path
    before: "DemoRun"
    run: PipelineRun
    action: RemediationAction
    approved: bool
    applied: bool
    after: "DemoRun"
    report: VerificationReport
    notes: tuple[str, ...] = field(default=())

    @property
    def before_passed(self) -> bool:
        return self.before.passed

    @property
    def after_passed(self) -> bool:
        return self.after.passed

    @property
    def worked(self) -> bool:
        """True when the demo demonstrated what it claims to demonstrate."""
        return not self.before.passed and self.after.passed

    def as_rows(self) -> tuple[tuple[str, str], ...]:
        """Flat (label, value) rows for the demo panel."""
        return (
            ("Scenario", self.title),
            ("BEFORE", "PASSED" if self.before_passed else "FAILED"),
            ("AFTER", "PASSED" if self.after_passed else "FAILED"),
            ("Sample app", str(self.app_path)),
            ("Error log", str(self.log_path)),
            (
                "Diagnosis",
                self.run.diagnosis.summary if self.run.diagnosis else "no diagnosis",
            ),
            (
                "Reasoning",
                self.run.diagnosis.reasoning_mode.value
                if self.run.diagnosis
                else "n/a",
            ),
            (
                "Evidence",
                f"{len(self.run.query.evidence)} item(s) extracted, each cited by line",
            ),
            (
                "Retrieved context",
                "attached"
                if self.run.query.retrieved_context
                else "none (index empty - the app says so)",
            ),
            ("Fix", f"{self.action.title} ({self.action.risk.value})"),
            (
                "User approval",
                "APPROVED" if self.approved else "NOT GIVEN - nothing was run",
            ),
            ("Fix applied", "yes" if self.applied else "no"),
            ("Verification", self.report.summary()),
        )


@dataclass(frozen=True, slots=True)
class DemoRun:
    """One execution of the sample app, with its real exit code."""

    command: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False

    @property
    def passed(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def output(self) -> str:
        """Combined output, bounded. Enough to read a traceback, not a log dump."""
        text = (self.stderr or self.stdout).strip()
        return text[-2000:]


def _workspace() -> Path:
    """Where the demo keeps its files. Small, fixed, and never a user directory.

    It lives under the application data directory, not in the project tree, so
    the demo cannot leave broken source behind in a checkout and cannot touch
    anything the user did not ask it to touch.
    """
    return ensure_dir(runtime_dir() / "demo")


def _reset_app(app_path: Path) -> None:
    """Put the sample app back into its broken state.

    Without this the demo would pass on the second run purely because the first
    run fixed the file, which would make the "BEFORE: FAILED" half a lie.
    """
    app_path.write_text(broken_source(), encoding="utf-8")


def _run_app(app_path: Path, timeout: float = 20.0) -> DemoRun:
    """Execute the sample app and capture its real result.

    A short timeout and a bounded capture: a demo must not be able to hang the
    window or fill memory, whatever the sample app does.
    """
    from core.timing import Stopwatch

    argv = (sys.executable, str(app_path))
    watch = Stopwatch().start()
    timed_out = False
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, shell=False
            argv,
            cwd=str(app_path.parent),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            shell=False,
        )
        code, out, err = completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired:
        timed_out, code, out, err = True, -1, "", f"timed out after {timeout:.0f}s"
    except OSError as exc:  # pragma: no cover - platform dependent
        timed_out, code, out, err = True, -1, "", str(exc)
    return DemoRun(
        command=argv,
        exit_code=code,
        stdout=out[:4000],
        stderr=err[:4000],
        duration_ms=watch.stop().milliseconds,
        timed_out=timed_out,
    )


def _build_checks(app_path: Path) -> tuple[VerificationCheck, ...]:
    """The checks that define BEFORE and AFTER.

    Both halves of the demo are the *same* check run at two different times:
    does the sample app start? There is no separate "success" criterion that
    could drift from the failure criterion, which is the only way a
    before/after display means anything.
    """
    return (
        VerificationCheck(
            kind=CheckKind.COMMAND_SUCCEEDS,
            description=f"{app_path.name} starts and exits 0",
            check_id="demo.app_starts",
            parameters={"command": f'"{sys.executable}" "{app_path}"'},
            expectation="exit code 0",
            risk=RiskLevel.READ_ONLY,
            requires_approval=True,
            timeout_seconds=20.0,
        ),
    )


def _approval(action: RemediationAction) -> ApprovalDecision:
    """The recorded human decision for the demo's single action."""
    return ApprovalDecision.APPROVED


def _check_as_action(check: VerificationCheck) -> RemediationAction:
    """View a check as an action so the approval gate can be asked about it.

    A check is never executed as a remediation; this mirrors the adapter the
    desktop shell uses, so the demo and the UI agree on how checks are approved.
    """
    return RemediationAction(
        action_id=check.check_id,
        title=check.description,
        description=check.description,
        risk=check.risk,
        command=check.parameters.get("command"),
    )


def run_demo(
    settings,  # noqa: ANN001 - AppSettings
    *,
    approve: bool = True,
    open_history: bool = True,
) -> DemoResult:
    """Run the whole demo end to end and return everything it observed.

    Parameters
    ----------
    settings:
        Validated application configuration, used to build the real pipeline so
        the demo exercises the same code the user does.
    approve:
        When ``False`` the run stops at the gate: the plan is produced, nothing
        is applied, and the "AFTER" state is a second run of the still-broken
        app. This is what the "nothing happens without approval" claim means,
        and it is demonstrated rather than asserted.
    """
    workspace = _workspace()
    app_path = workspace / "sample_app.py"
    log_path = workspace / "sample_app_error.log"

    # 1. Reset, so the demo is repeatable and BEFORE is honestly a failure.
    _reset_app(app_path)
    before = _run_app(app_path)

    # 2. Capture the real failure to a log, as a user would paste or save it.
    log_path.write_text(before.output() or "(no output captured)", encoding="utf-8")
    log_event(
        _log,
        logging.INFO,
        "demo.before",
        exit_code=before.exit_code,
        duration_ms=round(before.duration_ms, 1),
    )

    # 3. The normal pipeline, on the real captured evidence. The retriever is
    #    seeded with a four-passage knowledge base so the retrieval stage does
    #    real work; nothing is read from disk and nothing is downloaded.
    retriever = build_retriever(settings)
    if retriever is not None:
        retriever.index_documents(list(KNOWLEDGE_BASE))
    pipeline = build_pipeline(
        settings, retriever=retriever, open_history=open_history
    )
    query = build_query(
        _QUESTION,
        log_path.read_text(encoding="utf-8"),
        source=log_path.name,
    )
    run = pipeline.run(query)

    # 4. The gate. The plan is a single whitelisted action, shown before it runs.
    action = build_demo_action(str(app_path))
    approved = False
    applied = False
    applied_detail = ""
    if approve:
        pipeline.gate.decide(
            action, _approval(action), note="approved by the presenter in the demo"
        )
        record = pipeline.gate.get(action.action_id)
        approved = record is not None and record.decision.is_approved
        if approved:
            applied_detail = apply_safe_action(action, approved=True)
            applied = True

    # 5. Verification re-asks the identical question, after the change. The
    #    checks are approved on the same gate, one decision per check, so a
    #    headless demo run is auditable rather than silently permissive.
    checks = _build_checks(app_path)
    for check in checks:
        pipeline.gate.approve(_check_as_action(check), note="approved for the demo")
    report = pipeline.verify(run, checks)

    # 6. AFTER is measured the same way BEFORE was: by running the app.
    after = _run_app(app_path)

    result = DemoResult(
        title=DEMO_TITLE,
        workspace=workspace,
        app_path=app_path,
        log_path=log_path,
        before=before,
        run=run,
        action=action,
        approved=approved,
        applied=applied,
        after=after,
        report=report,
        notes=(applied_detail,) if applied_detail else (),
    )
    log_event(
        _log,
        logging.INFO,
        "demo.completed",
        before_passed=result.before_passed,
        after_passed=result.after_passed,
        applied=applied,
    )
    return result


