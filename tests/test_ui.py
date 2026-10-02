"""The Qt shell: it must render results and never compute them.

Marked ``ui`` because every test here constructs widgets and therefore needs
PySide6. They run headless via the ``offscreen`` platform, so no display is
required.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="the UI tests need PySide6")
pytestmark = pytest.mark.ui

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.ui.main_window import MainWindow  # noqa: E402
from app.ui.theme import Theme, stylesheet_for  # noqa: E402
from app.ui.widgets import StatusPill  # noqa: E402
from app.pipeline import build_retriever  # noqa: E402
from core.contracts.verification import CheckKind, VerificationCheck  # noqa: E402
from core.safety import RemediationAction  # noqa: E402
from diagnostics import build_query  # noqa: E402
from verification import checks_for_plan  # noqa: E402

from .conftest import SAMPLE_LOG  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qapp, settings):
    win = MainWindow(settings)
    yield win
    win.close()


# -- theme ---------------------------------------------------------------


def test_both_themes_resolve_and_produce_a_stylesheet():
    for name in ("dark", "light"):
        theme = Theme(name)  # type: ignore[arg-type]
        assert theme.status("ok").startswith("#")
        assert "background" in stylesheet_for(theme)


def test_an_unknown_theme_is_rejected():
    with pytest.raises(ValueError):
        Theme("neon")


def test_a_status_pill_always_shows_its_word(qapp):
    pill = StatusPill(Theme("dark"))
    pill.set_status("approved", "ok")
    assert pill.text() == "approved", "status must never be colour alone"
    assert "approved" in pill.accessibleName().lower()


# -- window --------------------------------------------------------------


def test_the_window_builds_with_every_tab(window):
    labels = [window._tabs.tabText(i) for i in range(window._tabs.count())]
    assert labels == [
        "Diagnosis",
        "Evidence",
        "Suggested fixes",
        "Verification",
        "Demo",
        "Benchmark",
        "This machine",
    ]


def test_the_status_strip_states_the_mode_in_words(window):
    """The four indicators are words, and the mock one is honest about itself."""
    words = [
        window._status_strip.itemAt(i).widget().text()
        for i in range(window._status_strip.count())
        if window._status_strip.itemAt(i).widget() is not None
        and hasattr(window._status_strip.itemAt(i).widget(), "text")
    ]
    assert "LOCAL MODE" in words
    assert "MOCK AI" in words, "the default backend is the mock, and says so"
    assert not any(w == "REAL AI" for w in words)
    assert not any(w == "NPU" for w in words), "no NPU claim before a verified run"


def test_the_demo_and_benchmark_tabs_start_empty(window):
    """Nothing is claimed before a run happens."""
    assert "not run" in window.demo._outcome.text().lower()
    assert "No benchmark has run" in window.benchmark._summary.text()


def test_a_fresh_window_has_run_nothing(window):
    assert "Nothing has been run yet" in window.statusBar().currentMessage()
    assert window.remediation._approved_ids() == []


def test_a_run_renders_into_the_panels(window):
    run = window._pipeline.run(build_query("why?", SAMPLE_LOG, source="app.log"))
    window._on_finished(run)

    assert window.evidence._table.rowCount() == len(run.query.evidence)
    assert window.diagnosis._summary.text() == run.diagnosis.summary
    assert window.remediation._cards, "the proposed actions must be shown"
    assert all(
        card.findChild(StatusPill).text() == "pending"
        for card in window.remediation._cards.values()
    ), "every action starts unapproved"
    assert "capture: completed" in window._stage_bar.text()


def test_the_diagnosis_view_shows_its_provenance(window):
    run = window._pipeline.run(build_query("why?", SAMPLE_LOG, source="app.log"))
    window._on_finished(run)
    html = window.diagnosis._hypotheses.toPlainText()
    assert "rules_only" in html
    assert "cpu" in html.lower()
    assert window.diagnosis._mode_pill.text() == "rules only (no model)"


def test_an_unresolved_run_shows_a_refusal_not_an_answer(window):
    run = window._pipeline.run(build_query("why?", "nothing interesting here"))
    window._on_finished(run)
    assert run.diagnosis is None
    assert "did not support" in window.diagnosis._summary.text()
    assert window.diagnosis._mode_pill.text() == "no result"


def test_approving_records_a_decision_and_runs_nothing(window):
    run = window._pipeline.run(build_query("why?", SAMPLE_LOG, source="app.log"))
    window._on_finished(run)
    action = run.plan.actions[0]

    window._on_approved(action.action_id)
    assert window._pipeline.gate.status(action).value == "approved"
    assert "Nothing has run yet" in window.statusBar().currentMessage()


def test_denying_is_recorded_too(window):
    run = window._pipeline.run(build_query("why?", SAMPLE_LOG, source="app.log"))
    window._on_finished(run)
    action = run.plan.actions[0]
    window._on_denied(action.action_id)
    assert window._pipeline.gate.status(action).value == "denied"


def test_applying_runs_only_the_approved_action(window):
    run = window._pipeline.run(build_query("why?", SAMPLE_LOG, source="app.log"))
    window._on_finished(run)
    first = run.plan.actions[0]
    window._on_approved(first.action_id)
    window._on_apply()
    assert "1 of" in window.statusBar().currentMessage()


def test_verification_is_offered_then_run(window):
    from app.ui.main_window import _check_as_action

    run = window._pipeline.run(build_query("why?", SAMPLE_LOG, source="app.log"))
    window._on_finished(run)
    offered = checks_for_plan([run.plan.actions[0]], run.query.evidence)
    assert offered

    window.verification.show_checks(offered)
    assert window.verification._table.rowCount() == len(offered)
    assert window.verification._run.isEnabled(), "offered, but nothing has run yet"

    for check in offered:
        window._pipeline.gate.approve(_check_as_action(check))
    window._on_run_checks()
    assert window.verification._table.rowCount() == len(offered)
    assert "passed" in window.verification._summary.text()


def test_no_derived_checks_disables_the_run_button(window):
    window.verification.show_checks([])
    assert not window.verification._run.isEnabled()
    assert "No check could be derived" in window.verification._summary.text()


def test_a_verification_report_renders(window, log_file):
    check = VerificationCheck(
        CheckKind.FILE_EXISTS, "the log exists", parameters={"path": str(log_file)}
    )
    # Approve on the window's *own* gate. ``build_pipeline`` constructs a fresh
    # gate per window, so approving on any other instance would leave this check
    # unapproved - and the runner would correctly report it as skipped.
    window._pipeline.gate.approve(
        RemediationAction(
            action_id=check.check_id, title=check.description, description="d"
        )
    )
    report = window._pipeline.verify(None, [check])
    window.verification.show_report(report)
    assert window.verification._table.rowCount() == 1
    assert window.verification._resolved.text() == "resolved"


def test_the_environment_panel_reports_retrieval_and_history(window, settings):
    """Both layers are reported, and neither is claimed when absent."""
    window.environment.refresh(settings, window._pipeline)
    values = dict(
        (window.environment._table.item(i, 0).text(),
         window.environment._table.item(i, 1).text())
        for i in range(window.environment._table.rowCount())
    )

    assert "Retrieval" in values
    assert "Run history" in values
    # A fresh install indexes nothing, and must say so rather than claim a corpus.
    assert values["Retrieval"] == "available, empty"


def test_the_environment_panel_never_invents_a_chunk_count(window, settings):
    """Indexing a document must be visible in the panel, not hidden."""
    retriever = build_retriever(settings)
    retriever.index_documents([("notes.md", "requests is missing from the venv.")])
    window._pipeline._retriever = retriever

    window.environment.refresh(settings, window._pipeline)
    values = [
        (
            window.environment._table.item(i, 0).text(),
            window.environment._table.item(i, 1).text(),
        )
        for i in range(window.environment._table.rowCount())
    ]
    assert ("Retrieval", "1 chunk(s) indexed") in values


def test_the_environment_panel_admits_when_it_did_not_look(window, settings):
    """No pipeline means 'not inspected' - never 'not available'."""
    window.environment.refresh(settings)
    values = [
        window.environment._table.item(i, 1).text()
        for i in range(window.environment._table.rowCount())
    ]
    assert "not inspected" in values
    assert "not available" not in values


def test_the_environment_panel_never_claims_an_npu(window, settings):
    window.environment.refresh(settings)
    rows = window.environment._table.rowCount()
    values = [
        (
            window.environment._table.item(i, 0).text(),
            window.environment._table.item(i, 1).text(),
        )
        for i in range(rows)
    ]
    assert ("Execution target", "cpu (verified)") in values
    assert not any(
        "npu" in value.lower() for _, value in values
    ), "a machine with no verified NPU run must never report one"
