"""Wiring the layers that exist: run history, retrieval, and their honesty.

These cover the integration points rather than the components themselves
(``core/storage`` and ``retrieval`` are exercised through the pipeline, which is
how the application actually uses them).

Two rules dominate the file:

* **Persistence never destroys a result.** A failing store must cost the user
  their history and nothing else.
* **Retrieval is context, never evidence.** No matter how confident the search,
  a retrieved passage can never become a cited fact.
"""

from __future__ import annotations

import threading

import pytest
from pydantic import ValidationError

from app.pipeline import build_pipeline, build_retriever, build_store
from core.config.schema import AppSettings
from core.contracts.diagnostics import DiagnosticQuery
from core.contracts.pipeline import PipelineStage, StageStatus
from core.errors import StorageError

from .conftest import SAMPLE_LOG


def _query(text: str = SAMPLE_LOG, question: str = "why?") -> DiagnosticQuery:
    from diagnostics import build_query

    return build_query(question, text, source="app.log")


# -- run history ---------------------------------------------------------


def test_a_run_is_recorded_in_the_local_database(settings):
    pipeline = build_pipeline(settings)
    assert pipeline.store is not None, "history is on by default"

    run = pipeline.run(_query())

    stored = {record.run_id: record for record in pipeline.store.list_runs()}
    assert run.run_id in stored, "a finished run must be in the history"
    assert stored[run.run_id].outcome == run.outcome.value
    assert stored[run.run_id].question == run.query.question


def test_the_stored_payload_is_json_safe_and_keeps_the_reasoning_mode(settings):
    pipeline = build_pipeline(settings)
    pipeline.run(_query())

    payload = pipeline.store.list_runs()[0].payload
    assert payload["reasoning_mode"] == "rules_only"
    assert payload["evidence_count"] >= 1
    assert [h["status"] for h in payload["hypotheses"]]
    assert all("action_id" in action for action in payload["actions"])


def test_evidence_is_stored_alongside_the_run(settings):
    pipeline = build_pipeline(settings)
    run = pipeline.run(_query())

    assert run.query.evidence
    rows = pipeline.store.fetchall(
        "SELECT evidence_id FROM evidence WHERE run_id = ?", (run.run_id,)
    )
    assert {row["evidence_id"] for row in rows} == {
        item.evidence_id for item in run.query.evidence
    }


def test_an_unresolved_run_is_still_recorded(settings):
    """'I do not know' is an outcome worth keeping, not a failure to swallow."""
    pipeline = build_pipeline(settings)
    run = pipeline.run(_query("nothing interesting here"))

    assert run.diagnosis is None
    assert [r.run_id for r in pipeline.store.list_runs()] == [run.run_id]


def test_history_can_be_switched_off(settings):
    pipeline = build_pipeline(settings, open_history=False)
    assert pipeline.store is None

    run = pipeline.run(_query())
    assert run.diagnosis is not None, "the run still works with no history at all"
    assert not settings.database_path.exists(), "nothing may be written"


def test_storage_can_be_disabled_in_configuration():
    from core.config import load_settings

    configured = load_settings(
        use_environment=False, overrides={"storage": {"enabled": False}}
    )
    assert build_store(configured) is None


def test_a_failing_store_costs_history_not_the_answer(settings):
    """The diagnosis is the product; the database is only a record of it."""

    class BrokenStore:
        def save_run(self, **_kwargs):
            raise StorageError("the disk is full", detail="simulated")

    pipeline = build_pipeline(settings, store=BrokenStore())
    run = pipeline.run(_query())

    assert run.diagnosis is not None, "a broken store must not lose the diagnosis"
    assert run.diagnosis.hypotheses
    assert run.outcome.value == "awaiting_approval"


def test_an_unusable_database_degrades_instead_of_crashing(settings):
    """A directory sitting where the database file belongs makes the open fail."""
    blocked = settings.data_dir / "runtime" / "localops.db"
    blocked.mkdir(parents=True, exist_ok=True)

    assert build_store(settings) is None
    pipeline = build_pipeline(settings)
    assert pipeline.store is None
    assert pipeline.run(_query()).diagnosis is not None


def test_a_run_survives_a_worker_thread(settings):
    """The GUI runs the pipeline off the UI thread; history must follow it."""
    pipeline = build_pipeline(settings)
    results: list[object] = []

    worker = threading.Thread(target=lambda: results.append(pipeline.run(_query())))
    worker.start()
    worker.join(timeout=30)

    assert not worker.is_alive(), "the worker thread must finish"
    assert len(results) == 1
    assert results[0].run_id in {r.run_id for r in pipeline.store.list_runs()}


def test_the_database_name_may_not_escape_the_data_directory():
    """A configured name is a file name, never a path."""
    with pytest.raises(ValidationError):
        AppSettings.model_validate({"storage": {"database_name": "../../escape.db"}})


# -- retrieval -----------------------------------------------------------


class _NoEmbeddings:
    """An AI backend that offers no embeddings, so no retriever can be built."""

    class info:  # noqa: N801 - mirrors the ProviderInfo attribute
        name = "stub-backend"

    embeddings = None


def test_retrieval_is_built_but_reports_an_empty_index_honestly(settings):
    """Nothing is indexed automatically, and the stage says exactly why."""
    pipeline = build_pipeline(settings)
    assert pipeline.retriever is not None, "a retriever exists in Phase 1"
    assert pipeline.retriever.index().count() == 0

    stage = pipeline.run(_query()).stage(PipelineStage.RETRIEVAL)
    assert stage.status is StageStatus.SKIPPED
    assert "no chunks" in stage.detail, "the reason must name the real cause"


def test_a_missing_retriever_is_reported_differently_from_an_empty_one(settings):
    """Two different causes get two different messages; neither is fudged."""
    pipeline = build_pipeline(
        settings, ai_backend=_NoEmbeddings(), open_history=False
    )
    assert pipeline.retriever is None

    stage = pipeline.run(_query()).stage(PipelineStage.RETRIEVAL)
    assert stage.status is StageStatus.SKIPPED
    assert stage.detail == "no retriever configured"


def test_retrieved_passages_become_context_and_never_evidence(settings):
    retriever = build_retriever(settings)
    retriever.index_documents(
        [
            ("notes.md", "requests is installed system-wide but absent from the venv."),
            ("unrelated.md", "Deployment notes about continuous integration pipelines."),
        ]
    )

    run = build_pipeline(settings, retriever=retriever, open_history=False).run(_query())

    assert run.stage(PipelineStage.RETRIEVAL).status is StageStatus.COMPLETED
    assert run.query.retrieved_context, "context should be attached"

    assert "retrieval" not in {i.kind.value for i in run.query.evidence}, (
        "a retrieved passage is never evidence"
    )
    assert set(run.diagnosis.evidence_ids) <= set(run.query.evidence_by_id()), (
        "every citation must still trace back to supplied evidence"
    )


def test_the_index_survives_a_restart(settings):
    retriever = build_retriever(settings)
    retriever.index_documents([("notes.md", "the venv lacks the requests package.")])
    retriever.index().save(str(settings.index_path))

    assert build_retriever(settings).index().count() == 1, "the index must reload"


def test_a_corrupt_index_does_not_stop_start_up(settings):
    settings.index_path.parent.mkdir(parents=True, exist_ok=True)
    settings.index_path.write_text("{ not json", encoding="utf-8")

    assert build_retriever(settings) is None
    assert build_pipeline(settings).run(_query()).diagnosis is not None


def test_the_configured_top_k_bounds_the_context(settings):
    retriever = build_retriever(settings)
    for number in range(6):
        retriever.index_documents(
            [(f"doc{number}.md", f"requests module note number {number}.")]
        )

    pipeline = build_pipeline(settings, retriever=retriever, open_history=False)
    pipeline._retrieval_top_k = 2  # a tighter bound than the default of 8
    run = pipeline.run(
        _query(
            "ModuleNotFoundError: No module named 'requests'",
            question="why does the requests module fail?",
        )
    )

    citations = [
        line for line in run.query.retrieved_context.splitlines() if line.startswith("[")
    ]
    assert 0 < len(citations) <= 2, "top_k must bound how much context is attached"

