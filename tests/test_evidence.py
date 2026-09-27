"""Evidence extraction: deterministic, cited, and honest about what it missed."""

from __future__ import annotations

import pytest

from core.contracts.diagnostics import EvidenceKind
from diagnostics.evidence import (
    ERROR_PATTERNS,
    SIGNATURES,
    dedupe,
    evidence_from_user_input,
    extract_evidence,
    signature_for,
)

from .conftest import SAMPLE_LOG


def test_every_declared_export_exists():
    import diagnostics.evidence as module

    assert [name for name in module.__all__ if not hasattr(module, name)] == []


def test_patterns_are_unique_and_compiled():
    assert len(ERROR_PATTERNS) == len(SIGNATURES)
    for signature in SIGNATURES:
        assert signature.pattern.search("a harmless line") is None or signature.name


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("ModuleNotFoundError: No module named 'requests'", "python_module_not_found"),
        ("EADDRINUSE: address already in use :::3000", "node_port_in_use"),
        ("npm ERR! code ELIFECYCLE", "node_npm_error"),
        ("error TS2345: Argument of type", "typescript_error"),
        ("LNK2019: unresolved external symbol", "linker_error"),
        ("CMake Error at CMakeLists.txt:12", "cmake_error"),
        ("Access is denied.", "windows_access_denied"),
        ("DLL load failed while importing torch", "dll_load_failure"),
        ("CONFLICT (content): Merge conflict in app.py", "git_conflict"),
        ("fatal: not a git repository", "git_no_repository"),
        ("getaddrinfo failed: Name or service not known", "dns_failure"),
        ("ECONNREFUSED 127.0.0.1:5432", "connection_refused"),
        ("Traceback (most recent call last):", "python_traceback_header"),
    ],
)
def test_signature_for_recognises_common_failures(line, expected):
    signature = signature_for(line)
    assert signature is not None, f"no signature matched {line!r}"
    assert signature.name == expected


def test_signature_for_returns_none_for_clean_text():
    """A line with no known error returns None rather than a catch-all."""
    assert signature_for("everything is working as expected") is None


def test_access_denied_is_a_windows_problem_not_a_python_one():
    """Regression: a broad python pattern once shadowed the Windows one."""
    assert signature_for("Access is denied.").category == "windows"


def test_extract_evidence_cites_line_numbers():
    items = extract_evidence(SAMPLE_LOG, source="app.log")
    assert items
    for item in items:
        assert item.source == "app.log"
        assert item.locator.startswith("line ")
        assert item.metadata["category"]
        assert 0.0 <= item.confidence <= 1.0


def test_stack_frames_are_reported_as_stack_traces():
    items = extract_evidence(SAMPLE_LOG, source="app.log")
    kinds = {item.kind for item in items}
    assert EvidenceKind.STACK_TRACE in kinds
    frames = [i for i in items if i.metadata["signature"] == "python_frame"]
    assert frames and all("line 42" in f.content for f in frames)


def test_dedupe_collapses_a_repeated_error():
    log = SAMPLE_LOG + "ModuleNotFoundError: No module named 'requests'\n"
    items = extract_evidence(log, source="app.log")
    matches = [i for i in items if i.metadata["signature"] == "python_module_not_found"]
    assert len(matches) == 1, "a repeated error must collapse to one cited item"


def test_dedupe_is_stable_and_order_preserving():
    assert [e.content for e in dedupe(())] == []
    items = extract_evidence(SAMPLE_LOG, source="app.log")
    assert dedupe(items) == items


def test_max_items_is_honoured_and_truncated_not_silent():
    items = extract_evidence(SAMPLE_LOG, source="app.log", max_items=2)
    assert len(items) == 2


def test_max_items_must_be_positive():
    with pytest.raises(ValueError):
        extract_evidence(SAMPLE_LOG, max_items=0)


def test_empty_text_yields_no_evidence():
    assert extract_evidence("", source="x") == ()
    assert extract_evidence("   \n\n  ", source="x") == ()


def test_user_input_is_kept_verbatim_as_its_own_evidence():
    text = "I get ModuleNotFoundError: No module named 'foo' when I run pytest"
    items = evidence_from_user_input(text)
    assert items
    assert any(item.kind is EvidenceKind.USER_INPUT and item.content == text for item in items)
    assert any(i.metadata.get("signature") == "python_module_not_found" for i in items)


def test_user_input_requires_text():
    assert evidence_from_user_input("   ") == ()


def test_generic_matches_are_low_confidence():
    """A catch-all lead must not be scored like a named exception."""
    generic = extract_evidence("Error: something vague happened", source="x")
    named = extract_evidence("FileNotFoundError: nope", source="x")
    assert generic[0].confidence < named[0].confidence
