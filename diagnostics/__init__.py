"""Diagnostics: turning raw symptoms into an evidence-backed conclusion.

Three modules, in dependency order:

``evidence``
    Deterministic pattern matching over text the caller already has. It reads
    no files and contacts nothing, so every extracted item can be checked by
    reading the line it came from.
``engine``
    Groups that evidence into ranked :class:`~core.contracts.diagnostics.
    Hypothesis` objects. It never states a cause that no evidence supports.
``remediation``
    Turns a diagnosis into proposed actions, and applies them only after an
    explicit per-action approval.

The engine is useful without a language model: it reports
``reasoning_mode=RULES_ONLY`` and cites the evidence it used. It never dresses
deterministic pattern matching up as model output.
"""

from __future__ import annotations

from diagnostics.engine import RuleBasedDiagnosticEngine, build_query
from diagnostics.evidence import (
    ERROR_PATTERNS,
    SIGNATURES,
    dedupe,
    evidence_from_user_input,
    extract_evidence,
    signature_for,
)

__all__ = [
    "ERROR_PATTERNS",
    "SIGNATURES",
    "RuleBasedDiagnosticEngine",
    "build_query",
    "dedupe",
    "evidence_from_user_input",
    "extract_evidence",
    "signature_for",
]
