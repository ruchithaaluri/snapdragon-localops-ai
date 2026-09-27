"""Diagnostics contracts: evidence in, evidence-backed conclusion out.

The engine is the part of the system most able to sound confident and be wrong,
so the data model is built to make that visible:

* every conclusion cites the :class:`Evidence` ids that support it,
* confidence is a number the engine must justify, not a vibe,
* when evidence is thin the engine says so - either by returning
  ``insufficient_evidence`` or by raising
  :class:`~core.errors.InsufficientEvidenceError`,
* a diagnosis produced without a language model is marked
  ``reasoned_by_model=None`` rather than dressed up as AI output.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from core.contracts.base import Capability, Component, Provenance
from core.ids import new_id, utc_now

__all__ = [
    "Diagnosis",
    "DiagnosticEngine",
    "DiagnosticQuery",
    "Evidence",
    "EvidenceKind",
    "Hypothesis",
    "HypothesisStatus",
    "ReasoningMode",
]


class EvidenceKind(str, Enum):
    """What an evidence item was extracted from."""

    SCREENSHOT = "screenshot"
    TERMINAL = "terminal"
    LOG_FILE = "log_file"
    ERROR_MESSAGE = "error_message"
    STACK_TRACE = "stack_trace"
    ENVIRONMENT = "environment"
    SOURCE_CODE = "source_code"
    DOCUMENT = "document"
    USER_INPUT = "user_input"
    RETRIEVAL = "retrieval"
    MODEL_OUTPUT = "model_output"
    COMMAND_OUTPUT = "command_output"


class HypothesisStatus(str, Enum):
    """Lifecycle of a hypothesis inside a run."""

    PROPOSED = "proposed"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


class ReasoningMode(str, Enum):
    """How the diagnosis was produced. Always reported, never implied."""

    #: Deterministic rules over the evidence. No model involved.
    RULES_ONLY = "rules_only"
    #: A local language model reasoned over the evidence.
    LOCAL_MODEL = "local_model"
    #: A model reasoned and the answer cites specific evidence ids.
    LOCAL_MODEL_GROUNDED = "local_model_grounded"


@dataclass(frozen=True, slots=True)
class Evidence:
    """A single extracted fact, with where it came from."""

    kind: EvidenceKind
    content: str
    source: str = ""
    captured_at: str = field(default_factory=lambda: utc_now().isoformat(timespec="seconds"))
    evidence_id: str = field(default_factory=lambda: new_id("ev"))
    confidence: float = 1.0
    locator: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("evidence must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("evidence confidence must be within [0, 1]")

    def excerpt(self, max_chars: int = 240) -> str:
        """Short, single-line preview for the UI."""
        flat = " ".join(self.content.split())
        return flat if len(flat) <= max_chars else flat[: max_chars - 1] + "\u2026"

    def reference(self) -> str:
        """Citation label, e.g. ``log_file:app.log``."""
        return f"{self.kind.value}:{self.source or self.evidence_id}"


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """A candidate explanation, tied to the evidence that supports it."""

    statement: str
    evidence_ids: tuple[str, ...] = field(default=())
    confidence: float = 0.0
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    hypothesis_id: str = field(default_factory=lambda: new_id("hyp"))
    contradicting_evidence_ids: tuple[str, ...] = field(default=())
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.statement.strip():
            raise ValueError("a hypothesis must state a claim")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("hypothesis confidence must be within [0, 1]")
        if not self.evidence_ids:
            # A claim with no cited evidence is the exact failure this
            # application must never produce.
            raise ValueError("a hypothesis must cite at least one evidence id")

    @property
    def is_supported(self) -> bool:
        return bool(self.evidence_ids) and self.confidence > 0.0


@dataclass(frozen=True, slots=True)
class Diagnosis:
    """The engine's answer, with its reasoning made explicit."""

    query: str
    hypotheses: tuple[Hypothesis, ...] = field(default=())
    summary: str = ""
    reasoning_mode: ReasoningMode = ReasoningMode.RULES_ONLY
    confidence: float = 0.0
    provenance: Provenance | None = None
    evidence_ids: tuple[str, ...] = field(default=())
    gaps: tuple[str, ...] = field(default=())
    model_output: str = ""
    created_at: str = field(default_factory=lambda: utc_now().isoformat(timespec="seconds"))

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("diagnosis confidence must be within [0, 1]")
        if self.hypotheses and not self.evidence_ids:
            raise ValueError("a diagnosis with hypotheses must cite its evidence")

    @property
    def is_empty(self) -> bool:
        return not self.hypotheses and not self.summary

    @property
    def top_hypothesis(self) -> Hypothesis | None:
        """Highest-confidence hypothesis, if any."""
        return max(self.hypotheses, key=lambda h: h.confidence, default=None)

    def ranked(self) -> tuple[Hypothesis, ...]:
        """Hypotheses ordered by descending confidence, ties broken by id."""
        return tuple(sorted(self.hypotheses, key=lambda h: (-h.confidence, h.hypothesis_id)))



@dataclass(frozen=True, slots=True)
class DiagnosticQuery:
    """Everything the engine is given for one analysis.

    ``evidence`` is supplied by the caller, never invented: in the pipeline the
    evidence store has already extracted it from a screenshot, the terminal
    buffer or a log file.
    """

    question: str
    evidence: tuple[Evidence, ...] = field(default=())
    context: str = ""
    project_root: Path | None = None
    run_id: str = ""
    max_hypotheses: int = 5
    retrieved_context: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("a diagnostic query needs a question")
        if self.max_hypotheses < 1:
            raise ValueError("max_hypotheses must be >= 1")

    def evidence_by_id(self) -> dict[str, Evidence]:
        """Index the evidence by id for citation checks."""
        return {item.evidence_id: item for item in self.evidence}

    def with_evidence(self, extra: Sequence[Evidence]) -> "DiagnosticQuery":
        """Return a new query with additional evidence appended."""
        return DiagnosticQuery(
            question=self.question,
            evidence=(*self.evidence, *extra),
            context=self.context,
            project_root=self.project_root,
            run_id=self.run_id,
            max_hypotheses=self.max_hypotheses,
            retrieved_context=self.retrieved_context,
            metadata=dict(self.metadata),
        )


class DiagnosticEngine(Component):
    """Produces an evidence-backed diagnosis.

    An implementation that has no language model available must still be
    useful: it may return rule-derived hypotheses with
    ``reasoning_mode=RULES_ONLY``. What it must not do is invent model output.
    """

    kind = "diagnostics"

    @abstractmethod
    def analyze(self, query: DiagnosticQuery) -> Diagnosis:
        """Analyse the supplied evidence and return a diagnosis."""

    def preferred_capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.ROOT_CAUSE_RANKING})

