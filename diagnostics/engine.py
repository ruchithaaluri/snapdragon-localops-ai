"""The rule-based diagnostic engine: evidence in, ranked hypotheses out.

This engine is deterministic on purpose. It groups the
:class:`~core.contracts.diagnostics.Evidence` it is given by the category the
signatures in :mod:`diagnostics.evidence` assigned, and turns each group into a
:class:`~core.contracts.diagnostics.Hypothesis` that cites the exact evidence
items behind it.

Two rules shape everything here:

* **No unsourced claim.** A hypothesis is only ever built from evidence that is
  actually present, and it cites those evidence ids. When the evidence does not
  support any conclusion the engine raises
  :class:`~core.errors.InsufficientEvidenceError` rather than guessing - "I do
  not know yet" is a valid, useful answer.
* **No borrowed authority.** The result is reported as
  ``reasoning_mode=RULES_ONLY``. Pattern matching is not inference, and this
  module never dresses it up as model output. Its provenance names a pure-Python
  CPU run, because that is exactly what produced the answer.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from core.contracts.base import Capability, ProviderHealth, ProviderInfo, Provenance
from core.contracts.diagnostics import (
    Diagnosis,
    DiagnosticEngine,
    DiagnosticQuery,
    Evidence,
    EvidenceKind,
    Hypothesis,
    HypothesisStatus,
    ReasoningMode,
)
from core.errors import InsufficientEvidenceError
from core.execution import ExecutionTarget
from core.observability import get_logger, log_event
from core.timing import measure
from core.version import __version__

from diagnostics.evidence import SIGNATURES, dedupe, extract_evidence

__all__ = ["CATEGORY_LABELS", "RuleBasedDiagnosticEngine", "build_query"]

_log = get_logger(__name__)

#: Human-readable name for each signature category.
CATEGORY_LABELS: dict[str, str] = {
    "python": "Python",
    "node": "Node / TypeScript",
    "build": "native build",
    "windows": "Windows",
    "vcs": "version control",
    "network": "network",
    "model_runtime": "local model runtime",
    "generic": "unclassified",
}

#: Evidence kinds that actually describe a failure. User questions and retrieved
#: passages are context, not proof, and never form a hypothesis on their own.
_FAULT_KINDS: frozenset[EvidenceKind] = frozenset(
    {
        EvidenceKind.ERROR_MESSAGE,
        EvidenceKind.STACK_TRACE,
        EvidenceKind.COMMAND_OUTPUT,
        EvidenceKind.LOG_FILE,
        EvidenceKind.TERMINAL,
    }
)

#: Never report more than this, however strong the evidence looks. A rule-based
#: engine claiming 0.99 is overstating how much it knows.


class RuleBasedDiagnosticEngine(DiagnosticEngine):
    """Groups evidence into ranked, cited hypotheses without a model.

    Parameters
    ----------
    require_evidence:
        When ``True`` (the default) a query carrying no fault-shaped evidence
        raises :class:`~core.errors.InsufficientEvidenceError` instead of
        returning an empty diagnosis.
    confidence_threshold:
        Hypotheses below this are still returned, but marked
        :attr:`~core.contracts.diagnostics.HypothesisStatus.INCONCLUSIVE`, so the
        UI can separate a weak lead from a supported finding.
    max_hypotheses:
        Upper bound on how many hypotheses one analysis returns. Evidence from
        many unrelated categories makes many groups; the user needs the few that
        actually matter.
    """

    kind = "diagnostics"

    def __init__(
        self,
        *,
        require_evidence: bool = True,
        confidence_threshold: float = 0.30,
        max_hypotheses: int = 5,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be within [0, 1]")
        if max_hypotheses < 1:
            raise ValueError("max_hypotheses must be >= 1")
        self._require_evidence = require_evidence
        self._threshold = confidence_threshold
        self._max_hypotheses = max_hypotheses

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="rules-engine",
            kind=self.kind,
            version=__version__,
            capabilities=frozenset({Capability.ROOT_CAUSE_RANKING}),
            detail=(
                f"deterministic pattern matching over {len(SIGNATURES)} known "
                "error signatures; no language model involved"
            ),
        )

    def health(self) -> ProviderHealth:
        return ProviderHealth.ready(f"{len(SIGNATURES)} error signatures compiled")

    def analyze(self, query: DiagnosticQuery) -> Diagnosis:
        """Rank the evidence in ``query`` into cited hypotheses.

        Raises
        ------
        InsufficientEvidenceError
            When the query carries no fault-shaped evidence and
            ``require_evidence`` is set. The engine does not invent a cause.
        """
        with measure() as watch:
            hypotheses, cited, gaps = self._build_hypotheses(query.evidence)

        if not hypotheses:
            detail = f"{len(query.evidence)} evidence item(s), none describing a fault"
            if self._require_evidence:
                raise InsufficientEvidenceError(
                    "not enough evidence to diagnose this problem",
                    question=query.question,
                    detail=detail,
                    hint=(
                        "paste the full error output or log excerpt; a single "
                        "line is rarely enough to identify a cause"
                    ),
                )
            return Diagnosis(
                query=query.question,
                summary="No recognised error pattern was found in the supplied evidence.",
                reasoning_mode=ReasoningMode.RULES_ONLY,
                gaps=(*gaps, detail),
                provenance=self._provenance(watch.milliseconds),
            )

        ranked = tuple(sorted(hypotheses, key=lambda h: (-h.confidence, h.statement)))
        kept = ranked[: min(self._max_hypotheses, query.max_hypotheses)]
        omitted = len(ranked) - len(kept)
        if omitted > 0:
            label = "y" if omitted == 1 else "ies"
            gaps.append(f"{omitted} lower-ranked categor{label} omitted")

        log_event(
            _log,
            logging.INFO,
            "diagnostics.analyzed",
            hypotheses=len(kept),
            evidence=len(cited),
            confidence=kept[0].confidence,
            reasoning_mode=ReasoningMode.RULES_ONLY.value,
        )
        return Diagnosis(
            query=query.question,
            hypotheses=kept,
            summary=self._summarise(query, kept, cited),
            reasoning_mode=ReasoningMode.RULES_ONLY,
            confidence=kept[0].confidence,
            provenance=self._provenance(watch.milliseconds),
            evidence_ids=cited,
            gaps=tuple(gaps),
        )


    # -- internals --------------------------------------------------------

    def _provenance(self, elapsed_ms: float) -> Provenance:
        """Describe the work that produced the answer, honestly.

        This is a pure-Python pass over text: no model ran, and no accelerator
        did either. Saying exactly that is the point of the field.
        """
        return Provenance(
            provider="rules-engine",
            execution=ExecutionTarget.cpu(
                "pure-python",
                detail="deterministic pattern matching, no model inference",
            ),
            model_id="",
            elapsed_ms=elapsed_ms,
            notes=(f"{len(SIGNATURES)} signatures compiled",),
        )

    def _build_hypotheses(
        self, evidence: Sequence[Evidence]
    ) -> tuple[tuple[Hypothesis, ...], tuple[str, ...], list[str]]:
        """Group evidence by category and turn each group into a hypothesis.

        Returns the hypotheses, the flat tuple of cited evidence ids, and a list
        of caveats the UI should show alongside the answer.
        """
        groups: dict[str, list[Evidence]] = {}
        context_only: list[Evidence] = []
        for item in evidence:
            if item.kind not in _FAULT_KINDS:
                context_only.append(item)
                continue
            groups.setdefault(item.metadata.get("category", "generic"), []).append(item)

        hypotheses: list[Hypothesis] = []
        cited: list[str] = []
        gaps: list[str] = []
        if context_only:
            gaps.append(
                f"{len(context_only)} context item(s) (user input, documents) were "
                "not treated as evidence of a fault"
            )

        for category, items in groups.items():
            items = dedupe(items)
            signature = self._dominant_signature(items)
            confidence = self._score(category, items)
            label = CATEGORY_LABELS.get(category, category)
            hypotheses.append(
                Hypothesis(
                    statement=f"A {label} problem is the most likely explanation.",
                    evidence_ids=tuple(item.evidence_id for item in items),
                    confidence=confidence,
                    status=(
                        HypothesisStatus.SUPPORTED
                        if confidence >= self._threshold
                        else HypothesisStatus.INCONCLUSIVE
                    ),
                    rationale=self._rationale(label, signature, items),
                )
            )
            cited.extend(item.evidence_id for item in items)
            if signature == "accelerator_unavailable":
                gaps.append(
                    "an accelerator was reported unavailable; any result must be "
                    "described as a CPU run unless a verified NPU run confirms otherwise"
                )
        return tuple(hypotheses), tuple(dict.fromkeys(cited)), gaps

    def _score(self, category: str, items: Sequence[Evidence]) -> float:
        """Confidence for one category, derived from the evidence inside it.

        Starts at the most trustworthy single item and grows with corroboration:
        two different signatures agreeing matters more than one strong line
        repeated a thousand times. Capped at :data:`_MAX_CONFIDENCE`.
        """
        best = max(item.confidence for item in items)
        distinct = {item.metadata.get("signature", "") for item in items} - {""}
        score = best + 0.05 * (len(distinct) - 1)
        if len(items) >= 3:
            score += 0.05
        if category == "generic":
            # A catch-all match is a lead, never a conclusion.
            score = min(score, 0.55)
        return round(min(score, _MAX_CONFIDENCE), 2)

    @staticmethod
    def _dominant_signature(items: Sequence[Evidence]) -> str:
        """The most frequent signature name within a group."""
        counts: dict[str, int] = {}
        for item in items:
            name = item.metadata.get("signature", "")
            if name:
                counts[name] = counts.get(name, 0) + 1
        return max(counts, key=lambda name: counts[name]) if counts else ""

    @staticmethod
    def _rationale(label: str, signature: str, items: Sequence[Evidence]) -> str:
        """Explain in one sentence why this hypothesis was raised."""
        cause = next((s.typical_cause for s in SIGNATURES if s.name == signature), "")
        extra = f" and {len(items) - 1} related line(s)" if len(items) > 1 else ""
        text = f"Matched {signature or 'a known error pattern'} in {items[0].reference()}{extra}."
        if cause:
            text += f" Usually this means: {cause}."
        return text

    @staticmethod
    def _summarise(
        query: DiagnosticQuery, kept: Sequence[Hypothesis], cited: Sequence[str]
    ) -> str:
        """One-paragraph answer that states its evidence count up front."""
        top = kept[0]
        plural = "" if len(cited) == 1 else "s"
        lines = [
            f"Based on {len(cited)} cited evidence item{plural}, the most likely "
            f"explanation is: {top.statement}",
            top.rationale,
        ]
        if len(kept) > 1:
            other = ", ".join(h.statement.rstrip(".") for h in kept[1:])
            lines.append(f"Other categories present: {other}.")
        if query.retrieved_context.strip():
            lines.append(
                "Retrieved project context was considered, but this answer is "
                "rule-derived and cites only the evidence listed."
            )
        return " ".join(lines)



def build_query(
    question: str,
    text: str = "",
    *,
    kind: EvidenceKind = EvidenceKind.LOG_FILE,
    source: str = "",
    run_id: str = "",
    max_hypotheses: int = 5,
) -> DiagnosticQuery:
    """Assemble a :class:`DiagnosticQuery` from raw text.

    A convenience for the UI and the CLI: the caller already has a blob of text -
    a pasted error, a log excerpt - and this turns it into a query whose
    evidence has been extracted. Files are still read by the caller; nothing
    here walks a directory.
    """
    evidence = extract_evidence(text, kind=kind, source=source) if text.strip() else ()
    return DiagnosticQuery(
        question=question,
        evidence=evidence,
        run_id=run_id,
        max_hypotheses=max_hypotheses,
    )

_MAX_CONFIDENCE = 0.95
