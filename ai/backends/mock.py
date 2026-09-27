"""Mock AI backend: deterministic output, no model, no download.

Purpose: let the entire workflow - evidence, diagnosis, fix, approval,
verification, benchmark - be developed, demonstrated and tested on any machine,
including one with no AI runtime and no budget for model inference.

What this module is **not**: a source of real AI answers. Every result is
labelled as mock data in three independent ways:

* ``ProviderInfo.name`` is prefixed with ``mock``,
* :meth:`MockAIBackend.is_mock` returns ``True`` and the UI shows a banner,
* every :class:`Provenance` uses ``AcceleratorKind.UNKNOWN`` (so nothing claims
  real CPU/NPU execution) and carries the note
  ``"mock backend: deterministic sample output, not model inference"``.

The mock embeddings use a hashing trick over words. They are stable and
reproducible, which is what the retrieval demo needs, but they capture no
semantics: similar sentences do not get similar vectors. Replace this backend
with :class:`~core.contracts.ai.AIBackend` returning a real model before drawing
any conclusion about retrieval quality.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path

from core.contracts.ai import AIBackend
from core.contracts.base import Capability, ProviderHealth, ProviderInfo, ProviderStatus, Provenance
from core.contracts.embeddings import EmbeddingBackend, EmbeddingBatch, EmbeddingRequest, EmbeddingVector
from core.contracts.llm import ChatMessage, GenerationParams, GenerationResult, LlmBackend
from core.contracts.ocr import ImageInput, OcrBackend, OcrPage, OcrResult, TextRegion
from core.contracts.vision import VisionBackend, VisionObservation, VisionRequest, VisionResult, VisionTask
from core.errors import ProviderUnavailableError
from core.execution import AcceleratorKind, ExecutionTarget
from core.timing import measure

__all__ = [
    "MOCK_NOTE",
    "MockAIBackend",
    "MockEmbeddingBackend",
    "MockLlmBackend",
    "MockOcrBackend",
    "MockVisionBackend",
    "mock_provenance",
    "mock_target",
]

#: Appended to the provenance of every mock result.
MOCK_NOTE = "mock backend: deterministic sample output, not model inference"

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def mock_target(provider: str = "mock") -> ExecutionTarget:
    """Execution target for mock output.

    Deliberately ``UNKNOWN`` and unverified: a mock performs no inference, so
    claiming it ran on a CPU or an NPU would be false in both directions.
    """
    return ExecutionTarget(
        kind=AcceleratorKind.UNKNOWN,
        provider=provider,
        device="none (mock output)",
    )


def mock_provenance(provider: str, elapsed_ms: float = 0.0) -> Provenance:
    """Provenance for a mock result, flagged as mock and as unexecuted."""
    target = ExecutionTarget(
        kind=AcceleratorKind.UNKNOWN,
        provider=f"mock/{provider}",
        device="none (mock output)",
    )
    return Provenance(
        provider=f"mock/{provider}",
        execution=target,
        model_id="mock",
        elapsed_ms=elapsed_ms,
        notes=(MOCK_NOTE,),
    )


def _info(kind: str, description: str, capabilities: frozenset[Capability]) -> ProviderInfo:
    return ProviderInfo(
        name=f"mock-{kind}",
        kind=kind,
        version="0.1.0-mock",
        capabilities=capabilities,
        detail=description,
    )


class MockOcrBackend(OcrBackend):
    """Text extraction stand-in.

    It cannot read pixels. When an image has a ``.txt`` sidecar - which is how
    the bundled sample screenshot is stored - that text is used as the
    "recognised" output. Otherwise it raises
    :class:`~core.errors.ProviderUnavailableError` instead of inventing text.
    """

    def __init__(self, *, languages: tuple[str, ...] = ("en",)) -> None:
        self._languages = languages

    @property
    def info(self) -> ProviderInfo:
        return _info("ocr", "Reads a .txt sidecar next to the image; cannot decode pixels.", {Capability.OCR_TEXT})

    def health(self) -> ProviderHealth:
        return ProviderHealth(ProviderStatus.READY, "mock OCR active (sidecar text only)")

    def available_languages(self) -> tuple[str, ...]:
        return self._languages

    def recognize(self, image: ImageInput, *, languages: Sequence[str] = ()) -> OcrResult:
        """Return the sidecar text as one page of OCR regions."""
        path = Path(image.path)
        sidecar = path.with_suffix(".txt")
        if not sidecar.is_file():
            raise ProviderUnavailableError(
                "mock OCR cannot read pixels: no text sidecar next to the image",
                image=str(path),
                expected_sidecar=str(sidecar),
            )
        with measure() as watch:
            lines = [
                line for line in sidecar.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip()
            ]
        regions = tuple(
            TextRegion(text=line, confidence=0.5, left=0, top=index * 12, width=0, height=12)
            for index, line in enumerate(lines)
        )
        page = OcrPage(index=0, regions=regions)
        return OcrResult(
            pages=(page,),
            provenance=mock_provenance("ocr", watch.milliseconds),
            languages=tuple(languages) or self._languages,
        )


class MockVisionBackend(VisionBackend):
    """Image "description" stand-in.

    Returns a fixed, clearly-labelled observation. It performs no analysis of
    the image, and says so in the summary the user sees.
    """

    def __init__(self) -> None:
        self._calls = 0

    @property
    def info(self) -> ProviderInfo:
        return _info("vision", "Returns a fixed placeholder observation; no image analysis.", {Capability.IMAGE_CAPTION})

    def health(self) -> ProviderHealth:
        return ProviderHealth(ProviderStatus.READY, "mock vision active (placeholder output)")

    def describe(self, request: VisionRequest) -> VisionResult:
        """Return a placeholder observation for the requested task."""
        self._calls += 1
        observation = VisionObservation(
            summary=(
                f"[MOCK VISION] No image analysis was performed for task "
                f"'{request.task.value}'. Connect a real vision backend to "
                f"interpret {request.image.path.name}."
            ),
            details=(f"image: {request.image.path.name}", f"task: {request.task.value}"),
            confidence=0.0,
        )
        return VisionResult(observation=observation, provenance=mock_provenance("vision"))


class MockEmbeddingBackend(EmbeddingBackend):
    """Deterministic hashed bag-of-words embeddings.

    Stable across processes and platforms (SHA-256 based), which is what an
    offline demo and a reproducible test need. Captures no semantics: the
    vector reflects word overlap only.
    """

    def __init__(self, dimension: int = 64) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self._dimension = dimension

    @property
    def info(self) -> ProviderInfo:
        return _info(
            "embeddings",
            f"Hashed bag-of-words, dimension {self._dimension}. No semantic meaning.",
            {Capability.TEXT_EMBEDDING},
        )

    def health(self) -> ProviderHealth:
        return ProviderHealth(ProviderStatus.READY, f"mock embeddings (dim {self._dimension})")

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed_text(self, text: str) -> tuple[float, ...]:
        values = [0.0] * self._dimension
        for word in _WORD_RE.findall(text.lower()):
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            values[bucket] += sign
        norm = sum(value * value for value in values) ** 0.5
        if norm:
            values = [value / norm for value in values]
        return tuple(values)

    def embed(self, request: EmbeddingRequest) -> EmbeddingBatch:
        with measure() as watch:
            vectors = tuple(
                EmbeddingVector(values=self._embed_text(text), index=index)
                for index, text in enumerate(request.texts)
            )
        return EmbeddingBatch(
            vectors=vectors,
            provenance=mock_provenance("embeddings", watch.milliseconds),
            dimension=self._dimension,
        )



#: Keyword -> canned explanation, used to make the mock reasoning deterministic
#: and inspectable. Keys are matched as substrings, lower-cased.
_SAMPLE_RULES: tuple[tuple[str, str], ...] = (
    ("modulenotfounderror", "a module imported by the failing code is not installed in the active environment"),
    ("cannot find module", "a module imported by the failing code is not installed in the active environment"),
    ("filenotfounderror", "a path used by the failing code does not exist at the expected location"),
    ("no such file", "a path used by the failing code does not exist at the expected location"),
    ("permissionerror", "the process lacks permission for the file, directory or device it tried to use"),
    ("access is denied", "the process lacks permission for the resource it tried to use"),
    ("commandnotfounderror", "the shell or process could not find the command it was asked to run"),
    ("is not recognized", "the command is missing from PATH or the shell session"),
    ("out of memory", "the process exceeded the memory available to it"),
    ("memoryerror", "the process exceeded the memory available to it"),
    ("timeout", "an operation exceeded the time it was allowed to take"),
    ("connection refused", "a service the code expected to be listening was not accepting connections"),
    ("eaddrinuse", "the port the service tried to bind is already in use"),
    ("address already in use", "the port the service tried to bind is already in use"),
    ("syntaxerror", "the source file could not be parsed because of a syntax error"),
    ("indentationerror", "the source file could not be parsed because of inconsistent indentation"),
    ("importerror", "a required dependency of the module is missing or broken"),
)

_MOCK_PREFIX = "[MOCK LLM - deterministic sample output, not model inference]"


class MockLlmBackend(LlmBackend):
    """Deterministic "reasoning" over keywords found in the prompt.

    The output is a lookup, not an inference. It exists so the diagnosis,
    approval and verification screens can be built and demonstrated end to end
    with zero model load, and so tests have a stable backend.
    """

    def __init__(self, *, max_rules: int = 3) -> None:
        self._max_rules = max_rules
        self._calls = 0

    @property
    def info(self) -> ProviderInfo:
        return _info(
            "llm",
            "Keyword lookup over a fixed rule table. No model is loaded.",
            {Capability.TEXT_GENERATION, Capability.CHAT},
        )

    def health(self) -> ProviderHealth:
        return ProviderHealth(ProviderStatus.READY, f"mock llm active ({len(_SAMPLE_RULES)} rules)")

    @property
    def call_count(self) -> int:
        """How many completions were produced (shown in the UI)."""
        return self._calls

    def rules_matched(self, text: str) -> tuple[tuple[str, str], ...]:
        """Return the (pattern, explanation) pairs a prompt triggers."""
        lowered = text.lower()
        hits = [(pattern, why) for pattern, why in _SAMPLE_RULES if pattern in lowered]
        return tuple(hits[: self._max_rules])

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        params: GenerationParams | None = None,
    ) -> GenerationResult:
        """Return a deterministic summary built from matched sample rules."""
        self._calls += 1
        prompt = "\n".join(message.content for message in messages)
        with measure() as watch:
            matches = self.rules_matched(prompt)
        if matches:
            lines = [f"{_MOCK_PREFIX}"]
            lines += [f"- matched '{pattern}': {why}" for pattern, why in matches]
            lines.append("This explanation comes from a fixed sample table, not from a model.")
            text = "\n".join(lines)
        else:
            text = (
                f"{_MOCK_PREFIX}\n"
                "- no sample rule matched this input, so no explanation is offered.\n"
                "Connect a real language model backend for actual analysis."
            )
        return GenerationResult(
            text=text,
            provenance=mock_provenance("llm", watch.milliseconds),
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
        )


class MockAIBackend(AIBackend):
    """Bundles the mock backends behind the :class:`AIBackend` interface.

    This is the default in Phase 1: the application starts, demonstrates the
    full workflow and runs its tests on any machine, with no model download and
    no accelerator.
    """

    def __init__(self) -> None:
        self._ocr = MockOcrBackend()
        self._embeddings = MockEmbeddingBackend()
        self._vision = MockVisionBackend()
        self._llm = MockLlmBackend()

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="mock-ai-backend",
            kind=self.kind,
            version="0.1.0-mock",
            capabilities=self.preferred_capabilities(),
            detail="Deterministic demo backend. No model, no download, no accelerator.",
        )

    def is_mock(self) -> bool:
        return True

    @property
    def ocr(self) -> OcrBackend:
        return self._ocr

    @property
    def embeddings(self) -> EmbeddingBackend:
        return self._embeddings

    @property
    def vision(self) -> VisionBackend:
        return self._vision

    @property
    def llm(self) -> LlmBackend:
        return self._llm

    def preferred_capabilities(self) -> frozenset[Capability]:
        return frozenset(
            {
                Capability.OCR_TEXT,
                Capability.TEXT_EMBEDDING,
                Capability.IMAGE_CAPTION,
                Capability.TEXT_GENERATION,
                Capability.CHAT,
            }
        )

