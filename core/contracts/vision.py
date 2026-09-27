"""Vision contract: structured observation of an image.

Kept separate from OCR on purpose. OCR answers "what characters are here";
vision answers "what is happening here", which is what turns a screenshot into
a usable error description. A vision backend is optional - the pipeline works
without it, because OCR output alone is already valid evidence.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from core.contracts.base import Capability, Component, Provenance
from core.contracts.ocr import ImageInput

__all__ = ["VisionBackend", "VisionObservation", "VisionRequest", "VisionResult", "VisionTask"]


class VisionTask(str, Enum):
    """What the caller wants to know about an image."""

    DESCRIBE = "describe"
    ERROR_SCREEN = "error_screen"
    UI_ELEMENTS = "ui_elements"
    DIFF = "diff"


@dataclass(frozen=True, slots=True)
class VisionRequest:
    """One image plus the question to answer about it."""

    image: ImageInput
    task: VisionTask = VisionTask.DESCRIBE
    prompt: str = ""
    max_output_tokens: int = 512

    def __post_init__(self) -> None:
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be >= 1")


@dataclass(frozen=True, slots=True)
class VisionObservation:
    """What a model reported about an image.

    ``confidence`` is the model's own estimate, or 0.0 when the backend does
    not provide one. It is never invented by the application.
    """

    summary: str
    details: tuple[str, ...] = field(default=())
    confidence: float = 0.0
    raw_text: str = ""

    def __post_init__(self) -> None:
        if not self.summary.strip():
            raise ValueError("a vision observation needs a summary")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")

    def as_lines(self) -> tuple[str, ...]:
        return (self.summary, *self.details)


@dataclass(frozen=True, slots=True)
class VisionResult:
    """An observation plus the provenance of the run that produced it."""

    observation: VisionObservation
    provenance: Provenance

    @property
    def is_npu(self) -> bool:
        """True when this was verified to have run on an NPU."""
        return self.provenance.is_npu


class VisionBackend(Component):
    """Interprets images beyond literal text extraction."""

    kind = "vision"

    @abstractmethod
    def describe(self, request: VisionRequest) -> VisionResult:
        """Analyse an image and return a structured observation."""

    def preferred_capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.IMAGE_CAPTION})
