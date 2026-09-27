"""The ``AIBackend`` facade.

One object the application talks to for "AI", regardless of what is underneath:

    AIBackend
      |- MockAIBackend        deterministic, no model (Phase 1, default)
      |- LocalAIBackend       ONNX Runtime + local models (Phase 2)
      |- SnapdragonAIBackend  vendor NPU runtime on the HP target (Phase 3+)

The facade exposes the individual capabilities separately (:attr:`AIBackend.ocr`,
:attr:`AIBackend.embeddings`, :attr:`AIBackend.vision`, :attr:`AIBackend.llm`)
because a deployment may have OCR and embeddings without a language model, and
the UI must be able to show exactly which capability is missing.

A backend must declare whether it is a mock via :meth:`AIBackend.is_mock`. The
UI displays that prominently, so demo output can never be mistaken for a real
model's answer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from core.contracts.base import Capability, Component, ProviderHealth, ProviderStatus
from core.contracts.embeddings import EmbeddingBackend
from core.contracts.llm import LlmBackend
from core.contracts.ocr import OcrBackend
from core.contracts.vision import VisionBackend

__all__ = ["AIBackend", "AIBackendCapabilities"]


@dataclass(frozen=True, slots=True)
class AIBackendCapabilities:
    """Which AI capabilities a backend actually provides right now."""

    ocr: bool = False
    embeddings: bool = False
    vision: bool = False
    llm: bool = False
    is_mock: bool = False

    def missing(self) -> tuple[str, ...]:
        """Names of the capabilities that are not available."""
        gaps = []
        if not self.ocr:
            gaps.append("ocr")
        if not self.embeddings:
            gaps.append("embeddings")
        if not self.vision:
            gaps.append("vision")
        if not self.llm:
            gaps.append("llm")
        return tuple(gaps)

    def as_rows(self) -> tuple[tuple[str, str], ...]:
        """Flatten to (capability, available) rows for a table."""
        return (
            ("ocr", "yes" if self.ocr else "no"),
            ("embeddings", "yes" if self.embeddings else "no"),
            ("vision", "yes" if self.vision else "no"),
            ("llm", "yes" if self.llm else "no"),
        )

    @property
    def is_complete(self) -> bool:
        return not self.missing()


class AIBackend(Component, ABC):
    """Aggregates the individual AI backends behind one replaceable object."""

    kind = "ai_backend"

    @property
    @abstractmethod
    def ocr(self) -> OcrBackend | None:
        """Text extraction backend, or None when unavailable."""

    @property
    @abstractmethod
    def embeddings(self) -> EmbeddingBackend | None:
        """Embedding backend, or None when unavailable."""

    @property
    @abstractmethod
    def vision(self) -> VisionBackend | None:
        """Vision backend, or None when unavailable."""

    @property
    @abstractmethod
    def llm(self) -> LlmBackend | None:
        """Language model backend, or None when unavailable."""

    def is_mock(self) -> bool:
        """True when this backend produces demo output, not model inference.

        Implementations that generate deterministic sample data **must**
        override this. The UI shows a warning banner whenever it is true.
        """
        return False

    def capabilities(self) -> AIBackendCapabilities:
        """Report which capabilities are wired up."""
        return AIBackendCapabilities(
            ocr=self.ocr is not None,
            embeddings=self.embeddings is not None,
            vision=self.vision is not None,
            llm=self.llm is not None,
            is_mock=self.is_mock(),
        )

    def health(self) -> ProviderHealth:
        """Aggregate the sub-backends' health into one status."""
        caps = self.capabilities()
        if caps.is_complete:
            return ProviderHealth(ProviderStatus.READY, f"{self.info.name} fully available")
        return ProviderHealth(
            ProviderStatus.DEGRADED,
            f"missing capabilities: {', '.join(caps.missing()) or 'none'}",
        )

    def preferred_capabilities(self) -> frozenset[Capability]:
        """Union of the capabilities the sub-backends advertise."""
        found: set[Capability] = set()
        for backend in (self.ocr, self.embeddings, self.vision, self.llm):
            if backend is not None:
                found.update(backend.info.capabilities)
        return frozenset(found)

    def close(self) -> None:
        """Close every sub-backend that owns resources."""
        for backend in (self.ocr, self.embeddings, self.vision, self.llm):
            if backend is not None:
                backend.close()
