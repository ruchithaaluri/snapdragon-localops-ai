"""Image input and OCR contracts.

OCR is the first stage of the evidence pipeline: a screenshot becomes text with
positions and per-region confidence, so a downstream diagnosis can point at the
exact pixels it relied on.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from core.contracts.base import Capability, Component, Provenance

__all__ = ["ImageInput", "OcrBackend", "OcrPage", "OcrResult", "TextRegion"]


@dataclass(frozen=True, slots=True)
class ImageInput:
    """An image to analyse, referenced by a local path.

    Only local paths are accepted. Accepting a URL here would quietly add a
    network dependency to a privacy-first tool, so a caller must download a
    remote image first and then pass a file.
    """

    path: Path
    captured_at: str = ""
    source: str = "file"
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))

    def exists(self) -> bool:
        return self.path.is_file()

    def require_exists(self) -> Path:
        """Return the path or raise a clear error."""
        from core.errors import ProviderUnavailableError

        if not self.exists():
            raise ProviderUnavailableError(
                "image file not found", path=str(self.path)
            )
        return self.path


@dataclass(frozen=True, slots=True)
class TextRegion:
    """A recognised text block with its location, in pixel coordinates."""

    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("OCR confidence must be within [0, 1]")
        if not self.text.strip():
            raise ValueError("an OCR region must contain text")

    def as_line(self) -> str:
        return f"[{self.left},{self.top} {self.width}x{self.height}] {self.text}"


@dataclass(frozen=True, slots=True)
class OcrPage:
    """Every region recognised on one page or image."""

    index: int
    regions: tuple[TextRegion, ...]
    width: int = 0
    height: int = 0

    @property
    def text(self) -> str:
        """Regions ordered top-to-bottom then left-to-right."""
        ordered = sorted(self.regions, key=lambda r: (r.top, r.left))
        return "\n".join(region.text for region in ordered)

    @property
    def mean_confidence(self) -> float:
        if not self.regions:
            return 0.0
        return sum(r.confidence for r in self.regions) / len(self.regions)

    def filter_confidence(self, minimum: float) -> "OcrPage":
        """Drop low-confidence regions, keeping the rest of the page intact."""
        kept = tuple(r for r in self.regions if r.confidence >= minimum)
        return OcrPage(index=self.index, regions=kept, width=self.width, height=self.height)


@dataclass(frozen=True, slots=True)
class OcrResult:
    """Output of one OCR run over one image."""

    pages: tuple[OcrPage, ...]
    provenance: Provenance
    languages: tuple[str, ...] = field(default=())

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def regions(self) -> tuple[TextRegion, ...]:
        return tuple(region for page in self.pages for region in page.regions)


class OcrBackend(Component):
    """Extracts text from screenshots, images and rendered terminals.

    Implementations must not fabricate text. If recognition fails, the backend
    either returns an empty result with an explanatory note or raises
    :class:`~core.errors.ProviderUnavailableError`.
    """

    kind = "ocr"

    @abstractmethod
    def recognize(self, image: ImageInput, *, languages: Sequence[str] = ()) -> OcrResult:
        """Run OCR over an image and return every recognised region."""

    def available_languages(self) -> tuple[str, ...]:
        """Languages this backend can recognise."""
        return ()

    def preferred_capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.OCR_TEXT})
