"""Embedding contract: text in, vectors out, provenance attached.

Embeddings power local retrieval. The contract is deliberately strict about two
things that are easy to fake and hard to notice:

* the vector dimension is fixed per backend and checked on construction,
* every batch carries the :class:`~core.contracts.base.Provenance` of the run
  that produced it, so a retrieval result can be traced to a model and a
  machine.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field

from core.contracts.base import Capability, Component, Provenance

__all__ = [
    "EmbeddingBackend",
    "EmbeddingBatch",
    "EmbeddingRequest",
    "EmbeddingVector",
]


@dataclass(frozen=True, slots=True)
class EmbeddingRequest:
    """A batch of texts to embed."""

    texts: tuple[str, ...]
    normalize: bool = True
    truncate: bool = True

    def __post_init__(self) -> None:
        if not self.texts:
            raise ValueError("an embedding request needs at least one text")
        if any(not text.strip() for text in self.texts):
            raise ValueError("an embedding request must not contain empty text")

    def __len__(self) -> int:
        return len(self.texts)


@dataclass(frozen=True, slots=True)
class EmbeddingVector:
    """A single vector plus the position of its source text."""

    values: tuple[float, ...]
    index: int = 0

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("an embedding vector must not be empty")
        if any(value != value for value in self.values):
            raise ValueError("embedding vector contains NaN")

    def __len__(self) -> int:
        return len(self.values)

    def as_list(self) -> list[float]:
        return list(self.values)

    def norm(self) -> float:
        """Euclidean norm, computed without numpy."""
        return sum(value * value for value in self.values) ** 0.5

    def cosine_similarity(self, other: "EmbeddingVector") -> float:
        """Cosine similarity to another vector of the same length."""
        if len(self.values) != len(other.values):
            raise ValueError(
                "cannot compare vectors of different dimensions: "
                f"{len(self.values)} vs {len(other.values)}"
            )
        dot = sum(a * b for a, b in zip(self.values, other.values, strict=True))
        denom = self.norm() * other.norm()
        return dot / denom if denom else 0.0


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    """Vectors for a batch of texts, in input order."""

    vectors: tuple[EmbeddingVector, ...]
    provenance: Provenance
    dimension: int = 0
    truncated: int = 0

    def __post_init__(self) -> None:
        if not self.vectors:
            raise ValueError("an embedding batch must not be empty")
        first = len(self.vectors[0].values)
        if self.dimension and self.dimension != first:
            raise ValueError(
                f"declared dimension {self.dimension} does not match the vectors ({first})"
            )
        if not self.dimension:
            object.__setattr__(self, "dimension", first)
        wrong = [v.index for v in self.vectors if len(v.values) != self.dimension]
        if wrong:
            raise ValueError(f"vectors with an unexpected dimension at indices {wrong}")

    def __len__(self) -> int:
        return len(self.vectors)

    def __iter__(self) -> Iterator[EmbeddingVector]:
        return iter(self.vectors)

    def __getitem__(self, item: int) -> EmbeddingVector:
        return self.vectors[item]


class EmbeddingBackend(Component):
    """Turns text into vectors for local retrieval."""

    kind = "embeddings"

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Length of the vectors this backend produces."""

    @abstractmethod
    def embed(self, request: EmbeddingRequest) -> EmbeddingBatch:
        """Embed a batch of texts, preserving order."""

    def embed_one(self, text: str) -> EmbeddingVector:
        """Convenience wrapper for a single string."""
        return self.embed(EmbeddingRequest(texts=(text,)))[0]

    def preferred_capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.TEXT_EMBEDDING})

    def close(self) -> None:
        """Backends holding a native session should override this."""
