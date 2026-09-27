"""Retrieval contracts: the local knowledge layer.

Retrieval is always local. An index stores chunk embeddings produced by an
:class:`~core.contracts.embeddings.EmbeddingBackend`; searching never leaves the
machine and never calls a hosted vector database.

Scores are reported alongside every hit, and every result carries provenance so
a citation in a diagnosis can be traced back to a chunk, a model and a run.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from core.contracts.base import Capability, Component, Provenance

__all__ = [
    "DocumentChunk",
    "IndexEntry",
    "Retriever",
    "RetrievalResult",
    "SearchHit",
    "SearchQuery",
    "VectorIndex",
]


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A retrievable unit of text with provenance of its own."""

    chunk_id: str
    text: str
    source: str = ""
    document_id: str = ""
    ordinal: int = 0
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("a chunk requires a chunk_id")
        if not self.text.strip():
            raise ValueError(f"chunk {self.chunk_id} is empty")

    def citation(self) -> str:
        """Short reference for the UI, e.g. ``log.txt#12``."""
        if self.source and self.ordinal:
            return f"{self.source}#{self.ordinal}"
        return self.source or self.chunk_id


@dataclass(frozen=True, slots=True)
class IndexEntry:
    """A chunk plus the vector that represents it."""

    chunk: DocumentChunk
    vector: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.vector:
            raise ValueError(f"chunk {self.chunk.chunk_id} has an empty vector")


@dataclass(frozen=True, slots=True)
class SearchQuery:
    """A query plus the retrieval knobs for one call."""

    text: str
    top_k: int = 8
    min_score: float = 0.0
    filters: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("a search query must contain text")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One retrieved chunk with its similarity score."""

    chunk: DocumentChunk
    score: float

    def citation(self) -> str:
        return f"{self.chunk.citation()} (score {self.score:.3f})"


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Ranked hits for one query, with the provenance of the search itself."""

    query: str
    hits: tuple[SearchHit, ...]
    provenance: Provenance
    total_candidates: int = 0

    def __len__(self) -> int:
        return len(self.hits)

    def __iter__(self):
        return iter(self.hits)

    def texts(self) -> tuple[str, ...]:
        return tuple(hit.chunk.text for hit in self.hits)

    def citations(self) -> tuple[str, ...]:
        return tuple(hit.citation() for hit in self.hits)

    def as_context(self, max_chars: int = 8000) -> str:
        """Render the hits as a context block for a prompt.

        Truncates at ``max_chars`` and marks the cut, so a language model is
        never silently shown a partial context.
        """
        parts: list[str] = []
        used = 0
        truncated = False
        for hit in self.hits:
            block = f"[{hit.chunk.citation()}]\n{hit.chunk.text}"
            if used + len(block) > max_chars:
                truncated = True
                break
            parts.append(block)
            used += len(block)
        text = "\n\n".join(parts)
        if truncated:
            text += "\n\n[context truncated: more retrieved passages were not included]"
        return text


class VectorIndex(Component):
    """Stores and searches chunk vectors."""

    kind = "vector_index"

    @abstractmethod
    def add(self, entries: Sequence[IndexEntry]) -> int:
        """Add entries, returning how many were stored."""

    @abstractmethod
    def search(self, query_vector: tuple[float, ...], query: SearchQuery) -> tuple[SearchHit, ...]:
        """Return the best matches for a query vector."""

    @abstractmethod
    def delete(self, chunk_ids: Iterable[str]) -> int:
        """Remove entries by chunk id, returning how many were removed."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector length this index accepts."""

    @abstractmethod
    def count(self) -> int:
        """Number of stored entries."""

    def save(self, path: str) -> None:
        """Persist the index. Optional."""
        raise NotImplementedError(f"{self.info.name} cannot be persisted")

    def load(self, path: str) -> None:
        """Restore a persisted index. Optional."""
        raise NotImplementedError(f"{self.info.name} cannot be restored")

    def preferred_capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.VECTOR_SEARCH})


class Retriever(Component):
    """Embeds a query, searches an index, and returns cited passages.

    This is the interface the diagnostics engine depends on, so an alternative
    implementation (hybrid BM25 + vectors, a different index, a different
    embedding model) is a drop-in replacement.
    """

    kind = "retrieval"

    @abstractmethod
    def retrieve(self, query: SearchQuery) -> RetrievalResult:
        """Retrieve the passages most relevant to a query."""

    def index(self) -> VectorIndex | None:
        """The underlying index, when the retriever exposes one."""
        return None

    def preferred_capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.VECTOR_SEARCH, Capability.KEYWORD_SEARCH})
