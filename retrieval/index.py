"""In-memory vector index: exact cosine search, no dependencies.

Exact search over a few hundred chunks costs microseconds and needs no native
library, which is the right trade for a local developer tool. When the index
grows, :class:`~core.contracts.retrieval.VectorIndex` is the seam to swap in a
larger engine without touching callers.

The store keeps chunks in memory and can save/load a JSON snapshot, so a run can
be reproduced without re-indexing.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from core.contracts.base import Capability, ProviderHealth, ProviderInfo, ProviderStatus
from core.contracts.retrieval import (
    DocumentChunk,
    IndexEntry,
    SearchHit,
    SearchQuery,
    VectorIndex,
)
from core.errors import DimensionMismatchError, RetrievalError
from core.observability import get_logger
from core.timing import measure

__all__ = ["InMemoryVectorIndex", "cosine_similarity"]

_log = get_logger(__name__)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors. Zero-length safe."""
    if len(left) != len(right):
        raise DimensionMismatchError("vector dimensions differ", left=len(left), right=len(right))
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right, strict=True):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / ((left_norm**0.5) * (right_norm**0.5))


class InMemoryVectorIndex(VectorIndex):
    """Exact-search vector index holding chunks in process memory."""

    def __init__(self, *, dimension: int = 64, name: str = "in-memory") -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self._name = name
        self._entries: dict[str, IndexEntry] = {}

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name=self._name,
            kind=self.kind,
            version="0.1.0",
            capabilities=frozenset({Capability.VECTOR_SEARCH, Capability.PERSISTENT_INDEX}),
            detail=f"Exact cosine search, dimension {self._dimension}.",
        )

    def health(self) -> ProviderHealth:
        return ProviderHealth(ProviderStatus.READY, f"{self.count()} chunks indexed")

    @property
    def dimension(self) -> int:
        return self._dimension

    def count(self) -> int:
        return len(self._entries)

    def add(self, entries: Sequence[IndexEntry]) -> int:
        """Add or replace entries, returning how many were stored."""
        for entry in entries:
            if len(entry.vector) != self._dimension:
                raise DimensionMismatchError(
                    "vector dimension does not match the index",
                    chunk_id=entry.chunk.chunk_id,
                    expected=self._dimension,
                    actual=len(entry.vector),
                )
            self._entries[entry.chunk.chunk_id] = entry
        return len(entries)

    def search(
        self,
        query_vector: Sequence[float],
        query: SearchQuery,
    ) -> tuple[SearchHit, ...]:
        """Rank every entry by cosine similarity and return the best hits."""
        if len(query_vector) != self._dimension:
            raise DimensionMismatchError(
                "query vector dimension does not match the index",
                expected=self._dimension,
                actual=len(query_vector),
            )
        with measure() as watch:
            scored = [
                (cosine_similarity(query_vector, entry.vector), entry.chunk)
                for entry in self._entries.values()
                if self._matches_filters(entry.chunk, query.filters)
            ]
        scored.sort(key=lambda pair: (-pair[0], pair[1].chunk_id))
        hits = tuple(
            SearchHit(chunk=chunk, score=score)
            for score, chunk in scored[: query.top_k]
            if score >= query.min_score
        )
        _log.debug(
            "retrieval.search",
            extra={"fields": {"candidates": len(scored), "hits": len(hits),
                              "ms": round(watch.milliseconds, 3)}},
        )
        return hits

    @staticmethod
    def _matches_filters(chunk: DocumentChunk, filters: dict[str, str]) -> bool:
        """Every filter key must match the chunk metadata (or its source)."""
        for key, value in filters.items():
            if key == "source":
                if chunk.source != value:
                    return False
            elif chunk.metadata.get(key) != value:
                return False
        return True

    def delete(self, chunk_ids: Iterable[str]) -> int:
        """Remove entries by id, returning how many were removed."""
        removed = 0
        for chunk_id in chunk_ids:
            if self._entries.pop(chunk_id, None) is not None:
                removed += 1
        return removed


    def save(self, path: str) -> None:
        """Write a JSON snapshot of the index to ``path``."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dimension": self._dimension,
            "entries": [
                {
                    "chunk": {
                        "chunk_id": entry.chunk.chunk_id,
                        "text": entry.chunk.text,
                        "source": entry.chunk.source,
                        "document_id": entry.chunk.document_id,
                        "ordinal": entry.chunk.ordinal,
                        "metadata": entry.chunk.metadata,
                    },
                    "vector": list(entry.vector),
                }
                for entry in self._entries.values()
            ],
        }
        target.write_text(json.dumps(payload), encoding="utf-8")

    def load(self, path: str) -> None:
        """Restore a snapshot written by :meth:`save`."""
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RetrievalError("index snapshot could not be read", path=str(source)) from exc
        self._entries.clear()
        self._dimension = int(payload.get("dimension", self._dimension))
        for raw in payload.get("entries", []):
            data = raw["chunk"]
            self.add(
                [
                    IndexEntry(
                        chunk=DocumentChunk(
                            chunk_id=data["chunk_id"],
                            text=data["text"],
                            source=data.get("source", ""),
                            document_id=data.get("document_id", ""),
                            ordinal=int(data.get("ordinal", 0)),
                            metadata=data.get("metadata", {}),
                        ),
                        vector=tuple(float(value) for value in raw["vector"]),
                    )
                ]
            )

