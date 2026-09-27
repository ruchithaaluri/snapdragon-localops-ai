"""Retriever: embed a query, search the local index, return cited passages.

Binds an :class:`~core.contracts.embeddings.EmbeddingBackend` to a
:class:`~core.contracts.retrieval.VectorIndex` and hides the plumbing from the
diagnostics engine. Indexing is explicit (:meth:`Retriever.index_documents`) so
nothing scans the filesystem behind the user's back - only the files the user
picks are ever read.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from core.contracts.base import Capability, ProviderHealth, ProviderInfo, ProviderStatus
from core.contracts.embeddings import EmbeddingBackend, EmbeddingRequest
from core.contracts.retrieval import (
    DocumentChunk,
    IndexEntry,
    Retriever,
    RetrievalResult,
    SearchQuery,
    VectorIndex,
)
from core.errors import ProviderUnavailableError
from core.timing import measure

from retrieval.chunking import chunk_text

__all__ = ["LocalRetriever"]


class LocalRetriever(Retriever):
    """Embed-and-search over a local index. Never touches the network."""

    def __init__(self, embeddings: EmbeddingBackend, index: VectorIndex) -> None:
        if embeddings is None:
            raise ProviderUnavailableError("a retriever needs an embedding backend")
        self._embeddings = embeddings
        self._index = index

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name=f"retriever({self._embeddings.info.name})",
            kind=self.kind,
            version="0.1.0",
            capabilities=frozenset({Capability.VECTOR_SEARCH}),
            detail=(
                f"Embeddings from {self._embeddings.info.name} "
                f"(dim {self._embeddings.dimension}), index {self._index.info.name}."
            ),
        )

    def health(self) -> ProviderHealth:
        embedding_health = self._embeddings.health()
        if not embedding_health.is_usable:
            return embedding_health
        return ProviderHealth(
            ProviderStatus.READY, f"{self._index.count()} chunks ready to search"
        )

    def index(self) -> VectorIndex:
        return self._index

    @property
    def embeddings(self) -> EmbeddingBackend:
        """The embedding backend used for both documents and queries."""
        return self._embeddings

    def index_documents(
        self,
        documents: Sequence[tuple[str, str]],
        *,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
    ) -> int:
        """Chunk and index ``(source, text)`` pairs the caller already has.

        Files are read by the caller: this method never walks a directory.
        """
        chunks: list[DocumentChunk] = []
        for source, text in documents:
            chunks.extend(
                chunk_text(
                    text,
                    source=source,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            )
        if not chunks:
            return 0
        batch = self._embeddings.embed(
            EmbeddingRequest(texts=tuple(chunk.text for chunk in chunks))
        )
        entries = [
            IndexEntry(chunk=chunk, vector=vector.values)
            for chunk, vector in zip(chunks, batch, strict=True)
        ]
        return self._index.add(entries)

    def index_file(self, path: str | Path, *, max_bytes: int = 512_000) -> int:
        """Index one file. Bounded by ``max_bytes`` so a huge log cannot stall the UI."""
        target = Path(path)
        if not target.is_file():
            return 0
        text = target.read_text(encoding="utf-8", errors="replace")[:max_bytes]
        return self.index_documents([(target.name, text)])

    def retrieve(self, query: SearchQuery) -> RetrievalResult:
        """Embed the query and return the best matching passages.

        The result's provenance is the *embedding run* that produced the query
        vector, so a retrieval result inherits the real execution target
        (including a verified NPU run, when there is one) rather than an
        invented one.
        """
        with measure() as watch:
            batch = self._embeddings.embed(EmbeddingRequest(texts=(query.text,)))
            vector = batch[0]
            hits = self._index.search(vector.values, query)
        provenance = replace(
            batch.provenance,
            provider=f"retriever({self._embeddings.info.name})",
            elapsed_ms=watch.milliseconds,
            notes=("local index search",),
        )
        return RetrievalResult(
            query=query.text,
            hits=hits,
            provenance=provenance,
            total_candidates=self._index.count(),
        )
