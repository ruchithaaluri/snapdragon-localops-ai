"""Local retrieval: chunking, an in-memory vector index, and a retriever.

No network, no native dependency, and no filesystem walking - the retriever
indexes exactly the text it is handed.
"""

from __future__ import annotations

from retrieval.chunking import chunk_lines, chunk_text
from retrieval.index import InMemoryVectorIndex, cosine_similarity
from retrieval.retriever import LocalRetriever

__all__ = [
    "InMemoryVectorIndex",
    "LocalRetriever",
    "chunk_lines",
    "chunk_text",
    "cosine_similarity",
]
