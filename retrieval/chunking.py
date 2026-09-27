"""Text chunking for the local index.

Deliberately simple and dependency-free: a sliding window over lines with a
character budget. Long logs are chunked at line boundaries so a chunk never
splits a stack frame in half, and chunks carry the line range they came from so
a citation can point back at the file.

This is not a model and makes no semantic claims.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

from core.contracts.retrieval import DocumentChunk
from core.ids import new_id

__all__ = ["chunk_text", "chunk_lines"]


def chunk_text(
    text: str,
    *,
    source: str = "",
    document_id: str = "",
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> tuple[DocumentChunk, ...]:
    """Split text into overlapping chunks on line boundaries.

    ``chunk_size`` and ``chunk_overlap`` are in characters.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if not text.strip():
        return ()
    return tuple(chunk_lines(text.splitlines(), source=source, document_id=document_id,
                             chunk_size=chunk_size, chunk_overlap=chunk_overlap))


def chunk_lines(
    lines: Sequence[str],
    *,
    source: str = "",
    document_id: str = "",
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> tuple[DocumentChunk, ...]:
    """Group lines into chunks of at most ``chunk_size`` characters.

    Overlap is expressed in characters of trailing context, approximated by
    whole lines, so a chunk never ends mid-line.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[DocumentChunk] = []
    buffer: list[str] = []
    size = 0
    start_line = 1
    line_no = 0
    ordinal = 0

    def flush(end_line: int) -> None:
        nonlocal buffer, size, start_line, ordinal
        body = "\n".join(buffer).strip()
        if body:
            chunks.append(
                DocumentChunk(
                    chunk_id=new_id("chk"),
                    text=body,
                    source=source,
                    document_id=document_id or source,
                    ordinal=ordinal,
                    metadata={
                        "start_line": str(start_line),
                        "end_line": str(end_line),
                    },
                )
            )
            ordinal += 1
        # Carry the tail forward so context is not lost between chunks.
        tail: list[str] = []
        tail_size = 0
        for line in reversed(buffer):
            if tail_size + len(line) > chunk_overlap:
                break
            tail.insert(0, line)
            tail_size += len(line)
        buffer = list(tail)
        size = tail_size
        start_line = max(start_line, end_line - len(tail) + 1)

    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        if size + len(line) + 1 > chunk_size and buffer:
            flush(line_no - 1)
        if not buffer:
            start_line = line_no
        buffer.append(line)
        size += len(line) + 1

    if buffer:
        flush(line_no)
    return tuple(chunks)


def iter_chunks(texts: Iterable[tuple[str, str]], **kwargs: object) -> Iterator[DocumentChunk]:
    """Chunk many ``(source, text)`` pairs in order."""
    for source, text in texts:
        yield from chunk_text(text, source=source, **kwargs)  # type: ignore[arg-type]
