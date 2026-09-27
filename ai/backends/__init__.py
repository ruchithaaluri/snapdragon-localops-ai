"""Concrete AI backends."""

from __future__ import annotations

from ai.backends.mock import (
    MOCK_NOTE,
    MockAIBackend,
    MockEmbeddingBackend,
    MockLlmBackend,
    MockOcrBackend,
    MockVisionBackend,
)

__all__ = [
    "MOCK_NOTE",
    "MockAIBackend",
    "MockEmbeddingBackend",
    "MockLlmBackend",
    "MockOcrBackend",
    "MockVisionBackend",
]
