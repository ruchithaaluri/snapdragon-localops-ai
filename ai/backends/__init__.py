"""Concrete AI backends.

Three implementations, one facade. The application depends only on
:class:`~core.contracts.ai.AIBackend`, so swapping any of these is a
configuration change:

``mock`` (:class:`~ai.backends.mock.MockAIBackend`)
    Deterministic, offline, no model. The default, and the only backend that
    needs nothing installed.
``local`` (:class:`~ai.backends.local.LightweightLocalAIBackend`)
    Real ONNX Runtime inference against a model that already exists on disk.
    Downloads nothing and never falls back to the mock.
``snapdragon`` (:class:`~ai.backends.snapdragon.SnapdragonAIBackend`)
    The NPU path for the HP target. Structurally complete, **not validated** -
    see that module's docstring before treating it as operational.
"""

from __future__ import annotations

from ai.backends.local import (
    NOTE_TOKEN_GENERATION,
    LightweightLocalAIBackend,
    LocalModelLlmBackend,
)
from ai.backends.mock import (
    MOCK_NOTE,
    MockAIBackend,
    MockEmbeddingBackend,
    MockLlmBackend,
    MockOcrBackend,
    MockVisionBackend,
)
from ai.backends.snapdragon import (
    SNAPDRAGON_VALIDATION_STATUS,
    SnapdragonAIBackend,
    SnapdragonRuntime,
    is_snapdragon_host,
)

__all__ = [
    "MOCK_NOTE",
    "NOTE_TOKEN_GENERATION",
    "SNAPDRAGON_VALIDATION_STATUS",
    "LightweightLocalAIBackend",
    "LocalModelLlmBackend",
    "MockAIBackend",
    "MockEmbeddingBackend",
    "MockLlmBackend",
    "MockOcrBackend",
    "MockVisionBackend",
    "SnapdragonAIBackend",
    "SnapdragonRuntime",
    "is_snapdragon_host",
]
