"""AI layer: backend implementations behind the contracts in ``core.contracts``.

Contents
--------
``backends.mock``
    Deterministic demo backend (the Phase 1 default). No model, no download.
``runtime.onnxruntime_backend``
    Optional portable execution layer. The dependency is not installed in
    Phase 1; the module imports fine without it.
``registry``
    Chooses a backend from configuration. Never falls back silently.

Adding a real backend later - ``LocalAIBackend`` on ONNX Runtime, or
``SnapdragonAIBackend`` on the NPU target - means adding a module here and one
entry in :data:`ai.registry.BACKENDS`. No caller changes.
"""

from __future__ import annotations

from ai.backends.mock import (
    MOCK_NOTE,
    MockAIBackend,
    MockEmbeddingBackend,
    MockLlmBackend,
    MockOcrBackend,
    MockVisionBackend,
)
from ai.registry import BACKENDS, build_ai_backend, build_model_registry

__all__ = [
    "BACKENDS",
    "MOCK_NOTE",
    "MockAIBackend",
    "MockEmbeddingBackend",
    "MockLlmBackend",
    "MockOcrBackend",
    "MockVisionBackend",
    "build_ai_backend",
    "build_model_registry",
]
