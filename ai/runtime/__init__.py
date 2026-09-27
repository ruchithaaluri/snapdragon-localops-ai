"""Execution runtimes. Optional dependencies only, imported lazily."""

from __future__ import annotations

from ai.runtime.onnxruntime_backend import (
    OnnxRuntimeBackend,
    classify_execution_provider,
    onnxruntime_available,
)

__all__ = [
    "OnnxRuntimeBackend",
    "classify_execution_provider",
    "onnxruntime_available",
]
