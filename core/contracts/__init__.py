"""Every interface the application is written against.

Import from this package rather than from a submodule; the layout here is an
implementation detail. The layering rule is one-directional:

    contracts  <-  ai, retrieval, diagnostics, verification, benchmark
    app (UI)   ->  contracts + the concrete layers, never the reverse

No module in this package imports PySide6, an AI engine, or a layer package.
"""

from __future__ import annotations

from core.contracts.ai import AIBackend, AIBackendCapabilities
from core.contracts.base import (
    Capability,
    Component,
    ProviderHealth,
    ProviderInfo,
    ProviderStatus,
    Provenance,
)
from core.contracts.benchmark import (
    BenchmarkCase,
    BenchmarkReport,
    BenchmarkRunner,
    Measurement,
    MeasurementProvenance,
    Statistic,
)
from core.contracts.diagnostics import (
    Diagnosis,
    DiagnosticEngine,
    DiagnosticQuery,
    Evidence,
    EvidenceKind,
    Hypothesis,
    HypothesisStatus,
    ReasoningMode,
)
from core.contracts.embeddings import (
    EmbeddingBackend,
    EmbeddingBatch,
    EmbeddingRequest,
    EmbeddingVector,
)
from core.contracts.inference import InferenceRuntime, LoadedModel, RuntimeProbe
from core.contracts.llm import (
    ChatMessage,
    ChatRole,
    GenerationParams,
    GenerationResult,
    LlmBackend,
)
from core.contracts.ocr import ImageInput, OcrBackend, OcrPage, OcrResult, TextRegion
from core.contracts.pipeline import (
    Pipeline,
    PipelineRun,
    PipelineStage,
    StageOutcome,
    StageRecord,
    StageStatus,
)
from core.contracts.remediation import (
    ExecutionOutcome,
    RemediationExecutor,
    RemediationPlanner,
    RemediationResult,
)
from core.contracts.retrieval import (
    DocumentChunk,
    IndexEntry,
    Retriever,
    RetrievalResult,
    SearchHit,
    SearchQuery,
    VectorIndex,
)
from core.contracts.verification import (
    CheckKind,
    CheckOutcome,
    CheckResult,
    VerificationCheck,
    VerificationReport,
    VerificationRunner,
)
from core.contracts.vision import (
    VisionBackend,
    VisionObservation,
    VisionRequest,
    VisionResult,
    VisionTask,
)

__all__ = [
    "AIBackend",
    "AIBackendCapabilities",
    "BenchmarkCase",
    "BenchmarkReport",
    "BenchmarkRunner",
    "Capability",
    "ChatMessage",
    "ChatRole",
    "CheckKind",
    "CheckOutcome",
    "CheckResult",
    "Component",
    "Diagnosis",
    "DiagnosticEngine",
    "DiagnosticQuery",
    "DocumentChunk",
    "EmbeddingBackend",
    "EmbeddingBatch",
    "EmbeddingRequest",
    "EmbeddingVector",
    "Evidence",
    "EvidenceKind",
    "ExecutionOutcome",
    "GenerationParams",
    "GenerationResult",
    "Hypothesis",
    "HypothesisStatus",
    "ImageInput",
    "IndexEntry",
    "InferenceRuntime",
    "LlmBackend",
    "LoadedModel",
    "Measurement",
    "MeasurementProvenance",
    "OcrBackend",
    "OcrPage",
    "OcrResult",
    "Pipeline",
    "PipelineRun",
    "PipelineStage",
    "ProviderHealth",
    "ProviderInfo",
    "ProviderStatus",
    "Provenance",
    "ReasoningMode",
    "RemediationExecutor",
    "RemediationPlanner",
    "RemediationResult",
    "Retriever",
    "RetrievalResult",
    "RuntimeProbe",
    "SearchHit",
    "SearchQuery",
    "StageOutcome",
    "StageRecord",
    "StageStatus",
    "Statistic",
    "TextRegion",
    "VectorIndex",
    "VerificationCheck",
    "VerificationReport",
    "VerificationRunner",
    "VisionBackend",
    "VisionObservation",
    "VisionRequest",
    "VisionResult",
    "VisionTask",
]
