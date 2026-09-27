"""Error hierarchy shared by every layer.

Callers catch :class:`LocalOpsError` to separate expected, user-facing failures
from programming errors. Each concrete error carries a machine-readable
``code`` so the UI and the structured logs can react to it consistently.
"""

from __future__ import annotations

from typing import Any, ClassVar


class LocalOpsError(Exception):
    """Base class for every error raised by this application."""

    code: ClassVar[str] = "localops_error"

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = context

    def __str__(self) -> str:  # pragma: no cover - trivial
        if not self.context:
            return self.message
        rendered = ", ".join(f"{k}={v!r}" for k, v in sorted(self.context.items()))
        return f"{self.message} ({rendered})"


class ConfigurationError(LocalOpsError):
    """Configuration is missing, malformed, or internally inconsistent."""

    code = "configuration_error"


class ProviderError(LocalOpsError):
    """Base class for failures originating from a pluggable component."""

    code = "provider_error"


class ProviderNotFoundError(ProviderError):
    """A provider was requested by name but is not registered."""

    code = "provider_not_found"


class ProviderUnavailableError(ProviderError):
    """A provider exists but cannot serve requests on this machine.

    Raised instead of returning a placeholder result. The application must
    never substitute fabricated inference output for a missing backend.
    """

    code = "provider_unavailable"


class ModelArtifactError(LocalOpsError):
    """A model artifact is missing, malformed, or not loadable."""

    code = "model_artifact_error"


class ModelIntegrityError(ModelArtifactError):
    """A model artifact failed its checksum/integrity verification."""

    code = "model_integrity_error"


class ExecutionTargetError(LocalOpsError):
    """Base class for execution-target (CPU/GPU/NPU) failures."""

    code = "execution_target_error"


class UnverifiedNpuClaimError(ExecutionTargetError):
    """Raised when something tries to report NPU execution that was not measured.

    This is the guard rail behind the "never claim NPU execution unless it is
    actually verified" rule. It is intentionally a hard failure: reporting an
    unverified accelerator is worse than reporting a slower CPU run.
    """

    code = "unverified_npu_claim"


class ExecutionBackendUnavailableError(ExecutionTargetError):
    """A requested execution backend does not exist on this machine."""

    code = "execution_backend_unavailable"


class RetrievalError(LocalOpsError):
    """Base class for retrieval/vector-index failures."""

    code = "retrieval_error"


class DimensionMismatchError(RetrievalError):
    """A vector was submitted with a dimension the index cannot accept."""

    code = "dimension_mismatch"


class DiagnosticsError(LocalOpsError):
    """Base class for diagnostics-engine failures."""

    code = "diagnostics_error"


class InsufficientEvidenceError(DiagnosticsError):
    """A diagnosis was requested without enough evidence to support one.

    Raised instead of guessing. The engine is expected to say "not enough
    information" rather than produce an unfounded root cause.
    """

    code = "insufficient_evidence"


class SafetyError(LocalOpsError):
    """Base class for safety-policy violations."""

    code = "safety_error"


class ApprovalRequiredError(SafetyError):
    """A remediation action was attempted without an explicit user approval."""

    code = "approval_required"


class ActionDeniedError(SafetyError):
    """A remediation action was rejected by the configured safety policy."""

    code = "action_denied"


class VerificationError(LocalOpsError):
    """Base class for verification-runner failures."""

    code = "verification_error"


class BenchmarkError(LocalOpsError):
    """Base class for benchmark-runner failures."""

    code = "benchmark_error"


class UnmeasuredBenchmarkError(BenchmarkError):
    """A benchmark report was asked for numbers that were never measured.

    Guards the "no fabricated benchmark numbers" rule at the data-model level.
    """

    code = "unmeasured_benchmark"


class StorageError(LocalOpsError):
    """Base class for SQLite/persistence failures."""

    code = "storage_error"


__all__ = [
    "ActionDeniedError",
    "ApprovalRequiredError",
    "BenchmarkError",
    "ConfigurationError",
    "DiagnosticsError",
    "DimensionMismatchError",
    "ExecutionBackendUnavailableError",
    "ExecutionTargetError",
    "InsufficientEvidenceError",
    "LocalOpsError",
    "ModelArtifactError",
    "ModelIntegrityError",
    "ProviderError",
    "ProviderNotFoundError",
    "ProviderUnavailableError",
    "RetrievalError",
    "SafetyError",
    "StorageError",
    "UnmeasuredBenchmarkError",
    "UnverifiedNpuClaimError",
    "VerificationError",
]
