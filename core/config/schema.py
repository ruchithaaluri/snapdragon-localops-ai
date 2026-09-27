"""Typed configuration schema.

Every setting has a default that is safe on a plain Windows x64 machine with
no models installed. Secrets are referenced by *environment variable name* only
(``api_key_env_var``), never by value, so a configuration file can be committed
and a log can be shared without leaking a key.

The model is a :class:`pydantic_settings.BaseSettings`, so values can come from
a TOML file, environment variables (``LOCALOPS_...``) or explicit overrides, in
that order of increasing precedence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.execution import AcceleratorKind, FallbackPolicy
from core.model_artifact import ModelFormat, ModelRole, ModelSource
from core.safety import DENIED_COMMAND_PATTERNS, RiskLevel

__all__ = [
    "AiSettings",
    "AppSettings",
    "BenchmarkSettings",
    "DiagnosticsSettings",
    "LoggingSettings",
    "ModelEntry",
    "ModelRole",
    "ProviderChoice",
    "RetrievalSettings",
    "RuntimeSettings",
    "SafetySettings",
    "UiSettings",
    "VerificationSettings",
]


class _Section(BaseModel):
    """Base for every configuration section: strict and self-documenting."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RuntimeSettings(_Section):
    """Which execution runtime to use and what to do when NPU is absent.

    ``preferred_targets`` is a preference, not a guarantee. Resolution happens
    in :func:`core.execution.resolve_target`, which only ever selects a target
    that was actually verified, and records the fallback when it happens.
    """

    runtime: str = "onnxruntime"
    preferred_targets: tuple[AcceleratorKind, ...] = (
        AcceleratorKind.NPU,
        AcceleratorKind.GPU,
        AcceleratorKind.CPU,
    )
    fallback_policy: FallbackPolicy = FallbackPolicy.ALLOW
    verify_execution: bool = True
    require_npu_verification: bool = True
    intra_op_threads: int | None = Field(default=None, ge=1)
    execution_providers: tuple[str, ...] = ()

    @field_validator("preferred_targets", mode="before")
    @classmethod
    def _coerce_targets(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> "RuntimeSettings":
        if AcceleratorKind.CPU not in self.preferred_targets and self.fallback_policy is FallbackPolicy.DENY:
            # Deny means "do not silently substitute hardware", so CPU must be
            # an explicit choice by the user, not an implied last resort.
            object.__setattr__(
                self,
                "preferred_targets",
                self.preferred_targets + (AcceleratorKind.CPU,),
            )
        return self


class LoggingSettings(_Section):
    """Structured logging output."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    json_console: bool = False
    file_enabled: bool = True
    file_name: str = "localops.jsonl"
    max_bytes: int = Field(default=5 * 1024 * 1024, ge=64 * 1024)
    backup_count: int = Field(default=5, ge=1)
    quiet_third_party: bool = True
    redact_keys: tuple[str, ...] = ()

    @property
    def log_path(self) -> Path:
        """Absolute path of the rotating log file under the data directory."""
        from core.config.paths import logs_dir

        return logs_dir() / self.file_name


class ModelEntry(_Section):
    """One configured model artifact.

    Phase 1 ships no models, so the default list is empty and every model-backed
    feature reports "not configured" instead of guessing a name.
    """

    role: ModelRole
    model_id: str = Field(min_length=1)
    path: Path
    format: ModelFormat = ModelFormat.UNKNOWN
    source: ModelSource = ModelSource.LOCAL
    sha256: str = ""
    quantization: str = ""
    enabled: bool = True
    options: dict[str, str] = Field(default_factory=dict)

    def resolved_path(self) -> Path:
        """Expand ``~`` and environment variables in the configured path."""
        return Path(self.path).expanduser()


class ProviderChoice(_Section):
    """Which implementation serves a role, and how it authenticates.

    ``api_key_env_var`` names an environment variable. Its value is read at
    runtime, only by an implementation that genuinely needs it, and is never
    written to configuration, logs or reports.
    """

    name: str = "unavailable"
    api_key_env_var: str | None = None
    options: dict[str, str] = Field(default_factory=dict)
    enabled: bool = False

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        return value.strip() or "unavailable"


class AiSettings(_Section):
    """Backend selection for the AI layer. All roles are off by default."""

    llm: ProviderChoice = Field(default_factory=ProviderChoice)
    embeddings: ProviderChoice = Field(default_factory=ProviderChoice)
    ocr: ProviderChoice = Field(default_factory=ProviderChoice)
    vision: ProviderChoice = Field(default_factory=ProviderChoice)
    models: tuple[ModelEntry, ...] = ()
    enable_llm_reasoning: bool = False


class RetrievalSettings(_Section):
    """Local vector/keyword retrieval. Fully offline by definition."""

    backend: str = "in_memory"
    top_k: int = Field(default=8, ge=1, le=200)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    chunk_size: int = Field(default=1200, ge=128)
    chunk_overlap: int = Field(default=200, ge=0)
    persist_index: bool = True
    include_error_patterns: bool = True

    @model_validator(mode="after")
    def _overlap_fits(self) -> "RetrievalSettings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class DiagnosticsSettings(_Section):
    """Evidence gathering and root-cause analysis behaviour."""

    max_evidence_items: int = Field(default=50, ge=1)
    confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    enable_llm_reasoning: bool = False
    require_evidence: bool = True
    max_candidates: int = Field(default=5, ge=1)


class VerificationSettings(_Section):
    """Post-fix verification. Read-only unless a user opts in."""

    timeout_seconds: float = Field(default=30.0, gt=0)
    read_only_only: bool = True
    require_approval_for_mutating_checks: bool = True
    stop_on_first_failure: bool = False


class BenchmarkSettings(_Section):
    """Measurement policy. Numbers are only ever produced by a real run."""

    enabled: bool = False
    warmup_iterations: int = Field(default=3, ge=0)
    measured_iterations: int = Field(default=10, ge=1)
    min_samples_for_statistics: int = Field(default=5, ge=1)
    targets: tuple[AcceleratorKind, ...] = (AcceleratorKind.NPU, AcceleratorKind.CPU)
    record_environment: bool = True
    output_dir: Path | None = None
    require_verified_npu: bool = True

    @field_validator("targets", mode="before")
    @classmethod
    def _coerce_targets(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value


class SafetySettings(_Section):
    """User-approval and remediation limits. Approval cannot be switched off."""

    allow_mutating_actions: bool = False
    allow_network_actions: bool = False
    require_reversible: bool = True
    require_approval: bool = True
    max_risk: RiskLevel = RiskLevel.MEDIUM
    denied_command_patterns: tuple[str, ...] = DENIED_COMMAND_PATTERNS

    @field_validator("require_approval")
    @classmethod
    def _approval_is_mandatory(cls, value: bool) -> bool:
        if not value:
            raise ValueError(
                "require_approval cannot be disabled: remediation always needs "
                "an explicit user decision"
            )
        return value

    def to_policy(self) -> "Any":  # noqa: ANN401 - avoids a circular import at type-check time
        """Build the runtime :class:`core.safety.SafetyPolicy` from settings."""
        from core.safety import SafetyPolicy

        return SafetyPolicy(
            allow_mutating_actions=self.allow_mutating_actions,
            allow_network_actions=self.allow_network_actions,
            require_reversible=self.require_reversible,
            max_risk=self.max_risk,
            denied_command_patterns=self.denied_command_patterns,
        )


class UiSettings(_Section):
    """Presentation preferences for the desktop shell."""

    theme: Literal["dark", "light"] = "dark"
    window_width: int = Field(default=1440, ge=800)
    window_height: int = Field(default=900, ge=600)
    restore_geometry: bool = True
    confirm_before_remediation: bool = True
    show_execution_panel: bool = True


class AppSettings(BaseSettings):
    """Root configuration object.

    Precedence (lowest to highest): built-in defaults, the TOML file, the
    environment, and explicit overrides passed to :func:`load_settings`.
    """

    model_config = SettingsConfigDict(
        env_prefix="LOCALOPS_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="forbid",
        validate_assignment=True,
    )

    app_name: str = "Snapdragon LocalOps AI"
    environment: Literal["development", "production"] = "development"
    telemetry: bool = False
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    ai: AiSettings = Field(default_factory=AiSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    diagnostics: DiagnosticsSettings = Field(default_factory=DiagnosticsSettings)
    verification: VerificationSettings = Field(default_factory=VerificationSettings)
    benchmark: BenchmarkSettings = Field(default_factory=BenchmarkSettings)
    safety: SafetySettings = Field(default_factory=SafetySettings)
    ui: UiSettings = Field(default_factory=UiSettings)

    @property
    def data_dir(self) -> Path:
        """Root directory for logs, database, indexes and reports."""
        from core.config.paths import data_dir

        return data_dir()

    @property
    def models_dir(self) -> Path:
        """Where model artifacts are expected to live."""
        from core.config.paths import models_dir

        return models_dir()

    def enabled_models(self) -> tuple[ModelEntry, ...]:
        return tuple(entry for entry in self.ai.models if entry.enabled)

