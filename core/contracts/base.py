"""Shared vocabulary for every pluggable component.

:class:`Provenance` is the important piece: every result produced by a backend
carries *who* produced it, *where* it ran, and *how long* it took. Its
constructor enforces the NPU rule structurally - a result object cannot even be
built if it claims unverified NPU execution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import ClassVar

from core.errors import UnverifiedNpuClaimError
from core.execution import AcceleratorKind, ExecutionTarget

__all__ = [
    "Capability",
    "Component",
    "ProviderHealth",
    "ProviderInfo",
    "ProviderStatus",
    "Provenance",
]


class Capability(str, Enum):
    """Optional features a backend may support.

    Capabilities are queried, never assumed: a retriever that supports
    ``PERSISTENT_INDEX`` is asked before it is asked to save.
    """

    TEXT_GENERATION = "text_generation"
    CHAT = "chat"
    STREAMING = "streaming"
    OCR_TEXT = "ocr_text"
    OCR_LAYOUT = "ocr_layout"
    IMAGE_CAPTION = "image_caption"
    TEXT_EMBEDDING = "text_embedding"
    VECTOR_SEARCH = "vector_search"
    KEYWORD_SEARCH = "keyword_search"
    PERSISTENT_INDEX = "persistent_index"
    ROOT_CAUSE_RANKING = "root_cause_ranking"
    REMEDIATION_PLANNING = "remediation_planning"
    VERIFICATION_EXECUTION = "verification_execution"
    BENCHMARK_MEASUREMENT = "benchmark_measurement"


class ProviderStatus(str, Enum):
    """Readiness of a component, as shown in the UI.

    ``NOT_CONFIGURED`` and ``UNAVAILABLE`` are distinct on purpose: the first
    means the user has not set anything up, the second means something is
    installed but cannot be used here.
    """

    READY = "ready"
    DEGRADED = "degraded"
    NOT_CONFIGURED = "not_configured"
    UNAVAILABLE = "unavailable"
    ERROR = "error"

    @property
    def is_usable(self) -> bool:
        return self in {ProviderStatus.READY, ProviderStatus.DEGRADED}


@dataclass(frozen=True, slots=True)
class ProviderInfo:
    """Static description of a component, safe to show in the UI."""

    name: str
    kind: str
    version: str = ""
    capabilities: frozenset[Capability] = field(default_factory=frozenset)
    detail: str = ""
    requires: tuple[str, ...] = field(default=())

    def has(self, capability: Capability) -> bool:
        return capability in self.capabilities

    def describe(self) -> str:
        return f"{self.name} ({self.kind}) v{self.version}" if self.version else f"{self.name} ({self.kind})"


@dataclass(frozen=True, slots=True)
class ProviderHealth:
    """Point-in-time health of a component."""

    status: ProviderStatus
    detail: str = ""
    checked_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    @classmethod
    def ready(cls, detail: str = "") -> "ProviderHealth":
        return cls(ProviderStatus.READY, detail)

    @classmethod
    def not_configured(cls, detail: str = "") -> "ProviderHealth":
        return cls(ProviderStatus.NOT_CONFIGURED, detail)

    @property
    def is_usable(self) -> bool:
        return self.status.is_usable


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where a result came from. Travels with every backend result.

    Constructing one of these with an unverified NPU target raises
    :class:`~core.errors.UnverifiedNpuClaimError`, so an unproven NPU result
    cannot travel through the system unnoticed.
    """

    provider: str
    execution: ExecutionTarget
    model_id: str = ""
    elapsed_ms: float = 0.0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.execution.kind is AcceleratorKind.NPU and not self.execution.is_verified:
            raise UnverifiedNpuClaimError(
                "result claims NPU execution without a verified run",
                provider=self.provider,
                state=self.execution.verification.state.value,
            )

    @property
    def is_npu(self) -> bool:
        """True when the work was verified to have run on an NPU."""
        return self.execution.npu_execution_claimed

    def summary(self) -> str:
        """One-line description for logs and report headers."""
        base = f"{self.provider} on {self.execution.describe()}"
        if self.model_id:
            base = f"{self.model_id} via {base}"
        return f"{base} in {self.elapsed_ms:.1f} ms"


class Component(ABC):
    """Base class for every replaceable backend.

    Implementations live in the layer packages and are registered by name. A
    component that cannot run must say so through :meth:`health` and raise
    :class:`~core.errors.ProviderUnavailableError` from its work methods - it
    must never return invented output.
    """

    #: Short identifier of the role this component fills.
    kind: ClassVar[str] = "component"

    @property
    @abstractmethod
    def info(self) -> ProviderInfo:
        """Static description of this component."""

    def health(self) -> ProviderHealth:
        """Report readiness. Default is a cheap, optimistic probe."""
        return ProviderHealth.ready()

    def close(self) -> None:
        """Release resources. Must be idempotent."""

    def supports(self, capability: Capability) -> bool:
        return capability in self.info.capabilities

    def __enter__(self) -> "Component":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
