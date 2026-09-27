"""Inference runtime contract: the hardware-agnostic execution layer.

This is the seam that keeps the application hardware-independent. Today the only
implementation loads ONNX models through ONNX Runtime; on the Snapdragon target
a vendor-backed runtime is added **here** and nothing above this line changes.

The rules a runtime implementation must follow:

* :meth:`InferenceRuntime.probe` reports what the installed runtime *advertises*.
  That is never the same as a measurement.
* :meth:`InferenceRuntime.load` returns a :class:`LoadedModel` whose target is
  unverified. Only a real inference - and :meth:`LoadedModel.mark_verified` -
  may turn it into a claimable result.
* A model that fails to load raises; it never falls back to a different model
  silently.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any

from core.contracts.base import Capability, Component, Provenance
from core.execution import (
    AcceleratorKind,
    ExecutionPreferences,
    ExecutionTarget,
    ExecutionVerification,
    RuntimeCapabilities,
)
from core.model_artifact import ModelArtifact

__all__ = ["InferenceRuntime", "LoadedModel", "RuntimeProbe", "model_provenance"]


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """A model resident in a runtime, with its (still unproven) target."""

    artifact: ModelArtifact
    target: ExecutionTarget
    session_id: str = ""
    load_ms: float = 0.0
    details: dict[str, str] = field(default_factory=dict)

    @property
    def model_id(self) -> str:
        return self.artifact.model_id

    @property
    def is_verified(self) -> bool:
        """True only once a real inference has been recorded."""
        return self.target.is_verified

    def mark_verified(self, method: str, detail: str = "") -> "LoadedModel":
        """Return a copy whose target is backed by a real inference.

        This is the single sanctioned way to promote a target to ``VERIFIED``,
        and it is called by runtimes immediately after a successful forward
        pass - never by configuration or by a UI action.
        """
        return replace(
            self,
            target=ExecutionTarget(
                kind=self.target.kind,
                provider=self.target.provider,
                device=self.target.device,
                runtime_version=self.target.runtime_version,
                verification=ExecutionVerification.verified(method, detail),
            ),
        )

    def assert_npu_claim_allowed(self) -> None:
        """Raise unless this model may be reported as running on an NPU."""
        self.target.assert_npu_claim_allowed()

    def provenance(self, elapsed_ms: float, notes: tuple[str, ...] = ()) -> Provenance:
        """Build the :class:`Provenance` for a result from this model.

        Raises :class:`~core.errors.UnverifiedNpuClaimError` for an unverified
        NPU target, which makes an unproven NPU result impossible to build.
        """
        return Provenance(
            provider=self.target.provider,
            execution=self.target,
            model_id=self.artifact.model_id,
            elapsed_ms=elapsed_ms,
            notes=notes,
        )


def model_provenance(
    provider: str,
    execution: ExecutionTarget,
    *,
    model_id: str = "",
    elapsed_ms: float = 0.0,
    notes: tuple[str, ...] = (),
) -> Provenance:
    """Build a :class:`Provenance` without a loaded model.

    Used by backends that compute results without a resident session (for
    example a pure-Python embedding implementation in a test).
    """
    return Provenance(
        provider=provider,
        execution=execution,
        model_id=model_id,
        elapsed_ms=elapsed_ms,
        notes=notes,
    )


@dataclass(frozen=True, slots=True)
class RuntimeProbe:
    """What a runtime reports about itself, right now."""

    capabilities: RuntimeCapabilities
    reachable: bool
    detail: str = ""

    @property
    def npu_advertised(self) -> bool:
        """An NPU provider is listed. This is *not* proof it can be used."""
        return self.capabilities.npu_advertised

    @property
    def available_kinds(self) -> tuple[AcceleratorKind, ...]:
        return self.capabilities.available_kinds()


class InferenceRuntime(Component):
    """Loads model artifacts and executes them, recording the real target."""

    kind = "runtime"

    @abstractmethod
    def probe(self) -> RuntimeProbe:
        """Report the execution providers this runtime can currently use."""

    @abstractmethod
    def load(
        self,
        artifact: ModelArtifact,
        *,
        preferences: ExecutionPreferences | None = None,
    ) -> LoadedModel:
        """Load a model and select an execution target.

        The returned target is unverified. Implementations resolve the target
        with :func:`core.execution.resolve_target` so a CPU fallback is always
        explicit and recorded.
        """

    def run(self, model: LoadedModel, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute one forward pass and return the outputs.

        Optional: runtimes that only host generation loops raise.
        """
        raise NotImplementedError(f"{self.info.name} does not expose run()")

    def unload(self, model: LoadedModel) -> None:
        """Release a loaded model. Idempotent."""
        return None

    @property
    def npu_advertised(self) -> bool:
        """Whether an NPU provider is advertised (never a measurement)."""
        return self.probe().capabilities.npu_advertised

    def preferred_capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.BENCHMARK_MEASUREMENT})
