"""ONNX Runtime execution backend (optional, not installed in Phase 1).

Imports cleanly even when ``onnxruntime`` is absent, so the application and the
test suite start on any machine. The dependency is touched only inside the
probe/load methods.

Execution-provider mapping is intentionally conservative:

* ``CPUExecutionProvider`` -> ``AcceleratorKind.CPU``
* a CUDA / DirectML provider -> ``AcceleratorKind.GPU``
* a provider whose name mentions QNN/Hexagon/HTP -> ``AcceleratorKind.NPU``
  **advertised only**: it becomes verified solely after a real forward pass is
  recorded by :meth:`OnnxRuntimeBackend.verify_run`.

On the Snapdragon target this module is either extended with a vendor execution
provider or superseded by a dedicated backend registered beside the mock. No
application code changes either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.contracts.base import Capability, ProviderHealth, ProviderInfo, ProviderStatus
from core.contracts.inference import InferenceRuntime, LoadedModel, RuntimeProbe
from core.errors import ProviderUnavailableError
from core.execution import (
    AcceleratorKind,
    ExecutionPreferences,
    ExecutionTarget,
    ExecutionVerification,
    RuntimeCapabilities,
    RuntimeCapability,
    resolve_target,
)
from core.model_artifact import ModelArtifact
from core.observability import get_logger, log_event
from core.timing import measure

__all__ = [
    "OnnxRuntimeBackend",
    "classify_execution_provider",
    "onnxruntime_available",
]

_log = get_logger(__name__)

_CPU_PROVIDER = "CPUExecutionProvider"
_NPU_HINTS = ("qnn", "hexagon", "htp", "npu", "snapdragon")
_GPU_HINTS = ("cuda", "dml", "directml", "tensorrt", "rocm", "openvino")


def onnxruntime_available() -> bool:
    """True when the optional ``onnxruntime`` package can be imported."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def classify_execution_provider(name: str) -> AcceleratorKind:
    """Map an ONNX Runtime provider name onto an accelerator kind.

    NPU is recognised by name only, and recognition is not proof: the result
    stays ``ADVERTISED`` until a real inference confirms it.
    """
    lowered = name.lower()
    if lowered == _CPU_PROVIDER.lower():
        return AcceleratorKind.CPU
    if any(hint in lowered for hint in _NPU_HINTS):
        return AcceleratorKind.NPU
    if any(hint in lowered for hint in _GPU_HINTS):
        return AcceleratorKind.GPU
    return AcceleratorKind.UNKNOWN


@dataclass(frozen=True, slots=True)
class _Probe:
    """Internal probe result: reachable, version, provider names."""

    reachable: bool
    version: str
    providers: tuple[str, ...]
    detail: str = ""


class OnnxRuntimeBackend(InferenceRuntime):
    """Loads ONNX models and runs them on whatever providers are available."""

    def __init__(self, *, providers: tuple[str, ...] = ()) -> None:
        self._requested = providers
        self._module: Any = None

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="onnxruntime",
            kind=self.kind,
            version=self._version(),
            capabilities=frozenset({Capability.BENCHMARK_MEASUREMENT}),
            detail="Portable ONNX Runtime execution (optional dependency).",
        )

    def _version(self) -> str:
        if self._module is not None:
            return str(getattr(self._module, "__version__", "unknown"))
        try:
            import onnxruntime
        except ImportError:
            return "not installed"
        return str(onnxruntime.__version__)

    def _probe(self) -> _Probe:
        try:
            import onnxruntime
        except ImportError as exc:
            return _Probe(False, "not installed", (), f"optional dependency missing: {exc}")
        self._module = onnxruntime
        try:
            available = tuple(onnxruntime.get_available_providers())
        except Exception as exc:  # pragma: no cover - depends on the installed build
            return _Probe(False, "unknown", (), f"provider query failed: {exc}")
        return _Probe(True, str(onnxruntime.__version__), available)

    def health(self) -> ProviderHealth:
        probe = self._probe()
        if not probe.reachable:
            return ProviderHealth(
                ProviderStatus.UNAVAILABLE,
                probe.detail or "onnxruntime is not installed (optional dependency)",
            )
        return ProviderHealth(ProviderStatus.READY, f"onnxruntime {probe.version}")

    def probe(self) -> RuntimeProbe:
        """Report available execution providers. Never a measurement."""
        probe = self._probe()
        capabilities = tuple(
            RuntimeCapability(
                name=name,
                kind=classify_execution_provider(name),
                available=True,
                version=probe.version,
                detail="advertised by onnxruntime; not yet verified by a run",
            )
            for name in probe.providers
        )
        return RuntimeProbe(
            capabilities=RuntimeCapabilities(
                runtime="onnxruntime",
                runtime_version=probe.version,
                capabilities=capabilities,
            ),
            reachable=probe.reachable,
            detail=probe.detail,
        )


    def load(
        self,
        artifact: ModelArtifact,
        *,
        preferences: ExecutionPreferences | None = None,
    ) -> LoadedModel:
        """Create an inference session and resolve the execution target.

        The returned target stays unverified until :meth:`verify_run` records a
        real forward pass.
        """
        probe = self._probe()
        if not probe.reachable:
            raise ProviderUnavailableError(
                "onnxruntime is not installed; install requirements/ai.txt or "
                "configure a different backend",
                detail=probe.detail,
            )
        artifact.verify_integrity()
        path = artifact.require_exists()
        prefs = preferences or ExecutionPreferences.cpu_only()
        with measure() as watch:
            session = self._module.InferenceSession(
                str(path), providers=list(self._requested) or None
            )
        target = self._target_for(session)
        log_event(
            _log,
            20,
            "runtime.model.loaded",
            model=artifact.model_id,
            target=target.describe(),
            load_ms=round(watch.milliseconds, 2),
        )
        return LoadedModel(
            artifact=artifact,
            target=target,
            session_id=str(id(session)),
            load_ms=watch.milliseconds,
            details={"providers": ",".join(session.get_providers())},
        )

    def _target_for(self, session: Any) -> ExecutionTarget:
        """Resolve the target for a freshly created session.

        Selection goes through :func:`core.execution.resolve_target`, so an
        advertised-but-unproven NPU is never chosen and a CPU run is recorded
        as an explicit fallback.
        """
        available: list[ExecutionTarget] = []
        for name in session.get_providers():
            kind = classify_execution_provider(name)
            if kind is AcceleratorKind.CPU:
                available.append(ExecutionTarget.cpu(name, detail="session created"))
            elif kind is AcceleratorKind.NPU:
                available.append(
                    ExecutionTarget.npu_advertised(name, detail="session created")
                )
            else:
                available.append(
                    ExecutionTarget(
                        kind=kind,
                        provider=name,
                        verification=ExecutionVerification.advertised(
                            detail="session created"
                        ),
                    )
                )
        decision = resolve_target(available, (AcceleratorKind.CPU,), provider="onnxruntime")
        return decision.target

    def run(self, model: LoadedModel, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute a forward pass. Output tensors are returned as plain lists."""
        session = self._session_for(model)
        outputs = session.run(None, inputs)
        return {
            name: output.tolist() if hasattr(output, "tolist") else output
            for name, output in zip(session.get_outputs(), outputs, strict=False)
        }

    def verify_run(self, model: LoadedModel, inputs: dict[str, Any]) -> LoadedModel:
        """Run once and return a model copy whose target is verified.

        This is the only path that promotes a target to ``VERIFIED``, and it
        requires the forward pass to have completed and produced outputs.
        """
        session = self._session_for(model)
        with measure() as watch:
            outputs = session.run(None, inputs)
        if not outputs:
            raise ProviderUnavailableError("inference produced no outputs")
        verified = model.mark_verified(
            "onnxruntime-forward-pass",
            f"{len(outputs)} output tensor(s) in {watch.milliseconds:.2f} ms",
        )
        log_event(
            _log,
            20,
            "runtime.execution.verified",
            model=model.model_id,
            target=verified.target.describe(),
        )
        return verified

    def _session_for(self, model: LoadedModel) -> Any:
        """Fetch the session for a loaded model, or fail loudly."""
        if self._module is None:
            raise ProviderUnavailableError("onnxruntime is not loaded")
        model.artifact.require_exists()
        return self._module.InferenceSession(str(model.artifact.path))

