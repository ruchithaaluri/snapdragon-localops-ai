"""Snapdragon backend: the NPU path, structurally present, **not validated**.

Read this before assuming anything works here.

**Status: "Snapdragon deployment validation pending target hardware."**

Nothing in this module has been executed on a Snapdragon device. It was written
and imported on Windows x64, where it correctly reports itself unavailable. The
classes below exist so that the application never depends on a
Snapdragon-specific API, and so that swapping to the real thing is a
configuration change. They are **not** evidence that NPU execution works.

What is real:

* :func:`is_snapdragon_host` genuinely probes the machine, and returns ``False``
  on x64. It is a runtime check, not a hard-coded answer.
* Every capability returns ``None`` and every work method raises
  :class:`~core.errors.ProviderUnavailableError` unless a vendor runtime is
  present *and* a measured inference has been recorded.
* The device record it produces is used verbatim. Nothing invents a
  Hexagon/HTP number.

What is **not** real, and must not be presented as real:

* No Qualcomm SDK, execution provider or compiled model is present.
* No latency, memory or NPU figure has been produced by this code.
* :meth:`SnapdragonRuntime.probe` lists nothing, because nothing is installed.

To make this operational on the HP Snapdragon target, see
``docs/SNAPDRAGON_TARGET.md``: install the Qualcomm runtime on the target, point
configuration at a QNN/HTP-compiled artifact, then run the benchmark harness so
a real forward pass can promote the target to ``VERIFIED``. Until that has
happened on real hardware, the UI shows ``CPU`` and the README says validation
is pending.
"""

from __future__ import annotations

import logging
import platform
from typing import Any

from core.contracts.ai import AIBackend
from core.contracts.base import Capability, ProviderHealth, ProviderInfo, ProviderStatus
from core.contracts.embeddings import EmbeddingBackend
from core.contracts.inference import InferenceRuntime, LoadedModel, RuntimeProbe
from core.contracts.llm import LlmBackend
from core.contracts.ocr import OcrBackend
from core.contracts.vision import VisionBackend
from core.errors import ExecutionBackendUnavailableError
from core.execution import (
    AcceleratorKind,
    ExecutionPreferences,
    ExecutionTarget,
    RuntimeCapabilities,
    RuntimeCapability,
)
from core.model_artifact import ModelArtifact
from core.observability import get_logger, log_event

__all__ = [
    "SNAPDRAGON_VALIDATION_STATUS",
    "SnapdragonAIBackend",
    "SnapdragonRuntime",
    "is_snapdragon_host",
]

_log = get_logger(__name__)

#: The single sentence this project uses everywhere to describe the NPU path.
SNAPDRAGON_VALIDATION_STATUS = (
    "Snapdragon deployment validation pending target hardware."
)

#: Runtime package that would supply a vendor execution provider. Absent here,
#: and deliberately not installed on the x64 development machine.
_RUNTIME_PACKAGE = "qnn"

#: Machine-architecture fragments that identify a Snapdragon host.
_SNAPDRAGON_HINTS = ("snapdragon", "qualcomm", "hexagon", "htp")


def is_snapdragon_host() -> bool:
    """Probe this machine for Snapdragon hardware. A check, never an assumption.

    Returns ``True`` only when the platform, the machine and the processor all
    point at Snapdragon/Qualcomm. It says nothing about whether a runtime is
    installed or whether a model would actually execute.
    """
    haystack = " ".join(
        (
            platform.system(),
            platform.machine(),
            platform.processor(),
            platform.uname().processor,
        )
    ).lower()
    return any(hint in haystack for hint in _SNAPDRAGON_HINTS)


class SnapdragonRuntime(InferenceRuntime):
    """The vendor execution layer. Placeholder until the target is available.

    It probes for the Qualcomm runtime and reports exactly what it finds. When
    the runtime is absent - which is the case on the development machine - the
    probe is simply empty and every load raises. That is the honest shape of an
    unvalidated backend: usable in structure, useless in fact, and impossible to
    mistake for a working NPU path.
    """

    kind = "runtime"

    def __init__(self, *, package: str = _RUNTIME_PACKAGE) -> None:
        self._package = package
        self._module: Any = None

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="snapdragon-runtime",
            kind=self.kind,
            version=self._version(),
            capabilities=frozenset({Capability.BENCHMARK_MEASUREMENT}),
            detail=SNAPDRAGON_VALIDATION_STATUS,
            requires=(f"{self._package} (Qualcomm runtime, target hardware only)",),
        )

    def _version(self) -> str:
        if self._module is not None:
            return str(getattr(self._module, "__version__", "unknown"))
        return "not installed"

    def _probe_module(self) -> Any:
        """Import the vendor runtime if it is present; never install it."""
        if self._module is not None:
            return self._module
        import importlib

        try:
            self._module = importlib.import_module(self._package)
        except ImportError:
            return None
        return self._module

    def health(self) -> ProviderHealth:
        if not is_snapdragon_host():
            return ProviderHealth(
                ProviderStatus.UNAVAILABLE,
                f"not a Snapdragon host ({platform.machine()}); "
                f"{SNAPDRAGON_VALIDATION_STATUS}",
            )
        if self._probe_module() is None:
            return ProviderHealth(
                ProviderStatus.NOT_CONFIGURED,
                f"Qualcomm runtime '{self._package}' is not installed on this host",
            )
        return ProviderHealth(
            ProviderStatus.DEGRADED,
            f"runtime '{self._package}' found, but no measured NPU inference has "
            f"been recorded yet. {SNAPDRAGON_VALIDATION_STATUS}",
        )

    def probe(self) -> RuntimeProbe:
        """Report installed execution providers. Empty here, by design."""
        module = self._probe_module()
        if module is None:
            return RuntimeProbe(
                capabilities=RuntimeCapabilities(
                    runtime="snapdragon",
                    runtime_version="not installed",
                    notes=(SNAPDRAGON_VALIDATION_STATUS,),
                ),
                reachable=False,
                detail=f"'{self._package}' is not installed; nothing was measured",
            )
        return RuntimeProbe(
            capabilities=RuntimeCapabilities(
                runtime="snapdragon",
                runtime_version=str(getattr(module, "__version__", "unknown")),
                capabilities=(
                    RuntimeCapability(
                        name="QNNExecutionProvider",
                        kind=AcceleratorKind.NPU,
                        available=True,
                        detail="advertised by the runtime; not verified by a run",
                    ),
                ),
                notes=(SNAPDRAGON_VALIDATION_STATUS,),
            ),
            reachable=True,
            detail="runtime present; run a real inference to verify a target",
        )

    def load(
        self,
        artifact: ModelArtifact,
        *,
        preferences: ExecutionPreferences | None = None,
    ) -> LoadedModel:
        """Refuse to load until a vendor runtime and a real target exist."""
        raise ExecutionBackendUnavailableError(
            f"{SNAPDRAGON_VALIDATION_STATUS} The Qualcomm runtime "
            f"'{self._package}' must be installed on the target hardware "
            f"before a model can be loaded.",
            model=artifact.model_id,
            host=f"{platform.system()} {platform.machine()}",
            is_snapdragon_host=is_snapdragon_host(),
        )


class SnapdragonAIBackend(AIBackend):
    """``AIBackend`` for the Snapdragon NPU target. **Not yet validated.**

    Registered as ``snapdragon``. On the x64 development machine it constructs
    cleanly, reports ``unavailable``, and exposes no capabilities - which is the
    correct behaviour, not a defect.

    The single method that must never lie is :meth:`is_validated`. It returns
    ``False`` until a real inference has been recorded on real Snapdragon
    hardware, and the UI keys its ``NPU`` indicator off it rather than off the
    backend's mere existence.
    """

    def __init__(self, *, runtime: SnapdragonRuntime | None = None) -> None:
        self._runtime = runtime or SnapdragonRuntime()
        self._validated = False

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="snapdragon-ai-backend",
            kind=self.kind,
            version=self._runtime.info.version,
            capabilities=frozenset(),
            detail=SNAPDRAGON_VALIDATION_STATUS,
            requires=("Snapdragon/Windows on ARM hardware", "Qualcomm runtime"),
        )

    def is_mock(self) -> bool:
        """False: it is a real backend, even though it is not yet operational."""
        return False

    def is_validated(self) -> bool:
        """True only after a measured NPU run on real target hardware.

        This is the single flag the ``NPU`` status indicator is allowed to read.
        It is set by :meth:`record_verified_run` and is ``False`` everywhere
        else, including on the Snapdragon host before anything has been measured.
        """
        return self._validated

    def record_verified_run(self, target: ExecutionTarget) -> None:
        """Promote this backend to validated, given a *verified* NPU target.

        Passing a non-NPU or an unverified target is refused: this method is
        the only door to the ``NPU`` indicator, so it must not be a formality.
        """
        target.assert_npu_claim_allowed()
        self._validated = True
        log_event(
            _log,
            logging.INFO,
            "snapdragon.validated",
            target=target.describe(),
        )

    @property
    def health_status(self) -> ProviderHealth:
        return self._runtime.health()

    def health(self) -> ProviderHealth:
        status = self._runtime.health()
        return ProviderHealth(
            status.status,
            f"{status.detail} | validated={self.is_validated()}",
        )

    @property
    def ocr(self) -> OcrBackend | None:
        return None

    @property
    def embeddings(self) -> EmbeddingBackend | None:
        return None

    @property
    def vision(self) -> VisionBackend | None:
        return None

    @property
    def llm(self) -> LlmBackend | None:
        return None

    @property
    def runtime(self) -> SnapdragonRuntime:
        """The vendor execution layer, for the benchmark harness to probe."""
        return self._runtime
