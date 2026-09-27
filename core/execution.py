"""Execution targets and the verification rules around accelerator claims.

This module encodes the project's central honesty rule:

    NPU execution is reported **only** when a real inference actually executed
    on the NPU and this application verified that execution.

Everything else - an accelerator being listed by an engine, a package being
installed, a model being compiled for an NPU, a device name containing
"Snapdragon" - is *not* evidence. Those cases are recorded as ``ADVERTISED``
(never claimable) or rejected with
:class:`~core.errors.UnverifiedNpuClaimError`.

Everything here is hardware-agnostic: ``AcceleratorKind.NPU`` is just an enum
value. Nothing in this module depends on ARM64, a vendor SDK, or a Snapdragon
machine. A Snapdragon backend fills in the same fields later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

from core.errors import ExecutionBackendUnavailableError, UnverifiedNpuClaimError
from core.ids import utc_now

__all__ = [
    "AcceleratorKind",
    "ExecutionPreferences",
    "ExecutionTarget",
    "ExecutionVerification",
    "ExecutionVerificationState",
    "FallbackDecision",
    "FallbackPolicy",
    "RuntimeCapabilities",
    "RuntimeCapability",
    "TargetStatus",
    "require_npu_claim_allowed",
    "resolve_target",
]


class AcceleratorKind(str, Enum):
    """Where work was executed. Values are stable strings for logs and UI."""

    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"
    UNKNOWN = "unknown"


class ExecutionVerificationState(str, Enum):
    """How strongly an execution target is backed by real measurement."""

    #: A real inference ran here and the result was checked.
    VERIFIED = "verified"
    #: The backend advertises this target, but nothing has been executed yet.
    ADVERTISED = "advertised"
    #: A probe or run was attempted and failed.
    FAILED = "failed"
    #: Nothing is known (the default, and the only safe starting point).
    UNKNOWN = "unknown"
    #: This target was never a candidate (e.g. a CPU-only build).
    NOT_APPLICABLE = "not_applicable"


class TargetStatus(str, Enum):
    """User-facing summary of an execution target."""

    VERIFIED_NPU = "verified_npu"
    ADVERTISED_NPU = "advertised_npu"
    VERIFIED_GPU = "verified_gpu"
    ADVERTISED_GPU = "advertised_gpu"
    VERIFIED_CPU = "verified_cpu"
    FALLBACK_CPU = "fallback_cpu"
    FAILED = "failed"
    UNKNOWN = "unknown"

    @property
    def is_claimable(self) -> bool:
        """True only when the status may be presented as a measured result."""
        return self in {
            TargetStatus.VERIFIED_NPU,
            TargetStatus.VERIFIED_GPU,
            TargetStatus.VERIFIED_CPU,
            TargetStatus.FALLBACK_CPU,
        }


class FallbackPolicy(str, Enum):
    """What to do when the preferred accelerator cannot be used."""

    #: Run on CPU and record that an explicit fallback happened.
    ALLOW = "allow"
    #: Refuse to run. Used for benchmarks, where a silent CPU run in an
    #: "NPU" row would corrupt the result.
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class ExecutionVerification:
    """Evidence backing (or lacking) an execution claim.

    ``method`` names *how* it was verified, e.g. ``"timed-run"`` or
    ``"provider-probe"``; ``detail`` carries the measured values. A
    verification without a method is not a verification, which is enforced in
    :meth:`__post_init__`.
    """

    state: ExecutionVerificationState = ExecutionVerificationState.UNKNOWN
    method: str = ""
    detail: str = ""
    verified_at: str = ""

    @classmethod
    def verified(cls, method: str, detail: str = "") -> "ExecutionVerification":
        """Record a target that a real run confirmed."""
        if not method:
            raise ValueError("a verified execution target must name its method")
        return cls(
            state=ExecutionVerificationState.VERIFIED,
            method=method,
            detail=detail,
            verified_at=utc_now().isoformat(timespec="seconds"),
        )

    @classmethod
    def advertised(
        cls,
        method: str = "engine-advertisement",
        detail: str = "",
    ) -> "ExecutionVerification":
        """Record a target the engine claims to support but that was not run."""
        return cls(
            state=ExecutionVerificationState.ADVERTISED,
            method=method,
            detail=detail,
            verified_at=utc_now().isoformat(timespec="seconds"),
        )

    @classmethod
    def failed(cls, method: str, detail: str = "") -> "ExecutionVerification":
        """Record a probe or run that did not succeed."""
        return cls(
            state=ExecutionVerificationState.FAILED,
            method=method,
            detail=detail,
            verified_at=utc_now().isoformat(timespec="seconds"),
        )

    @classmethod
    def unknown(cls, detail: str = "") -> "ExecutionVerification":
        """Default state: nothing has been measured."""
        return cls(state=ExecutionVerificationState.UNKNOWN, detail=detail)

    @property
    def is_verified(self) -> bool:
        """True only for a verification produced by a real run."""
        return self.state is ExecutionVerificationState.VERIFIED and bool(self.method)

    def __post_init__(self) -> None:
        if self.state is ExecutionVerificationState.VERIFIED and not self.method:
            raise ValueError("a VERIFIED execution must name its verification method")


@dataclass(frozen=True, slots=True)
class ExecutionTarget:
    """The single place that records *where* work ran and how that was proven.

    Instances are immutable and travel with every result object, so any report
    or benchmark row can be traced back to the hardware that produced it.
    """

    kind: AcceleratorKind
    provider: str
    device: str = ""
    runtime_version: str = ""
    verification: ExecutionVerification = field(default_factory=ExecutionVerification.unknown)

    @classmethod
    def cpu(
        cls,
        provider: str,
        *,
        verified: bool = True,
        device: str = "",
        runtime_version: str = "",
        detail: str = "",
    ) -> "ExecutionTarget":
        """Build a CPU target. CPU counts as verified whenever work ran on it."""
        verification = (
            ExecutionVerification.verified("execution-on-cpu", detail)
            if verified
            else ExecutionVerification.advertised(detail=detail)
        )
        return cls(
            kind=AcceleratorKind.CPU,
            provider=provider,
            device=device,
            runtime_version=runtime_version,
            verification=verification,
        )

    @classmethod
    def npu_advertised(
        cls,
        provider: str,
        *,
        device: str = "",
        detail: str = "",
    ) -> "ExecutionTarget":
        """Build an NPU target the engine claims to support but that is unproven.

        The result is deliberately non-verified, so the UI and the benchmark
        reporters cannot present it as a measurement.
        """
        return cls(
            kind=AcceleratorKind.NPU,
            provider=provider,
            device=device,
            verification=ExecutionVerification.advertised(detail=detail),
        )

    @property
    def is_verified(self) -> bool:
        """True when a real run confirmed this target."""
        return self.verification.is_verified

    @property
    def npu_execution_claimed(self) -> bool:
        """True only when NPU execution is both asserted *and* verified."""
        return self.kind is AcceleratorKind.NPU and self.is_verified

    @property
    def status(self) -> TargetStatus:
        """Map the target onto a displayable status."""
        if self.verification.state is ExecutionVerificationState.FAILED:
            return TargetStatus.FAILED
        if self.kind is AcceleratorKind.NPU:
            return TargetStatus.VERIFIED_NPU if self.is_verified else TargetStatus.ADVERTISED_NPU
        if self.kind is AcceleratorKind.GPU:
            return TargetStatus.VERIFIED_GPU if self.is_verified else TargetStatus.ADVERTISED_GPU
        if self.kind is AcceleratorKind.CPU:
            if not self.is_verified:
                return TargetStatus.UNKNOWN
            if self.verification.method == "fallback-from-unsupported-target":
                return TargetStatus.FALLBACK_CPU
            return TargetStatus.VERIFIED_CPU
        return TargetStatus.UNKNOWN

    def describe(self) -> str:
        """One-line description, safe to print anywhere."""
        state = self.verification.state.value
        method = self.verification.method or "no evidence"
        return f"{self.kind.value} via {self.provider} ({state}: {method})"

    def assert_npu_claim_allowed(self) -> None:
        """Raise unless this target is a *verified* NPU run.

        Every code path that would surface an NPU claim to the user or to a
        benchmark report must pass through this method.
        """
        if self.kind is not AcceleratorKind.NPU:
            raise UnverifiedNpuClaimError(
                "target is not an NPU target",
                kind=self.kind.value,
                provider=self.provider,
            )
        if not self.is_verified:
            raise UnverifiedNpuClaimError(
                "refusing to report NPU execution: no verified inference ran on the NPU",
                provider=self.provider,
                state=self.verification.state.value,
            )

    def mark_fallback(self, reason: str) -> "ExecutionTarget":
        """Return a CPU target that records *why* a preferred target was skipped."""
        return ExecutionTarget(
            kind=AcceleratorKind.CPU,
            provider=self.provider,
            device=self.device,
            runtime_version=self.runtime_version,
            verification=ExecutionVerification.verified(
                "fallback-from-unsupported-target", reason
            ),
        )


def require_npu_claim_allowed(targets: Sequence[ExecutionTarget]) -> None:
    """Assert that no target claims unverified NPU execution.

    Used by report writers and benchmark guards as a last line of defence.
    """
    for target in targets:
        if target.kind is AcceleratorKind.NPU and not target.is_verified:
            raise UnverifiedNpuClaimError(
                "unverified NPU claim present in results",
                provider=target.provider,
                state=target.verification.state.value,
            )


@dataclass(frozen=True, slots=True)
class FallbackDecision:
    """Outcome of resolving preferred targets against what is available."""

    target: ExecutionTarget
    fell_back: bool
    reason: str
    policy: FallbackPolicy


def resolve_target(
    available: Sequence[ExecutionTarget],
    preferences: Sequence[AcceleratorKind],
    *,
    provider: str,
    policy: FallbackPolicy = FallbackPolicy.ALLOW,
) -> FallbackDecision:
    """Pick an execution target, making any CPU fallback explicit.

    Selection order: the first *verified* target matching the caller's
    preference order, then - only if the policy allows it - a verified CPU run.

    A merely advertised target (an NPU the engine lists but that we have never
    run on) is never selected, because selecting it would produce a result
    claiming hardware that was never used. Its presence is reported in
    :attr:`FallbackDecision.reason` instead.
    """
    usable = [t for t in available if t.is_verified and t.kind is not AcceleratorKind.UNKNOWN]
    for wanted in preferences:
        for target in usable:
            if target.kind is wanted:
                return FallbackDecision(target=target, fell_back=False, reason="", policy=policy)

    unproven = sorted({t.kind.value for t in available if not t.is_verified})
    reason = f"no verified target matched {[k.value for k in preferences]}"
    if unproven:
        reason += f"; unproven candidates: {unproven}"

    cpu = next((t for t in usable if t.kind is AcceleratorKind.CPU), None)
    if cpu is None:
        raise ExecutionBackendUnavailableError(
            "no verified execution target available",
            provider=provider,
            available=[t.kind.value for t in available],
        )
    if policy is FallbackPolicy.DENY and preferences:
        raise ExecutionBackendUnavailableError(
            "fallback to CPU denied by policy",
            provider=provider,
            requested=[k.value for k in preferences],
            reason=reason,
        )
    return FallbackDecision(
        target=cpu.mark_fallback(reason),
        fell_back=bool(preferences),
        reason=reason,
        policy=policy,
    )


@dataclass(frozen=True, slots=True)
class RuntimeCapability:
    """One execution provider (or device) an installed runtime can report."""

    name: str
    kind: AcceleratorKind
    available: bool
    detail: str = ""
    version: str = ""
    device: str = ""


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    """Snapshot of what the local runtime stack can actually offer.

    ``npu_advertised`` is deliberately *not* a shortcut for "NPU works". It
    mirrors the runtime's own advertisement; measured support only ever comes
    from :class:`ExecutionVerification`.
    """

    runtime: str
    runtime_version: str
    capabilities: tuple[RuntimeCapability, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def by_kind(self) -> Mapping[AcceleratorKind, RuntimeCapability]:
        return {c.kind: c for c in self.capabilities}

    def get(self, kind: AcceleratorKind) -> RuntimeCapability | None:
        """Return the capability entry for a kind, if the runtime reports one."""
        return self.by_kind.get(kind)

    def available_kinds(self) -> tuple[AcceleratorKind, ...]:
        """Kinds the runtime reports as available (advertised, not measured)."""
        return tuple(c.kind for c in self.capabilities if c.available)

    @property
    def npu_advertised(self) -> bool:
        """True when a runtime advertises an NPU provider.

        A hint for diagnostics only. It must never be rendered as "running on
        the NPU".
        """
        cap = self.get(AcceleratorKind.NPU)
        return bool(cap and cap.available)

    def as_rows(self) -> tuple[tuple[str, str, str, str], ...]:
        """Flatten to displayable rows: (kind, name, available, detail)."""
        return tuple(
            (c.kind.value, c.name, "yes" if c.available else "no", c.detail)
            for c in self.capabilities
        )




@dataclass(frozen=True, slots=True)
class ExecutionPreferences:
    """What a caller would like to run on, and how strict to be about it.

    This is a *request*, not a decision. :func:`resolve_target` turns it into a
    :class:`FallbackDecision` against the targets that are actually available.
    """

    preferred: tuple[AcceleratorKind, ...] = (AcceleratorKind.CPU,)
    policy: FallbackPolicy = FallbackPolicy.ALLOW
    require_verified_npu: bool = True
    purpose: str = "inference"

    @classmethod
    def cpu_only(cls, purpose: str = "inference") -> "ExecutionPreferences":
        """Prefer CPU only, e.g. for a deterministic reference measurement."""
        return cls(preferred=(AcceleratorKind.CPU,), purpose=purpose)

    @classmethod
    def prefer_npu(cls, purpose: str = "inference") -> "ExecutionPreferences":
        """Prefer the NPU, allowing a recorded CPU fallback.

        ``require_verified_npu`` stays true: a preference can never be turned
        into a claim without a verified run.
        """
        return cls(
            preferred=(AcceleratorKind.NPU, AcceleratorKind.GPU, AcceleratorKind.CPU),
            require_verified_npu=True,
            purpose=purpose,
        )

    def __post_init__(self) -> None:
        if not self.preferred:
            raise ValueError("at least one preferred target is required")
        if not self.require_verified_npu and AcceleratorKind.NPU in self.preferred:
            raise ValueError(
                "an NPU preference requires verification: NPU execution is "
                "never claimed without a measured run"
            )

