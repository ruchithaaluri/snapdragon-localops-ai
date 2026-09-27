"""Model artifacts: what a model is, where it came from, and whether it is intact.

The application ships **no** model. Artifacts are produced later (for example
by Qualcomm AI Hub Workbench compilation on a build machine) and referenced by
path from configuration. Nothing here downloads anything and nothing invents a
model name: an artifact either exists on disk with a known format, or the
application reports that the model is missing.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from core.errors import ModelArtifactError, ModelIntegrityError

__all__ = [
    "ModelArtifact",
    "ModelFormat",
    "ModelRegistry",
    "ModelRole",
    "ModelSource",
    "sha256_file",
]

_CHUNK_SIZE = 1024 * 1024


class ModelFormat(str, Enum):
    """On-disk format of a model artifact.

    ``ONNX`` is the portable format used by the local runtime. The QNN and
    TFLite entries describe artifacts an accelerator backend may consume later;
    nothing in this repository produces or loads them.
    """

    ONNX = "onnx"
    QNN_DLC = "qnn_dlc"
    QNN_CONTEXT_BINARY = "qnn_context_binary"
    TFLITE = "tflite"
    GGUF = "gguf"
    SAFETENSORS = "safetensors"
    UNKNOWN = "unknown"


class ModelSource(str, Enum):
    """Provenance of an artifact. Recorded so results stay traceable."""

    LOCAL = "local"
    QUALCOMM_AI_HUB = "qualcomm_ai_hub"
    HUGGING_FACE = "hugging_face"
    OTHER = "other"


class ModelRole(str, Enum):
    """The job a model performs in the pipeline."""

    OCR = "ocr"
    EMBEDDINGS = "embeddings"
    VISION = "vision"
    LLM = "llm"


_SUFFIX_FORMATS: Mapping[str, ModelFormat] = {
    ".onnx": ModelFormat.ONNX,
    ".dlc": ModelFormat.QNN_DLC,
    ".tflite": ModelFormat.TFLITE,
    ".litert": ModelFormat.TFLITE,
    ".gguf": ModelFormat.GGUF,
    ".safetensors": ModelFormat.SAFETENSORS,
}


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """A single model file, with enough metadata to verify and trace it."""

    model_id: str
    path: Path
    format: ModelFormat = ModelFormat.UNKNOWN
    source: ModelSource = ModelSource.LOCAL
    sha256: str = ""
    size_bytes: int = 0
    quantization: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ModelArtifactError("model artifact requires a model_id")
        if not isinstance(self.path, Path):
            object.__setattr__(self, "path", Path(self.path))

    @classmethod
    def from_path(
        cls,
        model_id: str,
        path: str | Path,
        *,
        format: ModelFormat | None = None,  # noqa: A002 - mirrors the field name
        source: ModelSource = ModelSource.LOCAL,
        sha256: str = "",
        size_bytes: int = 0,
        quantization: str = "",
        metadata: Mapping[str, str] | None = None,
    ) -> "ModelArtifact":
        """Build an artifact, inferring the format from the file suffix."""
        resolved = Path(path).expanduser()
        inferred = format or _SUFFIX_FORMATS.get(resolved.suffix.lower(), ModelFormat.UNKNOWN)
        return cls(
            model_id=model_id,
            path=resolved,
            format=inferred,
            source=source,
            sha256=sha256,
            size_bytes=size_bytes,
            quantization=quantization,
            metadata=dict(metadata or {}),
        )

    def exists(self) -> bool:
        """True when the file is present. Does not imply it is loadable."""
        return self.path.is_file()

    def require_exists(self) -> Path:
        """Return the path or raise :class:`ModelArtifactError`."""
        if not self.exists():
            raise ModelArtifactError(
                "model artifact not found on disk",
                model_id=self.model_id,
                path=str(self.path),
            )
        return self.path

    def compute_sha256(self) -> str:
        """Digest the artifact now (used when the config omits a checksum)."""
        return sha256_file(self.require_exists())

    def verify_integrity(self) -> str:
        """Check the recorded checksum; raise on mismatch.

        An artifact with no recorded checksum passes: an *unverified* artifact
        is tolerated, a *wrong* one is not.
        """
        self.require_exists()
        if not self.sha256:
            return ""
        actual = self.compute_sha256()
        if actual.lower() != self.sha256.lower():
            raise ModelIntegrityError(
                "model artifact checksum mismatch",
                model_id=self.model_id,
                expected=self.sha256,
                actual=actual,
            )
        return actual

    def describe(self) -> str:
        """Short, log-safe description. Never includes file contents."""
        return f"{self.model_id} [{self.format.value}] from {self.path}"


class ModelRegistry:
    """Maps a :class:`ModelRole` onto the artifact configured for it.

    Empty by design: Phase 1 registers nothing, so every model-backed feature
    reports "no model configured" instead of guessing.
    """

    def __init__(self) -> None:
        self._by_role: dict[ModelRole, ModelArtifact] = {}

    def register(self, role: ModelRole, artifact: ModelArtifact) -> None:
        """Bind a role to an artifact, replacing any previous binding."""
        self._by_role[role] = artifact

    def get(self, role: ModelRole) -> ModelArtifact | None:
        """Return the artifact for a role, or None when unconfigured."""
        return self._by_role.get(role)

    def require(self, role: ModelRole) -> ModelArtifact:
        """Return the artifact for a role or raise a clear error."""
        artifact = self.get(role)
        if artifact is None:
            raise ModelArtifactError(
                "no model configured for this role",
                role=role.value,
                hint="set the corresponding model path in configuration",
            )
        return artifact

    def is_configured(self, role: ModelRole) -> bool:
        """True when a role has an artifact assigned."""
        return role in self._by_role

    def configured_roles(self) -> tuple[ModelRole, ...]:
        """Roles that have an artifact, in declaration order."""
        return tuple(self._by_role)

    def all(self) -> tuple[ModelArtifact, ...]:
        return tuple(self._by_role.values())

    def __len__(self) -> int:
        return len(self._by_role)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        roles = ", ".join(f"{r.value}={a.model_id}" for r, a in self._by_role.items()) or "empty"
        return f"ModelRegistry({roles})"

def sha256_file(path: Path) -> str:
    """Stream a SHA-256 digest of a file, so large models stay off the heap."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()
