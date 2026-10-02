"""Lightweight local AI backend: real ONNX Runtime inference, no downloads.

This is the seam that makes ``MockAIBackend`` replaceable. It implements the same
:class:`~core.contracts.ai.AIBackend` facade the application already talks to, so
selecting it is a configuration change and nothing else moves.

Three rules define what it is allowed to do:

* **No download, ever.** A model must already exist on disk and be named in
  configuration. Nothing in this module fetches anything.
* **No fabrication.** Without a configured model there is no language model:
  :attr:`LightweightLocalAIBackend.llm` is ``None`` and the UI reports
  ``not configured``. It never falls back to the mock, because a silent fallback
  would show demo output where a model was expected.
* **No unproven accelerator.** Results carry the target recorded by
  :class:`~ai.runtime.OnnxRuntimeBackend`, which only becomes ``VERIFIED`` after
  a real forward pass. An NPU that is merely advertised stays advertised.

What this backend is **not** yet: a token-level text generator. It performs a
real forward pass over a small encoder-style model and reads the output, which
is enough to classify a failure and to exercise the whole measured path on real
hardware. Sampled token-by-token generation is Phase 3 work on the target
machine, and until it exists :data:`NOTE_TOKEN_GENERATION` says so in the UI.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Sequence
from typing import Any

from core.contracts.ai import AIBackend
from core.contracts.base import (
    Capability,
    ProviderHealth,
    ProviderInfo,
    ProviderStatus,
)
from core.contracts.embeddings import EmbeddingBackend
from core.contracts.llm import ChatMessage, GenerationParams, GenerationResult, LlmBackend
from core.contracts.ocr import OcrBackend
from core.contracts.vision import VisionBackend
from core.errors import ModelArtifactError, ProviderUnavailableError
from core.execution import AcceleratorKind, ExecutionPreferences, ExecutionTarget
from core.model_artifact import ModelArtifact
from core.observability import get_logger, log_event
from core.timing import measure

from ai.runtime import OnnxRuntimeBackend, onnxruntime_available

__all__ = [
    "LightweightLocalAIBackend",
    "LocalModelLlmBackend",
    "NOTE_TOKEN_GENERATION",
    "encode_text",
]

_log = get_logger(__name__)

#: Shown in the UI so nobody reads a classification as a written explanation.
NOTE_TOKEN_GENERATION = (
    "local model: single encoder forward pass; "
    "token-by-token generation is not implemented yet"
)

#: Encoder input length. Small on purpose - this is a classification head, not a
#: language model, and a longer context would only make the demo slower.
_SEQUENCE_LENGTH = 64

#: Output label used when the model exposes no label list of its own.
_UNKNOWN_LABEL = "unlabelled"

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def encode_text(text: str, length: int = _SEQUENCE_LENGTH) -> list[list[float]]:
    """Turn text into a fixed-length numeric tensor, with no tokenizer required.

    A hashing encoder, not a language model input format. It is deterministic
    and dependency-free, which keeps the demo offline, and it is labelled as
    what it is: a way to feed *some* tensor to a real model so the measured
    execution path can be exercised. A real tokenizer replaces this on the
    target machine, and nothing above this function changes.
    """
    tokens = [word.lower() for word in _WORD_RE.findall(text)][:length] or ["<empty>"]
    return [
        [
            byte / 255.0
            for byte in hashlib.sha256(
                f"{i}:{tokens[i % len(tokens)]}".encode()
            ).digest()[:4]
        ]
        for i in range(length)
    ]


class LocalModelLlmBackend(LlmBackend):
    """Runs a real forward pass on a local ONNX model.

    Parameters
    ----------
    artifact:
        The model file. It must already exist; this class never downloads.
    runtime:
        Execution layer. Defaults to ONNX Runtime, which resolves the provider
        and records whether the result was actually verified.
    labels:
        Optional class names indexed by the output argmax. Without them the
        result is named ``class_<n>`` rather than inventing a label.
    """

    kind = "llm"

    def __init__(
        self,
        artifact: ModelArtifact,
        *,
        runtime: OnnxRuntimeBackend | None = None,
        labels: Sequence[str] = (),
    ) -> None:
        self._artifact = artifact
        self._runtime = runtime or OnnxRuntimeBackend()
        self._labels = tuple(labels)
        self._loaded: Any = None
        self._verified: Any = None
        self._calls = 0

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name=f"local-onnx-llm({self._artifact.model_id})",
            kind=self.kind,
            version=self._runtime.info.version,
            capabilities=frozenset({Capability.TEXT_GENERATION, Capability.CHAT}),
            detail=NOTE_TOKEN_GENERATION,
            requires=("onnxruntime", "a configured local model artifact"),
        )

    def health(self) -> ProviderHealth:
        """Report the real state: runtime present, artifact present, or neither."""
        if not onnxruntime_available():
            return ProviderHealth(
                ProviderStatus.UNAVAILABLE,
                "onnxruntime is not installed (optional dependency)",
            )
        try:
            self._artifact.require_exists()
        except ModelArtifactError as exc:
            return ProviderHealth(ProviderStatus.NOT_CONFIGURED, str(exc))
        return ProviderHealth(
            ProviderStatus.READY,
            f"model {self._artifact.model_id} on {self._runtime.info.version}",
        )

    @property
    def call_count(self) -> int:
        return self._calls

    @property
    def execution_target(self) -> ExecutionTarget | None:
        """The target of the last verified run, or None before any run."""
        return self._verified.target if self._verified is not None else None

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        params: GenerationParams | None = None,
    ) -> GenerationResult:
        """Run one forward pass and report the label it produced.

        Raises rather than falling back: a missing model or a missing runtime is
        a configuration problem the user needs to see, not something to paper
        over with sample text.
        """
        prompt = "\n".join(message.content for message in messages)
        loaded = self._load()
        inputs = self._inputs_for(prompt)
        with measure() as watch:
            outputs = self._runtime.run(loaded, inputs)
        # The forward pass is what turns an advertised target into a verified
        # one. It runs here so the recorded target reflects real work.
        verified = self._runtime.verify_run(loaded, inputs)
        self._verified = verified
        self._calls += 1

        label, score = self._read_output(outputs)
        log_event(
            _log,
            logging.INFO,
            "ai.local.forward_pass",
            model=self._artifact.model_id,
            target=verified.target.describe(),
            latency_ms=round(watch.milliseconds, 2),
        )
        return GenerationResult(
            text=(
                f"[LOCAL MODEL - ONNX Runtime] prediction: {label} "
                f"(score {score:.3f})\n{NOTE_TOKEN_GENERATION}."
            ),
            provenance=verified.provenance(
                watch.milliseconds, notes=("onnxruntime forward pass",)
            ),
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(label.split()),
            finish_reason="eos_classifier",
        )

    # -- internals --------------------------------------------------------
    # -- internals --------------------------------------------------------

    def _load(self) -> Any:
        """Load once and reuse, so a demo does not re-read the model per call."""
        if self._loaded is None:
            self._loaded = self._runtime.load(
                self._artifact, preferences=ExecutionPreferences.prefer_npu()
            )
        return self._loaded

    def _inputs_for(self, prompt: str) -> dict[str, Any]:
        """Build the session's input tensor, named from the session itself."""
        import numpy as np

        session = self._runtime.session_for(self._load())
        # Naming the input from the session keeps this working for any model
        # without hard-coding a tensor name.
        names = [item.name for item in session.get_inputs()] or ["input"]
        return {names[0]: np.asarray(encode_text(prompt), dtype=np.float32)}

    def _read_output(self, outputs: dict[str, Any]) -> tuple[str, float]:
        """Turn raw output tensors into a label and a score."""
        import numpy as np

        if not outputs:
            raise ProviderUnavailableError(
                "the model produced no outputs", model=self._artifact.model_id
            )
        vector = np.asarray(next(iter(outputs.values()))).reshape(-1)
        if vector.size == 0:
            raise ProviderUnavailableError("the model produced an empty output")
        index = int(np.argmax(vector))
        label = (
            self._labels[index]
            if index < len(self._labels)
            else (_UNKNOWN_LABEL if self._labels else f"class_{index}")
        )
        return label, float(vector[index])

        """Read ``["a","b"]`` from a model option, or nothing at all."""
        try:
            parsed = json.loads(options)
        except (TypeError, ValueError):
            return ()
        return tuple(str(item) for item in parsed) if isinstance(parsed, list) else ()

    def close(self) -> None:
        """Release the model. Idempotent."""
        if self._loaded is not None:
            self._runtime.unload(self._loaded)
            self._loaded = None

class LightweightLocalAIBackend(AIBackend):
    """``AIBackend`` over local ONNX artifacts, with nothing downloaded.

    Registered as ``local``. Capabilities that have no configured model are
    ``None`` rather than mocked, so a partially configured deployment shows
    exactly what it can and cannot do.
    """

    def __init__(
        self,
        *,
        runtime: OnnxRuntimeBackend | None = None,
        llm_artifact: ModelArtifact | None = None,
        llm_labels: Sequence[str] = (),
    ) -> None:
        self._runtime = runtime or OnnxRuntimeBackend()
        self._llm = (
            LocalModelLlmBackend(llm_artifact, runtime=self._runtime, labels=llm_labels)
            if llm_artifact is not None
            else None
        )

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="lightweight-local-ai-backend",
            kind=self.kind,
            version=self._runtime.info.version,
            capabilities=(
                frozenset({Capability.TEXT_GENERATION, Capability.CHAT})
                if self._llm is not None
                else frozenset()
            ),
            detail=(
                f"local models via {self._runtime.info.version}"
                if self._llm is not None
                else "no local model configured; the app runs on rules only"
            ),
            requires=("onnxruntime",),
        )

    def is_mock(self) -> bool:
        """False: this backend runs real inference, or reports that it cannot."""
        return False

    @property
    def ocr(self) -> OcrBackend | None:
        """No OCR model ships with the application."""
        return None

    @property
    def embeddings(self) -> EmbeddingBackend | None:
        """No embedding model is configured by default."""
        return None

    @property
    def vision(self) -> VisionBackend | None:
        """No vision model ships with the application."""
        return None

    @property
    def llm(self) -> LlmBackend | None:
        return self._llm

    @property
    def accelerator(self) -> AcceleratorKind:
        """Where the last real run executed, or ``UNKNOWN`` before one has."""
        target = getattr(self._llm, "execution_target", None)
        return target.kind if target is not None else AcceleratorKind.UNKNOWN

