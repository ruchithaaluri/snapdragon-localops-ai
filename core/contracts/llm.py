"""LLM contract: the local reasoning step.

Local-first means a model runs on the user's machine. A cloud model is
*allowed* as an explicitly configured option - the configuration stores only the
name of an environment variable holding the key, never the key itself - but
nothing in this interface requires one, and no implementation may fabricate a
completion when a backend is missing.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum

from core.contracts.base import Capability, Component, Provenance

__all__ = [
    "ChatMessage",
    "ChatRole",
    "GenerationParams",
    "GenerationResult",
    "LlmBackend",
]


class ChatRole(str, Enum):
    """Speaker of a message. Mirrors common chat formats."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One message in a conversation."""

    role: ChatRole
    content: str

    def __post_init__(self) -> None:
        if self.role is not ChatRole.ASSISTANT and not self.content.strip():
            raise ValueError("a message must carry content")

    def to_dict(self) -> dict[str, str]:
        """Render as a plain mapping, ready for a template or a chat format."""
        return {"role": self.role.value, "content": self.content}

    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        return cls(ChatRole.SYSTEM, content)

    @classmethod
    def user(cls, content: str) -> "ChatMessage":
        return cls(ChatRole.USER, content)

    @classmethod
    def assistant(cls, content: str) -> "ChatMessage":
        return cls(ChatRole.ASSISTANT, content)


@dataclass(frozen=True, slots=True)
class GenerationParams:
    """Sampling parameters.

    ``temperature`` defaults to 0.0: a diagnostic assistant should be
    reproducible, not creative, unless a user deliberately raises it.
    """

    temperature: float = 0.0
    max_output_tokens: int = 1024
    top_p: float = 1.0
    stop: tuple[str, ...] = field(default=())
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be within [0, 2]")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be within (0, 1]")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be >= 1")


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """One completion, with the provenance of the run that produced it."""

    text: str
    provenance: Provenance
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str = "stop"

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("a generation result must contain text")

    @property
    def total_tokens(self) -> int | None:
        """Total token count, or None when the backend does not report it."""
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens

    @property
    def is_npu(self) -> bool:
        """True when this completion was verified to have run on an NPU."""
        return self.provenance.is_npu


class LlmBackend(Component):
    """Language model used for the reasoning step.

    Implementations wrap a local engine (for example ONNX Runtime with a local
    model, or a vendor runtime on the Snapdragon target). They must return real
    model output or raise: a template-generated "answer" is not acceptable.
    """

    kind = "llm"

    @abstractmethod
    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        params: GenerationParams | None = None,
    ) -> GenerationResult:
        """Produce a single completion for a conversation."""

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        params: GenerationParams | None = None,
    ) -> Iterator[str]:
        """Yield completion chunks incrementally.

        Optional: the default raises, and callers fall back to
        :meth:`generate` when streaming is unavailable.
        """
        raise NotImplementedError(f"{self.info.name} does not support streaming")

    def count_tokens(self, text: str) -> int | None:
        """Token count for a string, or None when the backend cannot report it."""
        return None

    @property
    def context_window(self) -> int | None:
        """Maximum context length in tokens, when the backend knows it."""
        return None

    def preferred_capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.TEXT_GENERATION, Capability.CHAT})
