"""Backend selection.

One place decides which :class:`~core.contracts.ai.AIBackend` the application
uses. The rules:

* the default is the mock backend, so a fresh install works offline,
* a real backend is selected only by explicit configuration,
* an unknown name is an error - never a silent fallback to the mock, because
  that would hide a typo and show demo output where a model was expected.
"""

from __future__ import annotations

from core.config.schema import AppSettings
from core.contracts.ai import AIBackend
from core.errors import ProviderNotFoundError
from core.observability import get_logger, log_event

__all__ = ["BACKENDS", "build_ai_backend", "build_model_registry"]

_log = get_logger(__name__)

#: Registered backend factories. Keys are the names accepted in configuration.
BACKENDS: dict[str, str] = {
    "mock": "ai.backends.mock:MockAIBackend",
}

#: Names that mean "no backend configured".
_UNCONFIGURED = {"unavailable", "none", "disabled", ""}


def _load(path: str) -> type:
    module_name, _, class_name = path.partition(":")
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def build_ai_backend(settings: AppSettings) -> AIBackend:
    """Instantiate the configured AI backend.

    Raises :class:`~core.errors.ProviderNotFoundError` when the configured name
    is not registered, so a misconfiguration is visible immediately.
    """
    name = settings.ai.llm.name.strip().lower()
    if name in _UNCONFIGURED:
        log_event(
            _log, 20, "ai.backend.selected",
            backend="mock", reason="no backend configured, using deterministic mock",
        )
        return _load(BACKENDS["mock"])()

    target = BACKENDS.get(name)
    if target is None:
        raise ProviderNotFoundError(
            f"unknown AI backend {name!r}",
            available=sorted(BACKENDS),
        )
    backend = _load(target)()
    log_event(_log, 20, "ai.backend.selected", backend=name, is_mock=backend.is_mock())
    return backend


def build_model_registry(settings: AppSettings):
    """Map configured model entries onto a :class:`ModelRegistry`."""
    from core.model_artifact import ModelArtifact, ModelRegistry

    registry = ModelRegistry()
    for entry in settings.enabled_models():
        registry.register(
            entry.role,
            ModelArtifact.from_path(
                model_id=entry.model_id,
                path=entry.resolved_path(),
                format=entry.format,
                source=entry.source,
                sha256=entry.sha256,
                quantization=entry.quantization,
                metadata=dict(entry.options),
            ),
        )
    return registry
