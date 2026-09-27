"""Configuration: typed schema, layered loading, and filesystem layout."""

from __future__ import annotations

from core.config.loader import (
    CONFIG_FILE_ENV_VAR,
    SettingsSource,
    default_config_path,
    load_settings,
)
from core.config.paths import (
    APP_DIR_NAME,
    DATA_DIR_ENV_VAR,
    cache_dir,
    data_dir,
    ensure_dir,
    index_dir,
    logs_dir,
    models_dir,
    reports_dir,
    runtime_dir,
)
from core.config.schema import (
    AiSettings,
    AppSettings,
    BenchmarkSettings,
    DiagnosticsSettings,
    LoggingSettings,
    ModelEntry,
    ProviderChoice,
    RetrievalSettings,
    RuntimeSettings,
    SafetySettings,
    UiSettings,
    VerificationSettings,
)

__all__ = [
    "APP_DIR_NAME",
    "CONFIG_FILE_ENV_VAR",
    "DATA_DIR_ENV_VAR",
    "AiSettings",
    "AppSettings",
    "BenchmarkSettings",
    "DiagnosticsSettings",
    "LoggingSettings",
    "ModelEntry",
    "ProviderChoice",
    "RetrievalSettings",
    "RuntimeSettings",
    "SafetySettings",
    "SettingsSource",
    "UiSettings",
    "VerificationSettings",
    "cache_dir",
    "data_dir",
    "default_config_path",
    "ensure_dir",
    "index_dir",
    "load_settings",
    "logs_dir",
    "models_dir",
    "reports_dir",
    "runtime_dir",
]
