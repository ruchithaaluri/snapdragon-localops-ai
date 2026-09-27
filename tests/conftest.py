"""Shared fixtures.

Two rules for the whole suite:

* **No test touches the real user profile.** ``data_dir`` points every test at a
  temporary directory, so running the suite cannot read or write a real
  ``%LOCALAPPDATA%/SnapdragonLocalOps``.
* **No test reaches the network.** Every backend used here is local and
  deterministic.
"""

from __future__ import annotations

import pytest

from core.config.schema import AppSettings
from core.observability import configure_logging, shutdown_logging
from core.safety import InMemoryApprovalGate, SafetyPolicy
from diagnostics import RuleBasedDiagnosticEngine, build_query
from diagnostics.remediation import LocalRemediationExecutor
from verification import LocalVerificationRunner

#: A realistic mixed-toolchain failure, used by several tests.
SAMPLE_LOG = (
    "2026-09-26 09:14:02 INFO  starting dev server\n"
    "Traceback (most recent call last):\n"
    '  File "C:\\proj\\app.py", line 42, in run\n'
    "    import requests\n"
    "ModuleNotFoundError: No module named 'requests'\n"
    "EADDRINUSE: address already in use :::3000\n"
    "Access is denied.\n"
    "2026-09-26 09:14:03 INFO  shutting down\n"
)


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Point the whole application at a throwaway data directory."""
    root = tmp_path / "localops-data"
    root.mkdir()
    monkeypatch.setenv("LOCALOPS_DATA_DIR", str(root))
    monkeypatch.delenv("LOCALOPS_CONFIG_FILE", raising=False)
    return root


@pytest.fixture(autouse=True)
def quiet_logging():
    """Keep test output readable and detach handlers afterwards."""
    configure_logging(level="WARNING", json_console=False, log_file=None, force=True)
    yield
    shutdown_logging()


@pytest.fixture
def settings() -> AppSettings:
    """Default settings, built without reading the environment."""
    return AppSettings.model_validate({})


@pytest.fixture
def policy() -> SafetyPolicy:
    """The default, safest policy: read-only, approval always required."""
    return SafetyPolicy()


@pytest.fixture
def gate() -> InMemoryApprovalGate:
    return InMemoryApprovalGate()


@pytest.fixture
def engine() -> RuleBasedDiagnosticEngine:
    return RuleBasedDiagnosticEngine()


@pytest.fixture
def query():
    """A query built from :data:`SAMPLE_LOG`."""
    return build_query("Why does the dev server fail?", SAMPLE_LOG, source="app.log")


@pytest.fixture
def executor(policy) -> LocalRemediationExecutor:
    return LocalRemediationExecutor(policy)


@pytest.fixture
def verifier(gate) -> LocalVerificationRunner:
    """A verifier sharing the test gate, so approvals can be recorded."""
    return LocalVerificationRunner(gate)


@pytest.fixture
def log_file(tmp_path):
    """A log on disk that still contains the error, for re-reading checks."""
    path = tmp_path / "app.log"
    path.write_text(SAMPLE_LOG, encoding="utf-8")
    return path
