"""What this machine can actually do, and what is merely advertised.

This panel is the visible face of the project's honesty rule. An accelerator
that a runtime *lists* is shown as "advertised, not verified" and can never be
rendered as "running on the NPU". Only a real, measured run promotes a target
to verified, and that state is displayed as a word rather than a colour alone.
"""

from __future__ import annotations

import platform

from PySide6.QtWidgets import QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from core.execution import AcceleratorKind, ExecutionTarget
from core.version import __version__
from app.ui.theme import Theme

__all__ = ["EnvironmentPanel"]

_HEADERS = ("Capability", "Status", "Detail")


class EnvironmentPanel(QWidget):
    """Reports host, runtime, configured models, and execution targets."""

    def __init__(self, theme: Theme, parent=None) -> None:  # noqa: ANN001 - Qt parent
        super().__init__(parent)
        self._theme = theme
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._headline = QLabel()
        self._headline.setWordWrap(True)
        layout.addWidget(self._headline)

        self._table = QTableWidget(0, len(_HEADERS))
        self._table.setHorizontalHeaderLabels(_HEADERS)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table, 1)

        self._note = QLabel(
            "NPU execution is reported only after a real inference is verified on "
            "the accelerator. A provider that a runtime merely lists is shown as "
            "advertised, and is never presented as a measurement."
        )
        self._note.setWordWrap(True)
        self._note.setStyleSheet(f"color: {self._theme.text_dim};")
        layout.addWidget(self._note)

    def refresh(self, settings) -> None:  # noqa: ANN001 - AppSettings
        """Rebuild the table from configuration and the current runtime."""
        rows: list[tuple[str, str, str]] = [
            ("Application", f"v{__version__} ({settings.environment})", settings.app_name),
            ("Python", platform.python_version(), platform.python_implementation()),
            ("Platform", f"{platform.system()} {platform.machine()}", platform.node()),
        ]

        onnx_installed = _onnxruntime_installed()
        rows.append(
            (
                "ONNX Runtime",
                "installed" if onnx_installed else "not installed",
                "optional dependency; the app runs without it",
            )
        )

        for role, entry in _configured_models(settings):
            rows.append((f"Model ({role})", entry, ""))

        rows.append(
            (
                "AI backend",
                settings.ai.llm.name,
                "language model reasoning " + ("on" if settings.ai.enable_llm_reasoning else "off"),
            )
        )
        rows.append(
            (
                "Remediation",
                "read-only only" if not settings.safety.allow_mutating_actions else "mutating allowed",
                f"max risk {settings.safety.max_risk.value}; approval always required",
            )
        )
        rows.append(
            (
                "Execution target",
                "cpu (verified)",
                "diagnostics run as pure Python on the CPU",
            )
        )
        self._show(rows, settings)

    def show_target(self, target: ExecutionTarget) -> None:
        """Describe a real execution target, distinguishing claim from proof."""
        status = target.status
        if target.kind is AcceleratorKind.NPU:
            word = "verified NPU run" if target.is_verified else "advertised NPU (not verified)"
        else:
            word = status.value
        self._headline.setText(f"Last run: {word} - {target.describe()}")

    def _show(self, rows, settings) -> None:  # noqa: ANN001 - AppSettings
        verified = [r for r in rows if "verified" in r[1]]
        self._headline.setText(
            f"{len(verified)} capability row(s) reflect a measured or installed "
            f"state. Data directory: {settings.data_dir}"
        )
        self._table.setRowCount(len(rows))
        for row, (name, status, detail) in enumerate(rows):
            for column, text in enumerate((name, status, detail)):
                self._table.setItem(row, column, QTableWidgetItem(text))
        self._table.resizeColumnsToContents()
        self._table.horizontalHeader().setStretchLastSection(True)


def _onnxruntime_installed() -> bool:
    try:
        from ai.runtime import onnxruntime_available
    except Exception:  # pragma: no cover - defensive; the module imports cleanly
        return False
    return onnxruntime_available()


def _configured_models(settings) -> list[tuple[str, str]]:  # noqa: ANN001 - AppSettings
    """(role, description) for each enabled model entry."""
    return [
        (entry.role.value, f"{entry.model_id} at {entry.resolved_path()}")
        for entry in settings.enabled_models()
    ]
