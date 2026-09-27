"""Result panels. Each one renders a stage output; none computes anything."""

from __future__ import annotations

from app.ui.panels.diagnosis_panel import DiagnosisPanel
from app.ui.panels.environment_panel import EnvironmentPanel
from app.ui.panels.evidence_panel import EvidencePanel
from app.ui.panels.input_panel import InputPanel
from app.ui.panels.remediation_panel import RemediationPanel
from app.ui.panels.verification_panel import VerificationPanel

__all__ = [
    "DiagnosisPanel",
    "EnvironmentPanel",
    "EvidencePanel",
    "InputPanel",
    "RemediationPanel",
    "VerificationPanel",
]
