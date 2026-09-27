"""The PySide6 desktop shell.

Panels only render results and collect decisions; they hold no diagnostic logic.
Everything they display comes from a layer package through
:class:`~app.pipeline.LocalDiagnosticPipeline`, so the same pipeline drives both
this window and the headless ``--diagnose`` mode.
"""

from __future__ import annotations

__all__: list[str] = []
