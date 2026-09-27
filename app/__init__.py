"""The desktop application: the Qt shell and the pipeline that feeds it.

Layout
------
``pipeline``
    Composes the layer packages into one end-to-end run. It is pure Python and
    imports no Qt, so it can be driven from a test or a CLI without a display.
``ui``
    The PySide6 shell. It renders results and collects decisions; it contains no
    diagnostic logic of its own.
``main``
    The ``snapdragon-localops`` entry point.

Nothing in :mod:`core` imports this package - the dependency arrow points one
way, from the shell down into the contracts.
"""

from __future__ import annotations

from app.pipeline import LocalDiagnosticPipeline, build_pipeline

__all__ = ["LocalDiagnosticPipeline", "build_pipeline"]
