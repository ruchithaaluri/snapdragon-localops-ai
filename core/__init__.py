"""Core kernel: configuration, logging, storage and the cross-layer contracts.

This package contains no PySide6 and no AI engine code, so it can be imported
by every other layer (and by tests) without side effects.
"""

from core.version import __version__

__all__ = ["__version__"]
