"""Colours and stylesheet for the desktop shell.

Two palettes, one source of truth. The semantic names are what the rest of the
UI refers to - ``ok``, ``warn``, ``danger`` - so a status is never coloured by
picking a hex value at the call site.

The palettes are chosen for text contrast on their own background, so a status
remains readable for a user with colour vision deficiency: every status also
carries a word, never colour alone.
"""

from __future__ import annotations

from typing import Literal

__all__ = ["PALETTES", "STATUS_COLORS", "Theme", "stylesheet_for"]

ThemeName = Literal["dark", "light"]

#: Background / text colours per theme.
PALETTES: dict[str, dict[str, str]] = {
    "dark": {
        "bg": "#14161a",
        "surface": "#1c1f26",
        "surface_alt": "#232732",
        "border": "#333a48",
        "text": "#e6e9ef",
        "text_dim": "#9aa4b8",
        "accent": "#4c8dff",
    },
    "light": {
        "bg": "#f5f6f8",
        "surface": "#ffffff",
        "surface_alt": "#eef0f4",
        "border": "#d3d8e0",
        "text": "#1a1d23",
        "text_dim": "#5b6474",
        "accent": "#1f6feb",
    },
}

#: Semantic status colours, keyed by the same names the panels use.
STATUS_COLORS: dict[str, dict[str, str]] = {
    "ok": {"dark": "#3fb950", "light": "#1a7f37"},
    "warn": {"dark": "#d29922", "light": "#9a6700"},
    "danger": {"dark": "#f85149", "light": "#cf222e"},
    "info": {"dark": "#58a6ff", "light": "#0969da"},
    "muted": {"dark": "#8b949e", "light": "#57606a"},
}


class Theme:
    """Resolved colours for one theme name."""

    def __init__(self, name: ThemeName = "dark") -> None:
        if name not in PALETTES:
            raise ValueError(f"unknown theme: {name!r}")
        self.name = name
        self._colors = PALETTES[name]

    def __getattr__(self, key: str) -> str:
        try:
            return self._colors[key]
        except KeyError as exc:
            raise AttributeError(f"no colour named {key!r}") from exc

    def status(self, name: str) -> str:
        """Colour for a semantic status, falling back to a muted grey."""
        entry = STATUS_COLORS.get(name)
        return entry[self.name] if entry else STATUS_COLORS["muted"][self.name]


def stylesheet_for(theme: Theme) -> str:
    """Build the application stylesheet for a resolved theme."""
    return f"""
    QWidget {{
        background: {theme.bg};
        color: {theme.text};
        font-size: 13px;
    }}
    QMainWindow::separator {{ background: {theme.border}; width: 1px; height: 1px; }}
    QGroupBox {{
        border: 1px solid {theme.border};
        border-radius: 6px;
        margin-top: 14px;
        padding-top: 10px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 4px;
        color: {theme.text_dim};
    }}
    QTabWidget::pane {{ border: 1px solid {theme.border}; border-radius: 6px; }}
    QTabBar::tab {{
        background: {theme.surface};
        color: {theme.text_dim};
        padding: 6px 14px;
        border: 1px solid {theme.border};
        border-bottom: none;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
    }}
    QTabBar::tab:selected {{ background: {theme.surface_alt}; color: {theme.text}; }}
    QPushButton {{
        background: {theme.surface_alt};
        color: {theme.text};
        border: 1px solid {theme.border};
        border-radius: 5px;
        padding: 6px 14px;
    }}
    QPushButton:hover {{ border-color: {theme.accent}; }}
    QPushButton:disabled {{ color: {theme.text_dim}; border-color: {theme.border}; }}
    QPushButton[primary="true"] {{
        background: {theme.accent};
        color: #ffffff;
        border-color: {theme.accent};
    }}
    QTextEdit, QPlainTextEdit, QLineEdit, QTableWidget, QTreeWidget, QListWidget {{
        background: {theme.surface};
        border: 1px solid {theme.border};
        border-radius: 5px;
        selection-background-color: {theme.accent};
    }}
    QHeaderView::section {{
        background: {theme.surface_alt};
        color: {theme.text_dim};
        border: none;
        border-bottom: 1px solid {theme.border};
        padding: 5px 8px;
    }}
    QStatusBar {{ background: {theme.surface}; color: {theme.text_dim}; }}
    QToolTip {{
        background: {theme.surface_alt};
        color: {theme.text};
        border: 1px solid {theme.border};
    }}
    """
