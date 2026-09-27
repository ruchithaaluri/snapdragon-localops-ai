"""Where the user types the question and supplies the raw text.

This panel is the only place the application accepts input. It reads files the
user explicitly picks, and it never walks a directory: an application that
silently indexes a whole project is an application the user cannot reason about.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

__all__ = ["InputPanel"]

#: Refuse to read anything larger than this in one go.
_MAX_FILE_BYTES = 8 * 1024 * 1024


class InputPanel(QWidget):
    """Question, pasted or loaded text, and the button that starts a run."""

    analyze_requested = Signal()
    load_requested = Signal()

    def __init__(self, parent=None) -> None:  # noqa: ANN001 - Qt parent
        super().__init__(parent)
        self._build()
        self._connect()

    # -- construction -----------------------------------------------------

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(QLabel("What are you trying to work out?"))
        self._question = QLineEdit()
        self._question.setPlaceholderText(
            "e.g. Why does my dev server exit immediately on Windows?"
        )
        self._question.setClearButtonEnabled(True)
        layout.addWidget(self._question)

        evidence_box = QGroupBox("Evidence to analyse")
        evidence_layout = QVBoxLayout(evidence_box)
        evidence_layout.setSpacing(6)

        buttons = QHBoxLayout()
        self._load_button = QPushButton("Load a log or error file…")
        self._load_button.setToolTip(
            "Read one file. Nothing is indexed in the background."
        )
        self._source = QLineEdit()
        self._source.setPlaceholderText("source name shown in citations")
        self._source.setReadOnly(True)
        buttons.addWidget(self._load_button)
        buttons.addWidget(self._source, 1)
        evidence_layout.addLayout(buttons)

        self._text = QPlainTextEdit()
        self._text.setPlaceholderText(
            "Paste a stack trace, terminal output, or a log excerpt here.\n\n"
            "Every extracted item keeps the line it came from, so each claim in "
            "the answer can be traced back to this text."
        )
        self._text.setTabChangesFocus(True)
        evidence_layout.addWidget(self._text)
        layout.addWidget(evidence_box, 1)

        self._analyze = QPushButton("Analyse")
        self._analyze.setProperty("primary", True)
        self._analyze.setEnabled(False)
        layout.addWidget(self._analyze)

    def _connect(self) -> None:
        self._analyze.clicked.connect(self.analyze_requested.emit)
        self._load_button.clicked.connect(self._choose_file)
        self._text.textChanged.connect(self._refresh_enabled)
        self._question.textChanged.connect(self._refresh_enabled)

    # -- behaviour --------------------------------------------------------

    def _refresh_enabled(self) -> None:
        """Analyse needs both a question and something to analyse."""
        self._analyze.setEnabled(
            bool(self._question.text().strip() and self._text.toPlainText().strip())
        )

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a log or error file", "", "Text and logs (*.log *.txt *.err);;All files (*)"
        )
        if not path:
            return
        self.load_file(Path(path))

    def load_file(self, path: Path) -> bool:
        """Read one file into the editor. Returns False when it cannot be used."""
        if not path.is_file():
            return False
        if path.stat().st_size > _MAX_FILE_BYTES:
            self._source.setText(f"{path.name} (too large to read)")
            return False
        self._text.setPlainText(
            path.read_text(encoding="utf-8", errors="replace")
        )
        self._source.setText(path.name)
        return True

    # -- accessors --------------------------------------------------------

    def question(self) -> str:
        return self._question.text().strip()

    def text(self) -> str:
        return self._text.toPlainText()

    def source(self) -> str:
        return self._source.text().strip() or "pasted text"

    def set_busy(self, busy: bool) -> None:
        """Disable input while a run is in flight."""
        for widget in (self._analyze, self._load_button, self._text, self._question):
            widget.setEnabled(not busy)
        if not busy:
            self._refresh_enabled()

    def set_message(self, text: str) -> None:
        self._source.setText(text)

    def focus_question(self) -> None:
        self._question.setFocus(Qt.ShortcutFocusReason)
