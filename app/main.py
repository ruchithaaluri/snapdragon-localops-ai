"""The ``snapdragon-localops`` entry point.

Two modes:

``--diagnose``
    Read text from a file or stdin, run the pipeline headlessly, print a report
    and exit. No Qt import happens in this path, so it works over SSH and in
    CI where there is no display.

default
    Start the PySide6 desktop shell.

Logging is configured before anything else runs, because a failure to load
configuration is exactly the kind of problem the user needs to be able to read
from the log rather than from a traceback dialog.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from core.config import load_settings
from core.config.paths import ensure_dir, logs_dir
from core.errors import LocalOpsError
from core.observability import configure_logging, get_logger, log_event, set_session_id
from core.ids import new_session_id
from core.version import __version__

__all__ = ["build_parser", "main"]

_log = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface. Every flag has a working default."""
    parser = argparse.ArgumentParser(
        prog="snapdragon-localops",
        description="Local-first developer diagnostic assistant.",
    )
    parser.add_argument("--config", type=Path, default=None, help="path to a TOML config file")
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="override the data directory"
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="console log level (overrides configuration)",
    )
    parser.add_argument(
        "--diagnose",
        metavar="PATH",
        nargs="?",
        const="-",
        default=None,
        help="run headlessly on this file (or stdin with '-') and print a report",
    )
    parser.add_argument(
        "--question",
        default="What is causing this failure?",
        help="the question to answer in --diagnose mode",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Process entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    session = new_session_id()
    set_session_id(session)

    if args.data_dir is not None:
        import os

        os.environ["LOCALOPS_DATA_DIR"] = str(args.data_dir)

    try:
        overrides: dict[str, object] = {}
        if args.log_level:
            overrides["logging"] = {"level": args.log_level}
        settings = load_settings(args.config, overrides=overrides or None)
    except LocalOpsError as exc:
        # Logging is not configured yet, so report on stderr and stop.
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    log_file = None
    if settings.logging.file_enabled:
        log_file = ensure_dir(logs_dir()) / settings.logging.file_name
    configure_logging(
        level=settings.logging.level,
        log_file=log_file,
        json_console=settings.logging.json_console,
        json_file=True,
        max_bytes=settings.logging.max_bytes,
        backup_count=settings.logging.backup_count,
        quiet_third_party=settings.logging.quiet_third_party,
        force=True,
    )
    log_event(_log, logging.INFO, "app.started", session=session, version=__version__)

    if args.diagnose is not None:
        return _run_headless(settings, args)

    return _run_gui(settings)


def _run_headless(settings, args) -> int:
    """Run one diagnostic without Qt and print the result."""
    from app.pipeline import build_pipeline
    from diagnostics import build_query

    if args.diagnose == "-":
        text = sys.stdin.read()
        source = "stdin"
    else:
        path = Path(args.diagnose)
        if not path.is_file():
            print(f"no such file: {path}", file=sys.stderr)
            return 2
        text = path.read_text(encoding="utf-8", errors="replace")
        source = path.name

    pipeline = build_pipeline(settings)
    query = build_query(args.question, text, source=source)
    run = pipeline.run(query)

    print(f"run {run.run_id}  outcome={run.outcome.value}")
    print("\nStages")
    for stage, status, detail in run.as_rows():
        print(f"  {stage:12} {status:10} {detail}")

    if run.diagnosis is None:
        print("\nNo diagnosis: the evidence did not support a conclusion.")
        return 1

    d = run.diagnosis
    print(f"\nDiagnosis ({d.reasoning_mode.value}, confidence {d.confidence:.2f})")
    print(f"  {d.summary}")
    for h in d.ranked():
        print(f"  - [{h.status.value}] {h.confidence:.2f} {h.statement}")
        if h.rationale:
            print(f"      {h.rationale}")
    if d.gaps:
        print("\nCaveats")
        for gap in d.gaps:
            print(f"  - {gap}")
    if run.plan and run.plan.actions:
        print(f"\nProposed actions (nothing has been approved or run):")
        for action in run.plan.actions:
            print(f"  [{action.risk.value}] {action.title}")
            print(f"      {action.description}")
            if action.command:
                print(f"      $ {action.command}")
    return 0


def _run_gui(settings) -> int:
    """Start the desktop shell."""
    try:
        from app.ui.main_window import run_gui
    except ImportError as exc:
        print(
            "PySide6 is required for the desktop shell. Install it with "
            "'pip install .' or use --diagnose for the headless mode.\n"
            f"Import failed: {exc}",
            file=sys.stderr,
        )
        return 3
    return run_gui(settings)


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
