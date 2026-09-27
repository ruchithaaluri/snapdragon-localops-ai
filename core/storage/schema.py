"""SQLite persistence for runs, evidence, approvals and benchmark reports.

Uses the standard library ``sqlite3`` module only - no ORM, no driver to
install. The schema is created idempotently at open time and versioned with
``PRAGMA user_version``, so a future migration is a numbered step rather than a
rewrite.

The database holds *local* diagnostic history. It is created under the data
directory and is never synced anywhere.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.errors import StorageError
from core.ids import new_id, utc_now

__all__ = ["SCHEMA_VERSION", "RunRecord", "SqliteStore", "open_store"]

#: Bump when the schema below changes; see ``_apply_migrations``.
SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    question     TEXT NOT NULL,
    outcome      TEXT NOT NULL,
    payload      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence (
    evidence_id  TEXT PRIMARY KEY,
    run_id       TEXT,
    kind         TEXT NOT NULL,
    source       TEXT,
    content      TEXT NOT NULL,
    confidence   REAL NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS approvals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT,
    action_id    TEXT NOT NULL,
    decision     TEXT NOT NULL,
    decided_by   TEXT NOT NULL,
    note         TEXT,
    decided_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS benchmark_reports (
    report_id    TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    host         TEXT NOT NULL,
    runtime      TEXT NOT NULL,
    payload      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_evidence_run ON evidence(run_id);
CREATE INDEX IF NOT EXISTS idx_approvals_run ON approvals(run_id);
"""
