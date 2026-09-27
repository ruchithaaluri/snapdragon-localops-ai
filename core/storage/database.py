"""A small, explicit SQLite store.

Deliberately thin: open/execute/fetchall with error translation and explicit
transactions. The alternative - an ORM - would add a dependency and hide exactly
the behaviour a local-first tool should make obvious.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.errors import StorageError
from core.ids import utc_now
from core.storage.schema import SCHEMA_VERSION, _SCHEMA

__all__ = ["RunRecord", "SqliteStore", "open_store"]


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One stored diagnostic run."""

    run_id: str
    created_at: str
    question: str
    outcome: str
    payload: dict[str, Any]


def _connect(path: Path) -> sqlite3.Connection:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
    except (sqlite3.Error, OSError) as exc:  # pragma: no cover - filesystem dependent
        raise StorageError("could not open the local database", path=str(path)) from exc
    conn.row_factory = sqlite3.Row
    # WAL keeps the UI responsive while a run is being written.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


class SqliteStore:
    """Local database of runs, evidence, approvals and benchmark reports."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._conn = _connect(self._path)
        self._migrate()

    @property
    def path(self) -> Path:
        """Location of the database file."""
        return self._path

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def _migrate(self) -> None:
        """Create the schema and stamp the version. Idempotent."""
        try:
            with self.transaction():
                self._conn.executescript(_SCHEMA)
                current = self._conn.execute("PRAGMA user_version").fetchone()[0]
                if current < SCHEMA_VERSION:
                    self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        except sqlite3.Error as exc:
            raise StorageError("schema migration failed", detail=str(exc)) from exc

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Commit on success, roll back on error."""
        try:
            yield self._conn
        except Exception:
            self._conn.rollback()
            raise
        else:
            self._conn.commit()

    def execute(self, sql: str, params: Sequence[Any] | Mapping[str, Any] = ()) -> sqlite3.Cursor:
        """Run one statement and commit.

        ``params`` may be a sequence (for ``?`` placeholders) or a mapping (for
        named ``:placeholders``), which keeps the write statements readable.
        """
        try:
            cursor = self._conn.execute(sql, params)  # type: ignore[arg-type]
            self._conn.commit()
            return cursor
        except sqlite3.Error as exc:
            raise StorageError("statement failed", detail=str(exc)) from exc

    def fetchall(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        """Run a query and return every row."""
        try:
            return self._conn.execute(sql, tuple(params)).fetchall()
        except sqlite3.Error as exc:
            raise StorageError("query failed", detail=str(exc)) from exc

    def close(self) -> None:
        """Close the connection. Idempotent."""
        try:
            self._conn.close()
        except sqlite3.Error:  # pragma: no cover - already closed
            pass

    def __enter__(self) -> "SqliteStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


    # -- runs ------------------------------------------------------------- #

    def save_run(
        self,
        *,
        run_id: str,
        question: str,
        outcome: str,
        payload: dict[str, Any],
        created_at: str | None = None,
    ) -> str:
        """Store (or replace) a run and return its id."""
        record = {
            "run_id": run_id,
            "created_at": created_at or utc_now().isoformat(timespec="seconds"),
            "question": question,
            "outcome": outcome,
            "payload": json.dumps(payload, default=str),
        }
        self.execute(
            """
            INSERT INTO runs (run_id, created_at, question, outcome, payload)
            VALUES (:run_id, :created_at, :question, :outcome, :payload)
            ON CONFLICT(run_id) DO UPDATE SET
                created_at=excluded.created_at,
                question=excluded.question,
                outcome=excluded.outcome,
                payload=excluded.payload
            """,
            record,
        )
        return run_id

    def list_runs(self, limit: int = 50) -> tuple[RunRecord, ...]:
        """Most recent runs first."""
        rows = self.fetchall("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
        return tuple(
            RunRecord(
                run_id=row["run_id"],
                created_at=row["created_at"],
                question=row["question"],
                outcome=row["outcome"],
                payload=json.loads(row["payload"]),
            )
            for row in rows
        )

    # -- evidence --------------------------------------------------------- #

    def save_evidence(self, run_id: str, items: Sequence[Any]) -> int:
        """Store evidence items, returning how many were written."""
        rows = [
            {
                "evidence_id": item.evidence_id,
                "run_id": run_id,
                "kind": item.kind.value,
                "source": item.source,
                "content": item.content,
                "confidence": item.confidence,
                "created_at": item.captured_at,
            }
            for item in items
        ]
        if not rows:
            return 0
        with self.transaction():
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO evidence
                    (evidence_id, run_id, kind, source, content, confidence, created_at)
                VALUES (:evidence_id, :run_id, :kind, :source, :content,
                        :confidence, :created_at)
                """,
                rows,
            )
        return len(rows)

    # -- approvals -------------------------------------------------------- #

    def save_approval(self, run_id: str, record: Any) -> int:
        """Append an approval decision. Decisions are append-only."""
        cursor = self.execute(
            """
            INSERT INTO approvals (run_id, action_id, decision, decided_by, note, decided_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                record.action_id,
                record.decision.value,
                record.decided_by,
                record.note,
                record.decided_at,
            ),
        )
        return int(cursor.lastrowid or 0)

    def approvals_for(self, run_id: str) -> tuple[dict[str, Any], ...]:
        """Every recorded decision for a run, newest first."""
        rows = self.fetchall(
            "SELECT * FROM approvals WHERE run_id = ? ORDER BY decided_at DESC", (run_id,)
        )
        return tuple(dict(row) for row in rows)

    # -- benchmark reports ------------------------------------------------ #

    def save_benchmark_report(self, report: Any) -> str:
        """Store a measured report. Only real measurements belong here."""
        payload = report.to_dict()
        self.execute(
            """
            INSERT OR REPLACE INTO benchmark_reports (report_id, created_at, host, runtime, payload)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                payload["report_id"],
                payload["created_at"],
                payload["host"],
                payload["runtime"],
                json.dumps(payload, default=str),
            ),
        )
        return str(payload["report_id"])

    def list_benchmark_reports(self, limit: int = 20) -> tuple[dict[str, Any], ...]:
        """Report headers, newest first."""
        rows = self.fetchall(
            "SELECT report_id, created_at, host, runtime FROM benchmark_reports "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return tuple(dict(row) for row in rows)


def open_store(path: str | Path) -> SqliteStore:
    """Open (creating if needed) the local database at ``path``."""
    return SqliteStore(path)

