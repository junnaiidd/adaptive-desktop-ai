"""Local persistence for explicitly captured, Work Thread-scoped snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from app.core.activity_monitor import sanitize_window_title


class WorkspaceSnapshotError(Exception):
    """Base exception for workspace snapshot persistence operations."""


class WorkspaceSnapshotNotFoundError(WorkspaceSnapshotError, KeyError):
    """Raised when a requested snapshot does not exist."""


@dataclass(frozen=True, slots=True)
class CapturedForegroundWindow:
    """The minimal, privacy-sanitized facts captured after a user action."""

    application: str | None
    process_name: str
    executable_path: str
    window_title: str | None


@dataclass(frozen=True, slots=True)
class WorkspaceSnapshot:
    """One user-requested foreground-app snapshot for exactly one Work Thread."""

    id: int
    work_thread_id: int
    captured_at: datetime
    application: str | None
    process_name: str
    executable_path: str
    window_title: str | None


class WorkspaceSnapshotStore:
    """Own the isolated ``workspace_snapshots`` table in the shared local database."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        default_path = Path(__file__).resolve().parents[2] / "data" / "activity.db"
        self.database_path = Path(database_path) if database_path else default_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            with connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS workspace_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        work_thread_id INTEGER NOT NULL,
                        captured_at_utc TEXT NOT NULL,
                        application TEXT,
                        process_name TEXT NOT NULL,
                        executable_path TEXT NOT NULL,
                        window_title TEXT,
                        FOREIGN KEY (work_thread_id) REFERENCES work_threads(id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_workspace_snapshots_thread_id
                        ON workspace_snapshots (work_thread_id, captured_at_utc, id);
                    """
                )
        finally:
            connection.close()

    def create_snapshot(
        self,
        work_thread_id: int,
        captured_window: CapturedForegroundWindow,
        *,
        captured_at: datetime,
    ) -> WorkspaceSnapshot:
        """Persist one explicitly captured foreground app for an existing Work Thread."""
        _require_timezone_aware(captured_at, "captured_at")
        _validate_non_empty_string(captured_window.process_name, "process_name")
        _validate_non_empty_string(captured_window.executable_path, "executable_path")
        if captured_window.application is not None and not isinstance(captured_window.application, str):
            raise ValueError("application must be a string or None.")
        if captured_window.window_title is not None and not isinstance(captured_window.window_title, str):
            raise ValueError("window_title must be a string or None.")

        captured_at_utc = captured_at.astimezone(timezone.utc)
        sanitized_title = sanitize_window_title(captured_window.window_title)
        application = captured_window.application.strip() if captured_window.application else None
        connection = self._connect()
        try:
            with connection:
                cursor = connection.execute(
                    """
                    INSERT INTO workspace_snapshots (
                        work_thread_id, captured_at_utc, application, process_name,
                        executable_path, window_title
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        work_thread_id,
                        _format_utc(captured_at_utc),
                        application,
                        captured_window.process_name.strip(),
                        captured_window.executable_path.strip(),
                        sanitized_title,
                    ),
                )
                snapshot_id = cursor.lastrowid
        finally:
            connection.close()

        return WorkspaceSnapshot(
            id=snapshot_id,
            work_thread_id=work_thread_id,
            captured_at=captured_at_utc,
            application=application,
            process_name=captured_window.process_name.strip(),
            executable_path=captured_window.executable_path.strip(),
            window_title=sanitized_title,
        )

    def get_snapshot(self, snapshot_id: int) -> WorkspaceSnapshot | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM workspace_snapshots WHERE id = ?", (snapshot_id,)
            ).fetchone()
        finally:
            connection.close()
        return _snapshot_from_row(row) if row is not None else None

    def list_snapshots_for_work_thread(self, work_thread_id: int) -> list[WorkspaceSnapshot]:
        """Return snapshots oldest first, tie-broken by their database id."""
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT * FROM workspace_snapshots
                WHERE work_thread_id = ?
                ORDER BY captured_at_utc ASC, id ASC
                """,
                (work_thread_id,),
            ).fetchall()
        finally:
            connection.close()
        return [_snapshot_from_row(row) for row in rows]

    def delete_snapshot(self, snapshot_id: int) -> None:
        connection = self._connect()
        try:
            with connection:
                cursor = connection.execute("DELETE FROM workspace_snapshots WHERE id = ?", (snapshot_id,))
                if cursor.rowcount == 0:
                    raise WorkspaceSnapshotNotFoundError(f"Workspace snapshot with id {snapshot_id} not found.")
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def _validate_non_empty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _require_timezone_aware(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a datetime instance.")
    if value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware.")


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def _snapshot_from_row(row: sqlite3.Row) -> WorkspaceSnapshot:
    return WorkspaceSnapshot(
        id=row["id"],
        work_thread_id=row["work_thread_id"],
        captured_at=_parse_utc(row["captured_at_utc"]),
        application=row["application"],
        process_name=row["process_name"],
        executable_path=row["executable_path"],
        window_title=row["window_title"],
    )
