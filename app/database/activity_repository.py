"""Local SQLite storage for completed privacy-sanitized activity segments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3


@dataclass(frozen=True, slots=True)
class ActivitySegment:
    """A completed, time-bounded period of one foreground activity."""

    session_id: str
    started_at: datetime
    ended_at: datetime
    application: str | None
    process_name: str | None
    window_title: str | None
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class StoredActivity(ActivitySegment):
    """An activity segment after it has been assigned a database identifier."""

    id: int


@dataclass(frozen=True, slots=True)
class StoredSession:
    """A persisted continuous period of observed desktop activity."""

    id: str
    started_at: datetime
    ended_at: datetime | None


class ActivityRepository:
    """Persist activity segments locally without coupling to the Windows observer."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        default_path = Path(__file__).resolve().parents[2] / "data" / "activity.db"
        self.database_path = Path(database_path) if database_path else default_path
        self.initialize()

    def initialize(self) -> None:
        """Create the local schema and indexes when they do not already exist."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    started_at_utc TEXT NOT NULL,
                    ended_at_utc TEXT
                );

                CREATE TABLE IF NOT EXISTS activity_segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    started_at_utc TEXT NOT NULL,
                    ended_at_utc TEXT NOT NULL,
                    application TEXT,
                    process_name TEXT,
                    window_title TEXT,
                    duration_seconds REAL NOT NULL CHECK (duration_seconds >= 0),
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );

                CREATE INDEX IF NOT EXISTS idx_activity_segments_started_at
                    ON activity_segments (started_at_utc);
                CREATE INDEX IF NOT EXISTS idx_activity_segments_session_id
                    ON activity_segments (session_id);
                """
            )

    def start_session(self, session: StoredSession) -> None:
        """Store a new session before its activity segments are recorded."""
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO sessions (id, started_at_utc, ended_at_utc) VALUES (?, ?, ?)",
                (session.id, _as_utc_text(session.started_at), _as_utc_text(session.ended_at)),
            )

    def end_session(self, session: StoredSession) -> None:
        """Set a session end time after it becomes inactive or is shut down."""
        with self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET ended_at_utc = ? WHERE id = ?",
                (_as_utc_text(session.ended_at), session.id),
            )

    def insert_activity(self, segment: ActivitySegment) -> StoredActivity:
        """Insert one completed segment using parameterized SQL."""
        if segment.duration_seconds < 0:
            raise ValueError("Activity duration cannot be negative.")

        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO activity_segments (
                    session_id, started_at_utc, ended_at_utc, application,
                    process_name, window_title, duration_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    segment.session_id,
                    _as_utc_text(segment.started_at),
                    _as_utc_text(segment.ended_at),
                    segment.application,
                    segment.process_name,
                    segment.window_title,
                    segment.duration_seconds,
                ),
            )
        return StoredActivity(
            id=cursor.lastrowid,
            session_id=segment.session_id,
            started_at=segment.started_at,
            ended_at=segment.ended_at,
            application=segment.application,
            process_name=segment.process_name,
            window_title=segment.window_title,
            duration_seconds=segment.duration_seconds,
        )

    def list_activities(self) -> list[StoredActivity]:
        """Return stored activity in chronological order for later feature extraction."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM activity_segments ORDER BY started_at_utc, id"
            ).fetchall()
        return [_activity_from_row(row) for row in rows]

    def list_recent_activities(self, limit: int = 50) -> list[StoredActivity]:
        """Return the newest completed segments first for live dashboard presentation."""
        if limit <= 0:
            raise ValueError("The recent activity limit must be positive.")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM activity_segments
                ORDER BY ended_at_utc DESC, started_at_utc DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [_activity_from_row(row) for row in rows]

    def list_sessions(self) -> list[StoredSession]:
        """Return sessions in chronological order."""
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM sessions ORDER BY started_at_utc").fetchall()
        return [_session_from_row(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def _as_utc_text(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        raise ValueError("Timestamps must be timezone-aware.")
    return value.astimezone(timezone.utc).isoformat()


def _from_utc_text(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def _activity_from_row(row: sqlite3.Row) -> StoredActivity:
    return StoredActivity(
        id=row["id"],
        session_id=row["session_id"],
        started_at=_from_utc_text(row["started_at_utc"]),
        ended_at=_from_utc_text(row["ended_at_utc"]),
        application=row["application"],
        process_name=row["process_name"],
        window_title=row["window_title"],
        duration_seconds=row["duration_seconds"],
    )


def _session_from_row(row: sqlite3.Row) -> StoredSession:
    return StoredSession(
        id=row["id"],
        started_at=_from_utc_text(row["started_at_utc"]),
        ended_at=_from_utc_text(row["ended_at_utc"]),
    )
