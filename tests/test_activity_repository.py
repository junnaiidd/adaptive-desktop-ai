"""Tests for local SQLite activity persistence."""

from datetime import datetime, timedelta, timezone
import sqlite3

from app.core.activity_monitor import sanitize_window_title
from app.database.activity_repository import ActivityRepository, ActivitySegment, StoredSession


def _timestamp(seconds: int = 0) -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)


def test_repository_initializes_local_schema(tmp_path) -> None:
    database_path = tmp_path / "activity.db"
    ActivityRepository(database_path)

    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }

    assert {"sessions", "activity_segments"} <= tables


def test_repository_inserts_and_retrieves_activity_in_utc(tmp_path) -> None:
    repository = ActivityRepository(tmp_path / "activity.db")
    repository.start_session(StoredSession("session-1", _timestamp(), None))
    inserted = repository.insert_activity(
        ActivitySegment(
            session_id="session-1",
            started_at=_timestamp(),
            ended_at=_timestamp(12),
            application="Code",
            process_name="Code.exe",
            window_title="main.py - adaptive-desktop-ai",
            duration_seconds=12.0,
        )
    )

    activities = repository.list_activities()

    assert inserted.id == activities[0].id
    assert activities[0].duration_seconds == 12.0
    assert activities[0].started_at.tzinfo is timezone.utc
    assert activities[0].window_title == "main.py - adaptive-desktop-ai"


def test_repository_persists_the_sanitized_title_value(tmp_path) -> None:
    repository = ActivityRepository(tmp_path / "activity.db")
    repository.start_session(StoredSession("session-1", _timestamp(), None))
    repository.insert_activity(
        ActivitySegment(
            session_id="session-1",
            started_at=_timestamp(),
            ended_at=_timestamp(),
            application="Browser",
            process_name="browser.exe",
            window_title=sanitize_window_title("Reset password - Browser"),
            duration_seconds=0.0,
        )
    )

    assert repository.list_activities()[0].window_title == "[Sensitive window title hidden]"


def test_repository_returns_recent_activity_by_end_time_newest_first(tmp_path) -> None:
    repository = ActivityRepository(tmp_path / "activity.db")
    repository.start_session(StoredSession("session-1", _timestamp(), None))
    for application, start_seconds, end_seconds in (
        ("Old", 0, 10),
        ("Newest", 5, 30),
        ("Middle", 20, 25),
    ):
        repository.insert_activity(
            ActivitySegment(
                session_id="session-1",
                started_at=_timestamp(start_seconds),
                ended_at=_timestamp(end_seconds),
                application=application,
                process_name=f"{application}.exe",
                window_title=application,
                duration_seconds=end_seconds - start_seconds,
            )
        )

    recent = repository.list_recent_activities(limit=2)

    assert [activity.application for activity in recent] == ["Newest", "Middle"]
