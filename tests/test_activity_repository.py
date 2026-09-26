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


def test_list_activities_for_session_returns_only_that_sessions_activities(tmp_path) -> None:
    repository = ActivityRepository(tmp_path / "activity.db")
    repository.start_session(StoredSession("session-1", _timestamp(), None))
    repository.start_session(StoredSession("session-2", _timestamp(100), None))
    repository.insert_activity(
        ActivitySegment(
            session_id="session-1",
            started_at=_timestamp(0),
            ended_at=_timestamp(10),
            application="Code",
            process_name="Code.exe",
            window_title="Code",
            duration_seconds=10.0,
        )
    )
    repository.insert_activity(
        ActivitySegment(
            session_id="session-2",
            started_at=_timestamp(100),
            ended_at=_timestamp(110),
            application="Browser",
            process_name="browser.exe",
            window_title="Browser",
            duration_seconds=10.0,
        )
    )

    session_1_activities = repository.list_activities_for_session("session-1")

    assert len(session_1_activities) == 1
    assert session_1_activities[0].application == "Code"
    assert session_1_activities[0].session_id == "session-1"


def test_list_activities_for_session_with_multiple_sessions_filters_correctly(tmp_path) -> None:
    repository = ActivityRepository(tmp_path / "activity.db")
    repository.start_session(StoredSession("session-1", _timestamp(), None))
    repository.start_session(StoredSession("session-2", _timestamp(100), None))
    repository.start_session(StoredSession("session-3", _timestamp(200), None))

    for session_id, offset in (("session-1", 0), ("session-2", 100), ("session-1", 20), ("session-3", 200)):
        repository.insert_activity(
            ActivitySegment(
                session_id=session_id,
                started_at=_timestamp(offset),
                ended_at=_timestamp(offset + 5),
                application=f"App-{session_id}",
                process_name=f"{session_id}.exe",
                window_title=session_id,
                duration_seconds=5.0,
            )
        )

    session_1_activities = repository.list_activities_for_session("session-1")
    session_2_activities = repository.list_activities_for_session("session-2")
    session_3_activities = repository.list_activities_for_session("session-3")

    assert len(session_1_activities) == 2
    assert len(session_2_activities) == 1
    assert len(session_3_activities) == 1
    assert all(activity.session_id == "session-1" for activity in session_1_activities)


def test_list_activities_for_session_returns_empty_list_for_session_with_no_activities(tmp_path) -> None:
    repository = ActivityRepository(tmp_path / "activity.db")
    repository.start_session(StoredSession("session-1", _timestamp(), None))

    activities = repository.list_activities_for_session("session-1")

    assert activities == []


def test_list_activities_for_session_returns_empty_list_for_nonexistent_session(tmp_path) -> None:
    repository = ActivityRepository(tmp_path / "activity.db")

    activities = repository.list_activities_for_session("does-not-exist")

    assert activities == []


def test_list_activities_for_session_preserves_chronological_ordering(tmp_path) -> None:
    repository = ActivityRepository(tmp_path / "activity.db")
    repository.start_session(StoredSession("session-1", _timestamp(), None))
    for application, start_seconds, end_seconds in (
        ("Second", 20, 25),
        ("First", 0, 10),
        ("Third", 40, 45),
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

    activities = repository.list_activities_for_session("session-1")

    assert [activity.application for activity in activities] == ["First", "Second", "Third"]


def test_list_activities_for_session_uses_parameterized_query_safely(tmp_path) -> None:
    """A session_id containing SQL-special characters must be treated as literal data, not SQL."""
    repository = ActivityRepository(tmp_path / "activity.db")
    malicious_id = "session-1' OR '1'='1"
    repository.start_session(StoredSession("session-1", _timestamp(), None))
    repository.insert_activity(
        ActivitySegment(
            session_id="session-1",
            started_at=_timestamp(),
            ended_at=_timestamp(10),
            application="Code",
            process_name="Code.exe",
            window_title="Code",
            duration_seconds=10.0,
        )
    )

    result = repository.list_activities_for_session(malicious_id)

    assert result == []  # no injection; the literal string simply matches nothing


def test_list_activities_for_session_does_not_affect_list_activities(tmp_path) -> None:
    """Adding the new method must not change the pre-existing unfiltered method's behavior."""
    repository = ActivityRepository(tmp_path / "activity.db")
    repository.start_session(StoredSession("session-1", _timestamp(), None))
    repository.start_session(StoredSession("session-2", _timestamp(100), None))
    repository.insert_activity(
        ActivitySegment(
            session_id="session-1", started_at=_timestamp(0), ended_at=_timestamp(10),
            application="A", process_name="a.exe", window_title="A", duration_seconds=10.0,
        )
    )
    repository.insert_activity(
        ActivitySegment(
            session_id="session-2", started_at=_timestamp(100), ended_at=_timestamp(110),
            application="B", process_name="b.exe", window_title="B", duration_seconds=10.0,
        )
    )

    all_activities = repository.list_activities()

    assert len(all_activities) == 2
