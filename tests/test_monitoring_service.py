"""Tests for segmentation and session handling without a real Windows desktop."""

from datetime import datetime, timedelta, timezone

import pytest

from app.core.activity_monitor import ActivityEvent
from app.core.monitoring_service import ActivityMonitoringService
from app.core.session_manager import SessionManager
from app.database.activity_repository import ActivityRepository


class SequenceObserver:
    def __init__(self, events: list[ActivityEvent | None]) -> None:
        self._events = iter(events)

    def get_current_activity(self) -> ActivityEvent | None:
        return next(self._events, None)


def _event(seconds: int, application: str = "Code", title: str = "project.py") -> ActivityEvent:
    return ActivityEvent(
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds),
        application=application,
        process_name=f"{application}.exe",
        window_title=title,
    )


def _service(tmp_path, events, *, poll_interval: float = 5.0, inactivity_gap: float = 300.0):
    repository = ActivityRepository(tmp_path / "activity.db")
    service = ActivityMonitoringService(
        SequenceObserver(events),
        repository,
        SessionManager(inactivity_gap),
        poll_interval,
    )
    return service, repository


def test_consecutive_identical_activities_are_merged(tmp_path) -> None:
    service, repository = _service(tmp_path, [_event(0), _event(5), _event(12)])

    service.poll_once()
    service.poll_once()
    service.poll_once()
    service.shutdown()

    activities = repository.list_activities()
    assert len(activities) == 1
    assert activities[0].duration_seconds == 12.0


def test_activity_change_creates_separate_segments(tmp_path) -> None:
    service, repository = _service(tmp_path, [_event(0), _event(10, "Browser", "Docs")])

    service.poll_once()
    service.poll_once()
    service.shutdown()

    activities = repository.list_activities()
    assert [activity.application for activity in activities] == ["Code", "Browser"]
    assert [activity.duration_seconds for activity in activities] == [10.0, 0.0]


def test_temporary_observation_failure_does_not_create_a_segment(tmp_path) -> None:
    service, repository = _service(tmp_path, [_event(0), None, _event(8)])

    service.poll_once()
    service.poll_once()
    service.poll_once()
    service.shutdown()

    activities = repository.list_activities()
    assert len(activities) == 1
    assert activities[0].duration_seconds == 8.0


def test_idle_gap_creates_a_new_session_and_closes_the_prior_segment(tmp_path) -> None:
    service, repository = _service(tmp_path, [_event(0), _event(10)], inactivity_gap=5.0)

    service.poll_once()
    service.poll_once()
    service.shutdown()

    activities = repository.list_activities()
    sessions = repository.list_sessions()
    assert len(activities) == 2
    assert activities[0].duration_seconds == 0.0
    assert activities[0].session_id != activities[1].session_id
    assert len(sessions) == 2
    assert all(session.ended_at is not None for session in sessions)


def test_polling_and_session_thresholds_are_configurable(tmp_path) -> None:
    service, _ = _service(tmp_path, [], poll_interval=2.5, inactivity_gap=42.0)

    assert service.poll_interval_seconds == 2.5
    assert service.session_manager.inactivity_gap == timedelta(seconds=42.0)
    with pytest.raises(ValueError):
        ActivityMonitoringService(SequenceObserver([]), ActivityRepository(tmp_path / "bad.db"), poll_interval_seconds=0)
    with pytest.raises(ValueError):
        SessionManager(0)


def test_shutdown_persists_the_open_segment_and_ends_the_session(tmp_path) -> None:
    service, repository = _service(tmp_path, [_event(0)])

    service.poll_once()
    service.request_stop()
    service.shutdown()

    assert len(repository.list_activities()) == 1
    assert repository.list_sessions()[0].ended_at is not None
