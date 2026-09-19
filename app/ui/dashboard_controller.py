"""Read-only dashboard data composition, independent of Qt widgets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from app.core.monitoring_service import ActivityMonitoringService
from app.database.activity_repository import ActivityRepository, StoredActivity, StoredSession


@dataclass(frozen=True, slots=True)
class TimelineItem:
    """A formatted activity segment ready for presentation."""

    timestamp: str
    application: str
    window_title: str
    duration: str


@dataclass(frozen=True, slots=True)
class DashboardSnapshot:
    """The UI-facing state assembled from monitoring and local persistence."""

    monitoring_running: bool
    polling_interval: str
    database_status: str
    current_application: str
    current_process: str
    current_title: str
    current_duration: str
    total_today: str
    segment_count: int
    session_count: int
    session_label: str
    session_duration: str
    session_activity_count: int
    timeline: tuple[TimelineItem, ...]


class DashboardController:
    """Compose local data for the Activity screen without UI or SQL code."""

    def __init__(self, repository: ActivityRepository, service: ActivityMonitoringService) -> None:
        self.repository = repository
        self.service = service

    def snapshot(self, monitoring_running: bool) -> DashboardSnapshot:
        """Build a current dashboard snapshot from the local repository and service."""
        activities = self.repository.list_activities()
        sessions = self.repository.list_sessions()
        now = datetime.now(timezone.utc)
        today_activities = [item for item in activities if item.started_at.date() == now.date()]
        today_sessions = [item for item in sessions if item.started_at.date() == now.date()]
        current_activity = self.service.current_activity
        current_session = self.service.session_manager.current_session

        return DashboardSnapshot(
            monitoring_running=monitoring_running,
            polling_interval=f"Every {format_duration(self.service.poll_interval_seconds)}",
            database_status=f"Ready · {self.repository.database_path.name}",
            current_application=_text_or_placeholder(
                current_activity.application if current_activity else None
            ),
            current_process=_text_or_placeholder(
                current_activity.process_name if current_activity else None
            ),
            current_title=_text_or_placeholder(
                current_activity.window_title if current_activity else None
            ),
            current_duration=format_duration(self.service.current_duration_seconds),
            total_today=format_duration(sum(item.duration_seconds for item in today_activities)),
            segment_count=len(today_activities),
            session_count=len(today_sessions),
            session_label=_session_label(current_session, sessions),
            session_duration=_session_duration(current_session, sessions, now),
            session_activity_count=_session_activity_count(current_session, activities),
            timeline=tuple(
                _timeline_item(item) for item in self.repository.list_recent_activities()
            ),
        )


def format_duration(seconds: float | None) -> str:
    """Format a known duration compactly; use an em dash for unavailable values."""
    if seconds is None:
        return "—"
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _timeline_item(activity: StoredActivity) -> TimelineItem:
    return TimelineItem(
        timestamp=activity.started_at.astimezone().strftime("%H:%M"),
        application=_text_or_placeholder(activity.application),
        window_title=_text_or_placeholder(activity.window_title),
        duration=format_duration(activity.duration_seconds),
    )


def _text_or_placeholder(value: str | None) -> str:
    return value if value else "—"


def _session_label(current_session, sessions: list[StoredSession]) -> str:
    if current_session is not None:
        return "Current session"
    if sessions:
        return "Most recent session"
    return "No sessions yet"


def _session_duration(current_session, sessions: list[StoredSession], now: datetime) -> str:
    if current_session is not None:
        return format_duration((now - current_session.started_at).total_seconds())
    if not sessions:
        return "—"
    session = sessions[-1]
    if session.ended_at is None:
        return "—"
    return format_duration((session.ended_at - session.started_at).total_seconds())


def _session_activity_count(current_session, activities: list[StoredActivity]) -> int:
    if current_session is None:
        return 0
    completed = sum(item.session_id == current_session.id for item in activities)
    return completed + (1 if current_session is not None else 0)
