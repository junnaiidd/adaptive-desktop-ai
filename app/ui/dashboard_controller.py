"""Read-only dashboard data composition, independent of Qt widgets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from app.core.monitoring_service import ActivityMonitoringService
from app.database.activity_repository import ActivityRepository, StoredActivity, StoredSession
from app.ml.context_observation_store import ContextObservation, ContextObservationStore
from app.ml.task_store import Task, TaskStore
from app.ml.work_thread_store import WorkThread, WorkThreadObservation, WorkThreadStore
from app.workspace.windows_workspace_adapter import ForegroundWindowProbe
from app.workspace.workspace_restoration import WorkspaceRestoreResult, WorkspaceRestorer
from app.workspace.workspace_snapshot_store import WorkspaceSnapshot, WorkspaceSnapshotStore


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
    latest_context_label: str
    latest_context_session: str
    latest_context_observed_at: str
    latest_context_observation_id: int | None = None
    work_threads: tuple[WorkThread, ...] = ()
    # Each element is (thread, open_tasks_for_thread). Only threads that have
    # at least one incomplete task are included, ordered by thread creation.
    open_tasks: tuple[tuple[WorkThread, tuple[Task, ...]], ...] = ()


class DashboardController:
    """Compose local data for the Activity screen without UI or SQL code."""

    def __init__(
        self,
        repository: ActivityRepository,
        service: ActivityMonitoringService,
        context_observation_store: ContextObservationStore | None = None,
        work_thread_store: WorkThreadStore | None = None,
        task_store: TaskStore | None = None,
        workspace_snapshot_store: WorkspaceSnapshotStore | None = None,
        workspace_probe: ForegroundWindowProbe | None = None,
        workspace_restorer: WorkspaceRestorer | None = None,
    ) -> None:
        self.repository = repository
        self.service = service
        self.context_observation_store = context_observation_store
        self.work_thread_store = work_thread_store
        self.task_store = task_store
        self.workspace_snapshot_store = workspace_snapshot_store
        self.workspace_probe = workspace_probe
        self.workspace_restorer = workspace_restorer

    def snapshot(self, monitoring_running: bool) -> DashboardSnapshot:
        """Build a current dashboard snapshot from the local repository and service."""
        activities = self.repository.list_activities()
        sessions = self.repository.list_sessions()
        now = datetime.now(timezone.utc)
        today_activities = [item for item in activities if item.started_at.date() == now.date()]
        today_sessions = [item for item in sessions if item.started_at.date() == now.date()]
        current_activity = self.service.current_activity
        current_session = self.service.session_manager.current_session
        latest_label, latest_session, latest_observed_at, latest_obs_id = self._latest_context_details()
        threads = self._work_threads()
        open_tasks = self._open_tasks_by_thread(threads)

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
            latest_context_label=latest_label,
            latest_context_session=latest_session,
            latest_context_observed_at=latest_observed_at,
            latest_context_observation_id=latest_obs_id,
            work_threads=threads,
            open_tasks=open_tasks,
        )

    def create_work_thread(self, name: str) -> WorkThread | None:
        """Create a new work thread using current UTC time, or None if store not configured."""
        if self.work_thread_store is None:
            return None
        return self.work_thread_store.create_work_thread(
            name, created_at=datetime.now(timezone.utc)
        )

    def associate_latest_context(self, work_thread_id: int) -> WorkThreadObservation | None:
        """
        Associate the latest displayed context observation with a selected work thread,
        or None if no store or no observation is available.
        """
        if self.work_thread_store is None:
            return None
        latest_obs = self._latest_observation()
        if latest_obs is None:
            return None
        return self.work_thread_store.associate_observation(
            work_thread_id,
            latest_obs.id,
            associated_at=datetime.now(timezone.utc),
        )

    def list_tasks_for_work_thread(self, work_thread_id: int) -> tuple[Task, ...]:
        """List tasks for a specific work thread, or empty tuple if no task store."""
        if self.task_store is None:
            return ()
        return tuple(self.task_store.list_tasks_for_work_thread(work_thread_id))

    def create_task(self, work_thread_id: int, title: str) -> Task | None:
        """Create a task in a specific work thread, or None if no task store."""
        if self.task_store is None:
            return None
        return self.task_store.create_task(
            work_thread_id, title, created_at=datetime.now(timezone.utc)
        )

    def toggle_task(self, task_id: int) -> Task | None:
        """Toggle task completion, or None if no task store."""
        if self.task_store is None:
            return None
        return self.task_store.toggle_task(task_id)

    def delete_task(self, task_id: int) -> None:
        """Delete a task if task store is configured."""
        if self.task_store is not None:
            self.task_store.delete_task(task_id)

    def list_workspace_snapshots(self, work_thread_id: int) -> tuple[WorkspaceSnapshot, ...]:
        """List explicitly captured snapshots for one Work Thread, or none when unconfigured."""
        if self.workspace_snapshot_store is None:
            return ()
        return tuple(self.workspace_snapshot_store.list_snapshots_for_work_thread(work_thread_id))

    def capture_workspace_snapshot(self, work_thread_id: int) -> WorkspaceSnapshot | None:
        """Capture the foreground app only after the UI receives an explicit user request."""
        if self.workspace_snapshot_store is None or self.workspace_probe is None:
            return None
        captured_window = self.workspace_probe.capture_foreground_window()
        return self.workspace_snapshot_store.create_snapshot(
            work_thread_id, captured_window, captured_at=datetime.now(timezone.utc)
        )

    def restore_workspace_snapshot(self, snapshot_id: int) -> WorkspaceRestoreResult | None:
        """Run a best-effort restore only after the UI confirms an explicit user request."""
        if self.workspace_snapshot_store is None or self.workspace_restorer is None:
            return None
        snapshot = self.workspace_snapshot_store.get_snapshot(snapshot_id)
        if snapshot is None:
            return None
        return self.workspace_restorer.restore(snapshot)

    def list_open_tasks(self) -> tuple[tuple[WorkThread, tuple[Task, ...]], ...]:
        """
        Return all incomplete tasks across every Work Thread, grouped by thread.

        Each element is ``(thread, open_tasks_for_thread)`` where
        ``open_tasks_for_thread`` contains only tasks whose ``is_done`` is
        ``False``.  Threads with no open tasks are omitted.  Order follows
        thread creation (oldest first).

        Returns an empty tuple when either store is absent.
        """
        return self._open_tasks_by_thread(self._work_threads())

    def _open_tasks_by_thread(
        self, threads: tuple[WorkThread, ...]
    ) -> tuple[tuple[WorkThread, tuple[Task, ...]], ...]:
        """Internal helper shared by ``snapshot`` and ``list_open_tasks``."""
        if self.task_store is None or not threads:
            return ()
        result: list[tuple[WorkThread, tuple[Task, ...]]] = []
        for thread in threads:
            open_tasks = tuple(
                t
                for t in self.task_store.list_tasks_for_work_thread(thread.id)
                if not t.is_done
            )
            if open_tasks:
                result.append((thread, open_tasks))
        return tuple(result)

    def _latest_observation(self) -> ContextObservation | None:
        if self.context_observation_store is None:
            return None
        recent = self.context_observation_store.list_recent_observations(limit=1)
        return recent[0] if recent else None

    def _latest_context(self) -> tuple[str, str, str]:
        """Backward-compatible tuple (label, session, observed_at)."""
        label, session, observed_at, _ = self._latest_context_details()
        return label, session, observed_at

    def _latest_context_details(self) -> tuple[str, str, str, int | None]:
        """Read the single most recent context observation details."""
        observation = self._latest_observation()
        if observation is None:
            return "—", "—", "—", None
        return (
            observation.predicted_label,
            observation.session_id,
            observation.observed_at.astimezone().strftime("%Y-%m-%d %H:%M"),
            observation.id,
        )

    def _work_threads(self) -> tuple[WorkThread, ...]:
        if self.work_thread_store is None:
            return ()
        return tuple(self.work_thread_store.list_work_threads())


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
