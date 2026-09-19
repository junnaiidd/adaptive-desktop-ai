"""Polling service that turns foreground observations into local activity segments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import threading
from typing import Protocol

from app.core.activity_monitor import ActivityEvent, WindowsActivityMonitor
from app.core.session_manager import Session, SessionManager
from app.database.activity_repository import ActivityRepository, ActivitySegment, StoredSession


class ActivityObserver(Protocol):
    """The narrow interface required from a foreground activity observer."""

    def get_current_activity(self) -> ActivityEvent | None: ...


@dataclass(slots=True)
class _OpenSegment:
    session_id: str
    started_at: datetime
    last_seen_at: datetime
    application: str | None
    process_name: str | None
    window_title: str | None

    @classmethod
    def from_event(cls, event: ActivityEvent, session_id: str) -> _OpenSegment:
        return cls(
            session_id=session_id,
            started_at=event.timestamp,
            last_seen_at=event.timestamp,
            application=event.application,
            process_name=event.process_name,
            window_title=event.window_title,
        )

    def matches(self, event: ActivityEvent) -> bool:
        return (self.application, self.process_name, self.window_title) == (
            event.application,
            event.process_name,
            event.window_title,
        )


class ActivityMonitoringService:
    """Poll foreground activity and persist only completed activity segments."""

    def __init__(
        self,
        observer: ActivityObserver,
        repository: ActivityRepository,
        session_manager: SessionManager | None = None,
        poll_interval_seconds: float = 5.0,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("The polling interval must be positive.")
        self.observer = observer
        self.repository = repository
        self.session_manager = session_manager or SessionManager()
        self.poll_interval_seconds = poll_interval_seconds
        self._open_segment: _OpenSegment | None = None
        self._stop_event = threading.Event()
        self._state_lock = threading.RLock()
        self._current_activity: ActivityEvent | None = None

    @property
    def current_activity(self) -> ActivityEvent | None:
        """Return the latest observed activity for read-only UI display."""
        with self._state_lock:
            return self._current_activity

    @property
    def current_duration_seconds(self) -> float | None:
        """Return the elapsed duration of the active segment, if one is open."""
        with self._state_lock:
            if self._open_segment is None:
                return None
            return max(
                0.0,
                (datetime.now(timezone.utc) - self._open_segment.started_at).total_seconds(),
            )

    def poll_once(self) -> None:
        """Process one observation; temporary observation failures are ignored."""
        event = self.observer.get_current_activity()
        if event is None:
            return

        with self._state_lock:
            self._current_activity = event
            update = self.session_manager.observe(event.timestamp)
            if update.closed is not None:
                self._finalize_open_segment(update.closed.ended_at)
                self._persist_session_end(update.closed)
            if update.started:
                self._persist_session_start(update.current)

            if self._open_segment is None:
                self._open_segment = _OpenSegment.from_event(event, update.current.id)
                return

            if self._open_segment.session_id != update.current.id or not self._open_segment.matches(event):
                self._finalize_open_segment(event.timestamp)
                self._open_segment = _OpenSegment.from_event(event, update.current.id)
                return

            self._open_segment.last_seen_at = event.timestamp

    def run(self) -> None:
        """Poll until :meth:`request_stop` is called or Ctrl+C interrupts the process."""
        self._stop_event.clear()
        try:
            while not self._stop_event.is_set():
                self.poll_once()
                self._stop_event.wait(self.poll_interval_seconds)
        finally:
            self.shutdown()

    def request_stop(self) -> None:
        """Ask a running service to stop after its current poll completes."""
        self._stop_event.set()

    def shutdown(self) -> None:
        """Persist any open segment and close its session without extending its duration."""
        with self._state_lock:
            self._finalize_open_segment()
            closed = self.session_manager.close()
            if closed is not None:
                self._persist_session_end(closed)
            self._current_activity = None

    def _finalize_open_segment(self, ended_at: datetime | None = None) -> None:
        if self._open_segment is None:
            return
        segment = self._open_segment
        end_time = ended_at or segment.last_seen_at
        duration_seconds = max(0.0, (end_time - segment.started_at).total_seconds())
        self.repository.insert_activity(
            ActivitySegment(
                session_id=segment.session_id,
                started_at=segment.started_at,
                ended_at=end_time,
                application=segment.application,
                process_name=segment.process_name,
                window_title=segment.window_title,
                duration_seconds=duration_seconds,
            )
        )
        self._open_segment = None

    def _persist_session_start(self, session: Session) -> None:
        self.repository.start_session(
            StoredSession(id=session.id, started_at=session.started_at, ended_at=None)
        )

    def _persist_session_end(self, session: Session) -> None:
        self.repository.end_session(
            StoredSession(id=session.id, started_at=session.started_at, ended_at=session.ended_at)
        )


def main() -> None:
    """Run local activity monitoring from a Windows terminal until Ctrl+C."""
    service = ActivityMonitoringService(WindowsActivityMonitor(), ActivityRepository())
    print("Local activity monitoring started. Press Ctrl+C to stop.")
    try:
        service.run()
    except KeyboardInterrupt:
        service.request_stop()
        service.shutdown()


if __name__ == "__main__":
    main()
