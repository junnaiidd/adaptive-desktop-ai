"""Small, configurable session-boundary policy for observed desktop activity."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class Session:
    """One continuous period of meaningful observed desktop activity."""

    id: str
    started_at: datetime
    last_activity_at: datetime
    ended_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class SessionUpdate:
    """Describes a new session and, optionally, the session it closed."""

    current: Session
    started: bool
    closed: Session | None


class SessionManager:
    """Start a new session after a configurable gap between observations."""

    def __init__(self, inactivity_gap_seconds: float = 300.0) -> None:
        if inactivity_gap_seconds <= 0:
            raise ValueError("The inactivity gap must be positive.")
        self.inactivity_gap = timedelta(seconds=inactivity_gap_seconds)
        self._current: Session | None = None

    @property
    def current_session(self) -> Session | None:
        """Return the active session, if an observation has begun one."""
        return self._current

    def observe(self, observed_at: datetime) -> SessionUpdate:
        """Record an observation time and return any resulting session boundary."""
        if observed_at.tzinfo is None:
            raise ValueError("Observation timestamps must be timezone-aware.")

        if self._current is None:
            self._current = Session(str(uuid4()), observed_at, observed_at)
            return SessionUpdate(current=self._current, started=True, closed=None)

        if observed_at - self._current.last_activity_at > self.inactivity_gap:
            closed = replace(self._current, ended_at=self._current.last_activity_at)
            self._current = Session(str(uuid4()), observed_at, observed_at)
            return SessionUpdate(current=self._current, started=True, closed=closed)

        self._current = replace(self._current, last_activity_at=observed_at)
        return SessionUpdate(current=self._current, started=False, closed=None)

    def close(self) -> Session | None:
        """Close the active session at its last observed activity time."""
        if self._current is None:
            return None
        closed = replace(self._current, ended_at=self._current.last_activity_at)
        self._current = None
        return closed
