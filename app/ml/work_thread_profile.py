"""
Work Thread profile: the raw, timestamped evidence behind one Work Thread.

A Work Thread is a persistent, user-defined entity. Its *profile* is what the system has durably learned
about it from the user's own confirmations: one ``ThreadEvidence`` per linked session, each carrying when
it happened, what context was inferred, and which applications ran.

What this module IS
-------------------
A read-only composition of existing stores. Nothing is copied into a new table and nothing is stored here:

  - which sessions belong to a thread      <- WorkThreadStore (``work_thread_observations``; user-confirmed)
  - what context each session was labelled <- ContextObservationStore (an inference, kept as an inference)
  - which applications ran in each session <- ActivityRepository (sanitised application names only)

What this module is NOT
-----------------------
- Not a feature extractor and not a scorer: a profile holds NO derived statistics (no shares, no decayed
  values, no "current distribution"). Statistics are computed on demand by the feature extractor, relative to
  a decision time. That keeps a thread free to change over time: nothing about it is frozen into an immutable
  representation, so drift in the user's habits is reflected the next time it is read.
- Not a writer: it never creates, changes or removes any row.
- Not a matcher: it says nothing about which thread a new session belongs to.

Evidence semantics (the one-unit-per-session rule)
--------------------------------------------------
Several observations can exist for one session, and the store permits the same association to be recorded
twice. To avoid counting one session many times, the profile holds ONE unit per session per thread:

  - repeated associations of the same observation collapse to one; its ``associated_at`` is the EARLIEST
    confirmation, because that is when the evidence first became known to the system;
  - if a session has several linked observations, the most recent inference (latest ``observed_at``, then
    highest id) represents that session.

Privacy
-------
Only application names (``application``, else ``process_name``) are read from activity. Window titles are
never read into a profile.

Determinism
-----------
Threads are ordered by id and evidence by ``(observed_at, observation_id)``. Timestamps are timezone-aware
UTC exactly as the owning stores return them; a naive timestamp is rejected, matching the rest of the project.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Callable

from app.ml.work_thread_store import WorkThread

if TYPE_CHECKING:  # contracts only; the builder receives already-constructed stores
    from app.database.activity_repository import ActivityRepository
    from app.ml.context_observation_store import ContextObservation, ContextObservationStore
    from app.ml.work_thread_store import WorkThreadStore


def application_name(application: str | None, process_name: str | None) -> str | None:
    """The application name actually recorded for an activity segment, or ``None`` if nothing was recorded."""
    for candidate in (application, process_name):
        if candidate is not None and candidate.strip():
            return candidate.strip()
    return None


@dataclass(frozen=True, slots=True)
class ThreadEvidence:
    """One session that the user linked to a Work Thread (see the module docstring for the semantics)."""

    observation_id: int
    session_id: str
    context_label: str  # inferred: the label recorded for this session's observation
    observed_at: datetime  # UTC: when that inference was recorded (proxy for when the session happened)
    associated_at: datetime  # UTC: when the user first confirmed the link
    applications: tuple[str, ...]  # distinct, sorted application names; empty when unknown

    def __post_init__(self) -> None:
        for name in ("observed_at", "associated_at"):
            value = getattr(self, name)
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"ThreadEvidence.{name} must be timezone-aware (UTC); got a naive datetime.")


@dataclass(frozen=True, slots=True)
class WorkThreadProfile:
    """A Work Thread together with its timestamped evidence, oldest first. Holds no derived statistics."""

    work_thread: WorkThread
    evidence: tuple[ThreadEvidence, ...]


class WorkThreadProfileBuilder:
    """Derive ``WorkThreadProfile`` objects from the existing stores. Read-only and stateless between calls."""

    def __init__(
        self,
        work_thread_store: WorkThreadStore,
        context_observation_store: ContextObservationStore,
        activity_repository: ActivityRepository | None = None,
    ) -> None:
        self._threads = work_thread_store
        self._observations = context_observation_store
        self._activity = activity_repository

    def build_profiles(self) -> tuple[WorkThreadProfile, ...]:
        """A profile for EVERY Work Thread (including empty ones), ordered by thread id."""
        threads = sorted(self._threads.list_work_threads(), key=lambda t: t.id)
        if not threads:
            return ()
        by_id = {observation.id: observation for observation in self._observations.list_observations()}
        applications: dict[str, tuple[str, ...]] = {}
        return tuple(self._build(thread, by_id.get, applications) for thread in threads)

    def build_profile(self, work_thread_id: int) -> WorkThreadProfile | None:
        """The profile of one Work Thread, or ``None`` if it does not exist."""
        thread = self._threads.get_work_thread(work_thread_id)
        if thread is None:
            return None
        return self._build(thread, self._observations.get_observation, {})

    def _build(
        self,
        thread: WorkThread,
        lookup: Callable[[int], ContextObservation | None],
        applications_cache: dict[str, tuple[str, ...]],
    ) -> WorkThreadProfile:
        # earliest confirmation per distinct observation (repeated associations collapse to one)
        first_confirmed: dict[int, datetime] = {}
        for association in self._threads.list_observations_for_work_thread(thread.id):
            current = first_confirmed.get(association.observation_id)
            if current is None or association.associated_at < current:
                first_confirmed[association.observation_id] = association.associated_at

        # one representative observation per session: the most recent inference
        per_session: dict[str, ContextObservation] = {}
        for observation_id in sorted(first_confirmed):
            observation = lookup(observation_id)
            if observation is None:
                continue
            held = per_session.get(observation.session_id)
            if held is None or (observation.observed_at, observation.id) > (held.observed_at, held.id):
                per_session[observation.session_id] = observation

        evidence = [
            ThreadEvidence(
                observation_id=observation.id,
                session_id=observation.session_id,
                context_label=observation.predicted_label,
                observed_at=observation.observed_at,
                associated_at=first_confirmed[observation.id],
                applications=self._applications_for(observation.session_id, applications_cache),
            )
            for observation in per_session.values()
        ]
        evidence.sort(key=lambda unit: (unit.observed_at, unit.observation_id))
        return WorkThreadProfile(work_thread=thread, evidence=tuple(evidence))

    def _applications_for(self, session_id: str, cache: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
        if self._activity is None:
            return ()
        if session_id not in cache:
            names = {
                name
                for segment in self._activity.list_activities_for_session(session_id)
                if (name := application_name(segment.application, segment.process_name)) is not None
            }
            cache[session_id] = tuple(sorted(names))
        return cache[session_id]
