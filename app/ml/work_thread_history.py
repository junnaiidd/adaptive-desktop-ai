"""
Work Thread history: a deterministic read model composed from existing durable data.

A Work Thread is a persistent, user-defined semantic entity. This module answers "what is the history of
this Work Thread?" by reading sources that already exist; it stores nothing and adds no table or column.

What this module IS
-------------------
A read-only composition layer. Given a Work Thread id it asks each existing store for what that store owns
and returns one immutable ``WorkThreadHistory``:

  =========================  ===========================================  ==================================
  Kind of fact               Fields                                       Source of truth
  =========================  ===========================================  ==================================
  user-confirmed             associated_observation_count,                WorkThreadStore
                             last_associated_at                           (``work_thread_observations``)
  inferred                   context_history, last_context_observed_at    ContextObservationStore
  observed                   session_count, last_activity_at,             ContextObservationStore (which
                             recent_applications                          sessions) + ActivityRepository
                                                                          (what happened in them)
  product entities           open_task_count, completed_task_count        TaskStore
                             latest_workspace_snapshot_at                 WorkspaceSnapshotStore
  =========================  ===========================================  ==================================

The three kinds of fact are deliberately never merged. "The classifier said Software Development" (inferred),
"the user linked it to this thread" (user-confirmed) and "VS Code was in the foreground until 18:42"
(observed) have separate fields and separate timestamps.

Guarantees
----------
- Deterministic: the same persisted data always yields an equal result; every ordering has an explicit
  tie-break. Nothing is cached, so a reader always reflects the current stores.
- Persisted-only: it is rebuilt from the database on every call and therefore identical after a restart.
- No guessing: a source that is not configured, or has nothing to say, yields ``None`` (scalar values) or an
  empty tuple (collections). Counts are never estimated, and nameless activity is never given a name.
- UTC: every timestamp is passed through exactly as the owning store returns it (timezone-aware UTC).

What this module is NOT
-----------------------
- Not a matcher: it never suggests, scores, ranks or predicts which Work Thread an activity belongs to.
- Not an inference/ML component: no model, embeddings or LLM; it only re-reads recorded inferences.
- Not a writer: it never creates, changes or removes associations, threads, tasks or snapshots.
- Not a store: Work Thread rows are unchanged; no history is copied into them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from app.ml.work_thread_store import WorkThread

if TYPE_CHECKING:  # contracts only; the reader receives already-constructed stores
    from app.database.activity_repository import ActivityRepository
    from app.ml.context_observation_store import ContextObservationStore
    from app.ml.task_store import TaskStore
    from app.ml.work_thread_store import WorkThreadStore
    from app.workspace.workspace_snapshot_store import WorkspaceSnapshotStore


DEFAULT_RECENT_APPLICATION_LIMIT = 5


@dataclass(frozen=True, slots=True)
class ContextHistoryEntry:
    """One *inferred* context label seen among a Work Thread's linked observations."""

    label: str
    observation_count: int
    first_observed_at: datetime  # UTC
    last_observed_at: datetime  # UTC


@dataclass(frozen=True, slots=True)
class RecentApplication:
    """One *observed* application used in a Work Thread's linked sessions."""

    name: str
    last_used_at: datetime  # UTC; end of its most recent activity segment


@dataclass(frozen=True, slots=True)
class WorkThreadHistory:
    """The persisted history of one Work Thread. See the module docstring for each field's source."""

    work_thread: WorkThread

    # user-confirmed (explicit associations the user made)
    associated_observation_count: int
    last_associated_at: datetime | None

    # inferred (what the classifier recorded for the linked observations)
    context_history: tuple[ContextHistoryEntry, ...]
    last_context_observed_at: datetime | None

    # observed (what actually happened in the linked sessions)
    session_count: int
    last_activity_at: datetime | None
    recent_applications: tuple[RecentApplication, ...]

    # product entities; ``None`` means the owning store is not configured (unknown, not zero)
    open_task_count: int | None
    completed_task_count: int | None
    latest_workspace_snapshot_at: datetime | None

    @property
    def has_workspace_snapshot(self) -> bool:
        return self.latest_workspace_snapshot_at is not None


class WorkThreadHistoryReader:
    """Compose a ``WorkThreadHistory`` from the existing stores. Read-only and stateless between calls."""

    def __init__(
        self,
        work_thread_store: WorkThreadStore,
        context_observation_store: ContextObservationStore,
        *,
        task_store: TaskStore | None = None,
        activity_repository: ActivityRepository | None = None,
        workspace_snapshot_store: WorkspaceSnapshotStore | None = None,
        recent_application_limit: int = DEFAULT_RECENT_APPLICATION_LIMIT,
    ) -> None:
        if recent_application_limit <= 0:
            raise ValueError("recent_application_limit must be positive.")
        self._threads = work_thread_store
        self._observations = context_observation_store
        self._tasks = task_store
        self._activity = activity_repository
        self._snapshots = workspace_snapshot_store
        self._recent_application_limit = recent_application_limit

    def get_history(self, work_thread_id: int) -> WorkThreadHistory | None:
        """Return the history of one Work Thread, or ``None`` if no such Work Thread exists."""
        thread = self._threads.get_work_thread(work_thread_id)
        if thread is None:
            return None

        # user-confirmed: the explicit associations, in the store's deterministic order.
        associations = self._threads.list_observations_for_work_thread(work_thread_id)
        last_associated_at = max((a.associated_at for a in associations), default=None)

        # inferred: each distinct linked observation, read back from its own store.
        observation_ids = list(dict.fromkeys(a.observation_id for a in associations))
        observations = [
            observation
            for observation in (self._observations.get_observation(i) for i in observation_ids)
            if observation is not None
        ]
        context_history = _context_history(observations)
        last_context_observed_at = max((o.observed_at for o in observations), default=None)

        # observed: the distinct sessions behind those observations, and what ran inside them.
        session_ids = sorted({o.session_id for o in observations})
        last_activity_at, recent_applications = self._observed_activity(session_ids)

        return WorkThreadHistory(
            work_thread=thread,
            associated_observation_count=len(observations),
            last_associated_at=last_associated_at,
            context_history=context_history,
            last_context_observed_at=last_context_observed_at,
            session_count=len(session_ids),
            last_activity_at=last_activity_at,
            recent_applications=recent_applications,
            open_task_count=self._task_count(work_thread_id, done=False),
            completed_task_count=self._task_count(work_thread_id, done=True),
            latest_workspace_snapshot_at=self._latest_snapshot_at(work_thread_id),
        )

    def _observed_activity(
        self, session_ids: list[str]
    ) -> tuple[datetime | None, tuple[RecentApplication, ...]]:
        if self._activity is None:
            return None, ()

        last_activity_at: datetime | None = None
        last_used: dict[str, datetime] = {}
        for session_id in session_ids:
            for segment in self._activity.list_activities_for_session(session_id):
                if last_activity_at is None or segment.ended_at > last_activity_at:
                    last_activity_at = segment.ended_at
                name = _application_name(segment.application, segment.process_name)
                if name is not None and (name not in last_used or segment.ended_at > last_used[name]):
                    last_used[name] = segment.ended_at

        # Most recent first; equal times are ordered by name so the result never depends on iteration order.
        ordered = sorted(last_used.items(), key=lambda item: item[0])
        ordered.sort(key=lambda item: item[1], reverse=True)
        recent = tuple(
            RecentApplication(name=name, last_used_at=used_at)
            for name, used_at in ordered[: self._recent_application_limit]
        )
        return last_activity_at, recent

    def _task_count(self, work_thread_id: int, *, done: bool) -> int | None:
        if self._tasks is None:
            return None
        return sum(1 for task in self._tasks.list_tasks_for_work_thread(work_thread_id) if task.is_done is done)

    def _latest_snapshot_at(self, work_thread_id: int) -> datetime | None:
        if self._snapshots is None:
            return None
        snapshots = self._snapshots.list_snapshots_for_work_thread(work_thread_id)
        return max((s.captured_at for s in snapshots), default=None)


def _context_history(observations: list) -> tuple[ContextHistoryEntry, ...]:
    """Group recorded inferences by label; order by first appearance, then label."""
    grouped: dict[str, list[datetime]] = {}
    for observation in observations:
        grouped.setdefault(observation.predicted_label, []).append(observation.observed_at)

    entries = [
        ContextHistoryEntry(
            label=label,
            observation_count=len(times),
            first_observed_at=min(times),
            last_observed_at=max(times),
        )
        for label, times in grouped.items()
    ]
    entries.sort(key=lambda entry: (entry.first_observed_at, entry.label))
    return tuple(entries)


def _application_name(application: str | None, process_name: str | None) -> str | None:
    """The display name actually recorded for a segment; ``None`` when nothing was recorded."""
    for candidate in (application, process_name):
        if candidate is not None and candidate.strip():
            return candidate.strip()
    return None
