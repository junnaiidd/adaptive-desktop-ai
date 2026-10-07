"""
Tests for the Work Thread history read model (`app/ml/work_thread_history.py`).

The read model composes a deterministic, persisted history for one Work Thread from sources that already
exist (WorkThreadStore associations, ContextObservationStore, ActivityRepository, TaskStore,
WorkspaceSnapshotStore). It stores nothing itself.

Three kinds of fact stay distinct in the result:
  - observed        what happened on the desktop       (sessions / activity segments)
  - inferred        what context it appeared to be     (context observations)
  - user-confirmed  what the user explicitly linked    (work_thread_observations)
"""

from __future__ import annotations

import ast
import dataclasses
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.database.activity_repository import ActivityRepository, ActivitySegment, StoredSession
from app.ml.context_observation_store import ContextObservationStore
from app.ml.task_store import TaskStore
from app.ml.work_thread_history import (
    ContextHistoryEntry,
    RecentApplication,
    WorkThreadHistory,
    WorkThreadHistoryReader,
)
from app.ml.work_thread_store import (
    WorkThreadHasAssociationsError,
    WorkThreadHasTasksError,
    WorkThreadStore,
)
from app.workspace.workspace_snapshot_store import CapturedForegroundWindow, WorkspaceSnapshotStore

BASE = datetime(2026, 9, 20, 9, 0, tzinfo=timezone.utc)


def at(hours: float = 0.0) -> datetime:
    return BASE + timedelta(hours=hours)


class Env:
    """All stores pointed at ONE database file, exactly like the running application."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.repo = ActivityRepository(path)
        self.observations = ContextObservationStore(path)
        self.threads = WorkThreadStore(path)
        self.tasks = TaskStore(path)
        self.snapshots = WorkspaceSnapshotStore(path)

    def reader(self, **overrides) -> WorkThreadHistoryReader:
        kwargs = dict(
            task_store=self.tasks,
            activity_repository=self.repo,
            workspace_snapshot_store=self.snapshots,
        )
        kwargs.update(overrides)
        return WorkThreadHistoryReader(self.threads, self.observations, **kwargs)

    def session(self, session_id: str, start_hours: float, *segments) -> str:
        """segments: (application, process_name, start_hours, end_hours)"""
        self.repo.start_session(StoredSession(id=session_id, started_at=at(start_hours), ended_at=None))
        for application, process, s_h, e_h in segments:
            self.repo.insert_activity(
                ActivitySegment(
                    session_id=session_id,
                    started_at=at(s_h),
                    ended_at=at(e_h),
                    application=application,
                    process_name=process,
                    window_title=None,
                    duration_seconds=(e_h - s_h) * 3600,
                )
            )
        return session_id

    def observe(self, session_id: str, label: str, observed_hours: float):
        return self.observations.record_observation(
            session_id, label, {label: 1.0}, "test-model", observed_at=at(observed_hours)
        )

    def link(self, thread_id: int, observation, associated_hours: float):
        return self.threads.associate_observation(thread_id, observation.id, associated_at=at(associated_hours))

    def new_thread(self, name: str = "Adaptive Desktop AI"):
        return self.threads.create_work_thread(name, created_at=at(-24))


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path / "activity.db")


# ============================================================================
# 1. Work Thread with nothing attached
# ============================================================================


def test_empty_thread_reports_zeros_and_missing_values_not_guesses(env: Env) -> None:
    thread = env.new_thread()

    history = env.reader().get_history(thread.id)

    assert history.work_thread == thread
    assert history.associated_observation_count == 0
    assert history.session_count == 0
    assert history.context_history == ()
    assert history.recent_applications == ()
    assert history.open_task_count == 0 and history.completed_task_count == 0
    # Missing stays missing: no timestamps are invented.
    assert history.last_associated_at is None
    assert history.last_context_observed_at is None
    assert history.last_activity_at is None
    assert history.latest_workspace_snapshot_at is None
    assert history.has_workspace_snapshot is False


def test_unknown_thread_returns_none(env: Env) -> None:
    assert env.reader().get_history(999) is None


# ============================================================================
# 2 / 3. Associated observations, across several sessions
# ============================================================================


def test_single_associated_observation(env: Env) -> None:
    thread = env.new_thread()
    env.session("s1", 0, ("VS Code", "code.exe", 0, 1))
    obs = env.observe("s1", "Software Development", 1.1)
    env.link(thread.id, obs, 2)

    history = env.reader().get_history(thread.id)

    assert history.associated_observation_count == 1
    assert history.session_count == 1
    assert history.context_history == (
        ContextHistoryEntry(
            label="Software Development",
            observation_count=1,
            first_observed_at=at(1.1),
            last_observed_at=at(1.1),
        ),
    )
    assert history.last_associated_at == at(2)
    assert history.last_context_observed_at == at(1.1)


def test_multiple_sessions_and_contexts_are_aggregated(env: Env) -> None:
    thread = env.new_thread()
    for index, (label, hours) in enumerate(
        [("Software Development", 1), ("ML / Model Development", 5), ("Software Development", 9), ("Browsing", 13)]
    ):
        sid = env.session(f"s{index}", hours - 1, ("App", "app.exe", hours - 1, hours))
        env.link(thread.id, env.observe(sid, label, hours + 0.1), hours + 0.2)

    history = env.reader().get_history(thread.id)

    assert history.associated_observation_count == 4
    assert history.session_count == 4
    by_label = {entry.label: entry for entry in history.context_history}
    assert by_label["Software Development"].observation_count == 2
    assert by_label["Software Development"].first_observed_at == at(1.1)
    assert by_label["Software Development"].last_observed_at == at(9.1)
    assert by_label["ML / Model Development"].observation_count == 1
    assert by_label["Browsing"].observation_count == 1


def test_two_observations_in_one_session_count_as_one_session(env: Env) -> None:
    thread = env.new_thread()
    sid = env.session("s1", 0, ("VS Code", "code.exe", 0, 2))
    env.link(thread.id, env.observe(sid, "Software Development", 1), 3)
    env.link(thread.id, env.observe(sid, "ML / Model Development", 2), 3)

    history = env.reader().get_history(thread.id)

    assert history.associated_observation_count == 2
    assert history.session_count == 1


def test_duplicate_association_of_one_observation_is_counted_once(env: Env) -> None:
    thread = env.new_thread()
    sid = env.session("s1", 0, ("VS Code", "code.exe", 0, 1))
    obs = env.observe(sid, "Software Development", 1)
    env.link(thread.id, obs, 2)
    env.link(thread.id, obs, 3)  # the store permits it; the history must not double-count

    history = env.reader().get_history(thread.id)

    assert history.associated_observation_count == 1
    assert history.context_history[0].observation_count == 1
    assert history.last_associated_at == at(3)


# ============================================================================
# 4. Open / completed tasks
# ============================================================================


def test_open_and_completed_task_counts(env: Env) -> None:
    thread = env.new_thread()
    done = [env.tasks.create_task(thread.id, f"done {i}", created_at=at(i)) for i in range(5)]
    for task in done:
        env.tasks.complete_task(task.id)
    for i in range(3):
        env.tasks.create_task(thread.id, f"open {i}", created_at=at(10 + i))

    history = env.reader().get_history(thread.id)

    assert history.open_task_count == 3
    assert history.completed_task_count == 5


def test_task_counts_are_missing_when_no_task_store_is_configured(env: Env) -> None:
    thread = env.new_thread()
    env.tasks.create_task(thread.id, "x", created_at=at(0))

    history = env.reader(task_store=None).get_history(thread.id)

    assert history.open_task_count is None and history.completed_task_count is None


# ============================================================================
# 5. Deterministic UTC ordering
# ============================================================================


def test_context_history_is_ordered_by_first_observation_then_label(env: Env) -> None:
    thread = env.new_thread()
    # Observed out of order relative to association order.
    for index, (label, hours) in enumerate([("Zeta", 8), ("Alpha", 3), ("Mid", 3), ("Beta", 5)]):
        sid = env.session(f"s{index}", hours - 1, ("A", "a.exe", hours - 1, hours))
        env.link(thread.id, env.observe(sid, label, hours), 20 - index)

    labels = [entry.label for entry in env.reader().get_history(thread.id).context_history]

    assert labels == ["Alpha", "Mid", "Beta", "Zeta"]  # first_observed asc, ties by label


def test_recent_applications_are_most_recent_first_and_deduplicated(env: Env) -> None:
    thread = env.new_thread()
    sid1 = env.session("s1", 0, ("VS Code", "code.exe", 0, 1), ("Chrome", "chrome.exe", 1, 2))
    sid2 = env.session("s2", 3, ("Python", "python.exe", 3, 4), ("VS Code", "code.exe", 4, 5))
    env.link(thread.id, env.observe(sid1, "A", 2.1), 6)
    env.link(thread.id, env.observe(sid2, "A", 5.1), 6)

    apps = env.reader().get_history(thread.id).recent_applications

    assert apps == (
        RecentApplication(name="VS Code", last_used_at=at(5)),
        RecentApplication(name="Python", last_used_at=at(4)),
        RecentApplication(name="Chrome", last_used_at=at(2)),
    )


def test_recent_applications_with_equal_times_are_ordered_by_name(env: Env) -> None:
    thread = env.new_thread()
    # Inserted in a deliberately non-alphabetical order, all ending at the same instant.
    sid = env.session("s1", 0, ("Zed", "z.exe", 0, 1), ("Alpha", "a.exe", 0, 1), ("Mid", "m.exe", 0, 1))
    env.link(thread.id, env.observe(sid, "A", 2), 3)

    apps = env.reader().get_history(thread.id).recent_applications

    assert [a.name for a in apps] == ["Alpha", "Mid", "Zed"]


def test_recent_application_limit_is_respected(env: Env) -> None:
    thread = env.new_thread()
    sid = env.session("s1", 0, *[(f"App{i}", f"a{i}.exe", i, i + 1) for i in range(8)])
    env.link(thread.id, env.observe(sid, "A", 9), 10)

    apps = env.reader(recent_application_limit=3).get_history(thread.id).recent_applications

    assert [a.name for a in apps] == ["App7", "App6", "App5"]


def test_timestamps_are_utc_aware(env: Env) -> None:
    thread = env.new_thread()
    sid = env.session("s1", 0, ("VS Code", "code.exe", 0, 1))
    env.link(thread.id, env.observe(sid, "A", 1), 2)
    env.snapshots.create_snapshot(
        thread.id, CapturedForegroundWindow("Notepad", "notepad.exe", r"C:\Windows\notepad.exe", "n"), captured_at=at(3)
    )

    history = env.reader().get_history(thread.id)

    for value in (
        history.work_thread.created_at,
        history.last_associated_at,
        history.last_context_observed_at,
        history.last_activity_at,
        history.latest_workspace_snapshot_at,
        history.context_history[0].first_observed_at,
        history.recent_applications[0].last_used_at,
    ):
        assert value.tzinfo is not None and value.utcoffset() == timedelta(0)


# ============================================================================
# 6. Missing optional data
# ============================================================================


def test_application_name_falls_back_to_process_and_unnamed_segments_are_skipped(env: Env) -> None:
    thread = env.new_thread()
    sid = env.session("s1", 0, (None, "tool.exe", 0, 1), (None, None, 1, 2), ("  ", None, 2, 3))
    env.link(thread.id, env.observe(sid, "A", 3), 4)

    apps = env.reader().get_history(thread.id).recent_applications

    assert [a.name for a in apps] == ["tool.exe"]  # nothing invented for nameless segments


def test_without_activity_repository_observed_activity_is_missing_but_the_rest_is_present(env: Env) -> None:
    thread = env.new_thread()
    sid = env.session("s1", 0, ("VS Code", "code.exe", 0, 1))
    env.link(thread.id, env.observe(sid, "Software Development", 1), 2)

    history = env.reader(activity_repository=None).get_history(thread.id)

    assert history.session_count == 1
    assert history.associated_observation_count == 1
    assert history.last_activity_at is None
    assert history.recent_applications == ()


def test_session_without_activity_segments_yields_no_activity_but_keeps_the_inference(env: Env) -> None:
    thread = env.new_thread()
    sid = env.session("s1", 0)  # a session with no segments
    env.link(thread.id, env.observe(sid, "Software Development", 1), 2)

    history = env.reader().get_history(thread.id)

    assert history.session_count == 1
    assert history.last_activity_at is None
    assert history.recent_applications == ()
    assert history.context_history[0].label == "Software Development"


def test_without_snapshot_store_there_is_no_snapshot(env: Env) -> None:
    thread = env.new_thread()

    history = env.reader(workspace_snapshot_store=None).get_history(thread.id)

    assert history.latest_workspace_snapshot_at is None and history.has_workspace_snapshot is False


def test_workspace_snapshot_availability_reflects_the_latest_snapshot(env: Env) -> None:
    thread = env.new_thread()
    window = CapturedForegroundWindow("Notepad", "notepad.exe", r"C:\Windows\notepad.exe", "notes")
    env.snapshots.create_snapshot(thread.id, window, captured_at=at(1))
    env.snapshots.create_snapshot(thread.id, window, captured_at=at(4))

    history = env.reader().get_history(thread.id)

    assert history.has_workspace_snapshot is True
    assert history.latest_workspace_snapshot_at == at(4)


# ============================================================================
# Observed vs inferred vs user-confirmed stay separate
# ============================================================================


def test_last_activity_is_observed_time_not_inference_or_association_time(env: Env) -> None:
    thread = env.new_thread()
    sid = env.session("s1", 0, ("VS Code", "code.exe", 0, 1))  # activity ended at hour 1
    env.link(thread.id, env.observe(sid, "Software Development", 2), 10)  # inferred at 2, linked at 10

    history = env.reader().get_history(thread.id)

    assert history.last_activity_at == at(1)  # observed
    assert history.last_context_observed_at == at(2)  # inferred
    assert history.last_associated_at == at(10)  # user-confirmed
    assert len({history.last_activity_at, history.last_context_observed_at, history.last_associated_at}) == 3


def test_history_exposes_no_confidence_prediction_or_matching_fields() -> None:
    fields = {f.name for f in dataclasses.fields(WorkThreadHistory)}
    for forbidden in ("confidence", "score", "similarity", "predicted_thread", "suggested", "probability", "deadline"):
        assert not any(forbidden in name for name in fields)


# ============================================================================
# 7. Isolation between Work Threads
# ============================================================================


def test_threads_are_isolated(env: Env) -> None:
    first, second = env.new_thread("First"), env.new_thread("Second")
    s1 = env.session("s1", 0, ("VS Code", "code.exe", 0, 1))
    s2 = env.session("s2", 2, ("Chrome", "chrome.exe", 2, 3))
    env.link(first.id, env.observe(s1, "Software Development", 1), 5)
    env.link(second.id, env.observe(s2, "Browsing", 3), 6)
    env.tasks.create_task(first.id, "first task", created_at=at(0))
    env.snapshots.create_snapshot(
        second.id, CapturedForegroundWindow("Notepad", "notepad.exe", r"C:\Windows\notepad.exe", None), captured_at=at(7)
    )

    one, two = env.reader().get_history(first.id), env.reader().get_history(second.id)

    assert [e.label for e in one.context_history] == ["Software Development"]
    assert [e.label for e in two.context_history] == ["Browsing"]
    assert [a.name for a in one.recent_applications] == ["VS Code"]
    assert [a.name for a in two.recent_applications] == ["Chrome"]
    assert one.open_task_count == 1 and two.open_task_count == 0
    assert one.has_workspace_snapshot is False and two.has_workspace_snapshot is True


def test_one_observation_linked_to_two_threads_appears_in_both(env: Env) -> None:
    first, second = env.new_thread("First"), env.new_thread("Second")
    sid = env.session("s1", 0, ("VS Code", "code.exe", 0, 1))
    obs = env.observe(sid, "Software Development", 1)
    env.link(first.id, obs, 2)
    env.link(second.id, obs, 3)

    assert env.reader().get_history(first.id).associated_observation_count == 1
    assert env.reader().get_history(second.id).associated_observation_count == 1


# ============================================================================
# 8. Restart-equivalent behaviour (fresh store instances on the same file)
# ============================================================================


def _populate(env: Env):
    thread = env.new_thread()
    s1 = env.session("s1", 0, ("VS Code", "code.exe", 0, 1), ("Chrome", "chrome.exe", 1, 2))
    s2 = env.session("s2", 5, ("Python", "python.exe", 5, 6))
    env.link(thread.id, env.observe(s1, "Software Development", 2.1), 7)
    env.link(thread.id, env.observe(s2, "ML / Model Development", 6.1), 8)
    env.tasks.create_task(thread.id, "open", created_at=at(1))
    env.tasks.complete_task(env.tasks.create_task(thread.id, "done", created_at=at(2)).id)
    env.snapshots.create_snapshot(
        thread.id, CapturedForegroundWindow("Notepad", "notepad.exe", r"C:\Windows\notepad.exe", "n"), captured_at=at(9)
    )
    return thread


def test_history_is_identical_after_a_restart(tmp_path: Path) -> None:
    first_run = Env(tmp_path / "activity.db")
    thread = _populate(first_run)
    before = first_run.reader().get_history(thread.id)

    restarted = Env(tmp_path / "activity.db")  # brand-new store/repository instances, same file
    after = restarted.reader().get_history(thread.id)

    assert after == before
    assert after.session_count == 2 and after.associated_observation_count == 2


def test_history_is_deterministic_across_repeated_reads(env: Env) -> None:
    thread = _populate(env)

    assert env.reader().get_history(thread.id) == env.reader().get_history(thread.id)


# ============================================================================
# 9. No duplicated source data, and the read model never writes
# ============================================================================


def _schema_and_counts(path: Path):
    connection = sqlite3.connect(path)
    try:
        tables = sorted(r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'"))
        counts = {t: connection.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in tables}
        columns = {t: [r[1] for r in connection.execute(f'PRAGMA table_info("{t}")')] for t in tables}
    finally:
        connection.close()
    return tables, counts, columns


def test_reading_history_never_changes_the_database(env: Env) -> None:
    thread = _populate(env)
    before = _schema_and_counts(env.path)

    reader = env.reader()
    reader.get_history(thread.id)
    reader.get_history(thread.id)

    assert _schema_and_counts(env.path) == before


def test_work_thread_tables_hold_no_duplicated_history(env: Env) -> None:
    _populate(env)
    _tables, _counts, columns = _schema_and_counts(env.path)

    assert columns["work_threads"] == ["id", "name", "created_at_utc"]
    assert columns["work_thread_observations"] == ["id", "work_thread_id", "observation_id", "associated_at_utc"]
    assert "history" not in " ".join(_tables)


def test_history_follows_source_data_with_nothing_cached(env: Env) -> None:
    thread = env.new_thread()
    reader = env.reader()
    assert reader.get_history(thread.id).open_task_count == 0

    env.tasks.create_task(thread.id, "new", created_at=at(1))  # change a source store directly

    assert reader.get_history(thread.id).open_task_count == 1  # same reader instance sees it


# ============================================================================
# 10. Existing deletion / association rules remain intact
# ============================================================================


def test_deletion_rules_are_unchanged_and_history_tracks_the_removal(env: Env) -> None:
    thread = env.new_thread()
    sid = env.session("s1", 0, ("VS Code", "code.exe", 0, 1))
    association = env.link(thread.id, env.observe(sid, "Software Development", 1), 2)
    task = env.tasks.create_task(thread.id, "t", created_at=at(0))
    env.reader().get_history(thread.id)  # reading must not affect any rule below

    with pytest.raises(WorkThreadHasAssociationsError):
        env.threads.delete_work_thread(thread.id)

    env.threads.remove_association(association.id)
    assert env.reader().get_history(thread.id).associated_observation_count == 0
    assert env.reader().get_history(thread.id).context_history == ()

    with pytest.raises(WorkThreadHasTasksError):
        env.threads.delete_work_thread(thread.id)

    env.tasks.delete_task(task.id)
    env.threads.delete_work_thread(thread.id)
    assert env.reader().get_history(thread.id) is None


def test_association_requires_existing_rows_as_before(env: Env) -> None:
    thread = env.new_thread()
    with pytest.raises(sqlite3.IntegrityError):
        env.threads.associate_observation(thread.id, 12345, associated_at=at(1))


# ============================================================================
# Architecture guards: a read model, nothing more
# ============================================================================


def test_module_stays_a_pure_read_model() -> None:
    source = (Path(__file__).resolve().parents[1] / "app" / "ml" / "work_thread_history.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0] if node.level == 0 else node.module)

    for forbidden in ("sklearn", "numpy", "pandas", "joblib", "torch", "PySide6", "win32gui", "psutil", "openai"):
        assert forbidden not in imported

    lowered = source.lower()
    for forbidden_sql in ("insert into", "update ", "delete from", "create table", "alter table"):
        assert forbidden_sql not in lowered  # composes store APIs; never writes SQL itself
    assert "sqlite3" not in imported  # uses store contracts instead of bypassing them
