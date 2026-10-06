"""Tests for UI data composition that do not require PySide6 or a display."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app.core.activity_monitor import ActivityEvent, sanitize_window_title
from app.core.session_manager import Session
from app.database.activity_repository import ActivityRepository, ActivitySegment, StoredSession
from app.ui.dashboard_controller import DashboardController, format_duration


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def test_format_duration_handles_known_and_missing_values() -> None:
    assert format_duration(None) == "—"
    assert format_duration(59) == "0m"
    assert format_duration(3_661) == "1h 1m"


def test_dashboard_snapshot_uses_local_segments_and_sanitized_titles(tmp_path) -> None:
    repository = ActivityRepository(tmp_path / "activity.db")
    started_at = _now() - timedelta(minutes=10)
    repository.start_session(StoredSession("session-1", started_at, started_at + timedelta(minutes=3)))
    repository.insert_activity(
        ActivitySegment(
            session_id="session-1",
            started_at=started_at,
            ended_at=started_at + timedelta(minutes=3),
            application="Browser",
            process_name="browser.exe",
            window_title=sanitize_window_title("Reset password - Browser"),
            duration_seconds=180,
        )
    )
    service = SimpleNamespace(
        poll_interval_seconds=5.0,
        current_activity=None,
        current_duration_seconds=None,
        session_manager=SimpleNamespace(current_session=None),
    )

    snapshot = DashboardController(repository, service).snapshot(monitoring_running=False)

    assert snapshot.total_today == "3m"
    assert snapshot.segment_count == 1
    assert snapshot.session_count == 1
    assert snapshot.timeline[0].window_title == "[Sensitive window title hidden]"
    assert snapshot.current_application == "—"


def test_dashboard_snapshot_includes_live_activity_without_persisting_it(tmp_path) -> None:
    repository = ActivityRepository(tmp_path / "activity.db")
    started_at = _now() - timedelta(seconds=75)
    current_session = Session("current-session", started_at, _now())
    service = SimpleNamespace(
        poll_interval_seconds=2.5,
        current_activity=ActivityEvent(
            timestamp=_now(),
            application="Code",
            process_name="Code.exe",
            window_title="main.py - adaptive-desktop-ai",
        ),
        current_duration_seconds=75.0,
        session_manager=SimpleNamespace(current_session=current_session),
    )

    snapshot = DashboardController(repository, service).snapshot(monitoring_running=True)

    assert snapshot.monitoring_running is True
    assert snapshot.current_application == "Code"
    assert snapshot.current_duration == "1m"
    assert snapshot.session_label == "Current session"
    assert snapshot.session_activity_count == 1


def test_dashboard_snapshot_refreshes_with_newest_persisted_activity_first(tmp_path) -> None:
    repository = ActivityRepository(tmp_path / "activity.db")
    started_at = _now() - timedelta(minutes=10)
    repository.start_session(StoredSession("session-1", started_at, None))
    service = SimpleNamespace(
        poll_interval_seconds=5.0,
        current_activity=None,
        current_duration_seconds=None,
        session_manager=SimpleNamespace(current_session=None),
    )
    controller = DashboardController(repository, service)
    for application, offset in (("Older", 0), ("Latest", 60)):
        repository.insert_activity(
            ActivitySegment(
                session_id="session-1",
                started_at=started_at + timedelta(seconds=offset),
                ended_at=started_at + timedelta(seconds=offset + 30),
                application=application,
                process_name=f"{application}.exe",
                window_title=application,
                duration_seconds=30,
            )
        )

    snapshot = controller.snapshot(monitoring_running=True)

    assert [item.application for item in snapshot.timeline] == ["Latest", "Older"]


# ============================================================================
# Milestone A: latest-context fields (additive; existing tests above are
# unmodified and still construct DashboardController with two arguments)
# ============================================================================


def _service_stub() -> SimpleNamespace:
    return SimpleNamespace(
        poll_interval_seconds=5.0,
        current_activity=None,
        current_duration_seconds=None,
        session_manager=SimpleNamespace(current_session=None),
    )


def test_snapshot_shows_placeholder_when_no_observation_store_supplied(tmp_path) -> None:
    repository = ActivityRepository(tmp_path / "activity.db")

    snapshot = DashboardController(repository, _service_stub()).snapshot(monitoring_running=False)

    assert snapshot.latest_context_label == "—"
    assert snapshot.latest_context_session == "—"
    assert snapshot.latest_context_observed_at == "—"


def test_snapshot_shows_placeholder_when_store_supplied_but_empty(tmp_path) -> None:
    from app.ml.context_observation_store import ContextObservationStore

    repository = ActivityRepository(tmp_path / "activity.db")
    store = ContextObservationStore(tmp_path / "activity.db")

    snapshot = DashboardController(repository, _service_stub(), store).snapshot(monitoring_running=False)

    assert snapshot.latest_context_label == "—"


def test_snapshot_shows_the_most_recent_observation(tmp_path) -> None:
    from app.ml.context_observation_store import ContextObservationStore

    db_path = tmp_path / "activity.db"
    repository = ActivityRepository(db_path)
    started_at = _now() - timedelta(minutes=30)
    repository.start_session(StoredSession("session-1", started_at, started_at + timedelta(minutes=10)))
    store = ContextObservationStore(db_path)
    store.record_observation(
        "session-1", "Focused Work", {"Focused Work": 0.8, "Browsing": 0.2}, "v1",
        observed_at=started_at + timedelta(minutes=10),
    )

    snapshot = DashboardController(repository, _service_stub(), store).snapshot(monitoring_running=False)

    assert snapshot.latest_context_label == "Focused Work"
    assert snapshot.latest_context_session == "session-1"
    assert snapshot.latest_context_observed_at != "—"


def test_snapshot_shows_the_newest_of_several_observations(tmp_path) -> None:
    from app.ml.context_observation_store import ContextObservationStore

    db_path = tmp_path / "activity.db"
    repository = ActivityRepository(db_path)
    started_at = _now() - timedelta(hours=1)
    repository.start_session(StoredSession("session-1", started_at, started_at + timedelta(minutes=50)))
    store = ContextObservationStore(db_path)
    store.record_observation(
        "session-1", "Older", {"Older": 1.0}, "v1", observed_at=started_at + timedelta(minutes=10)
    )
    store.record_observation(
        "session-1", "Newer", {"Newer": 1.0}, "v1", observed_at=started_at + timedelta(minutes=40)
    )

    snapshot = DashboardController(repository, _service_stub(), store).snapshot(monitoring_running=False)

    assert snapshot.latest_context_label == "Newer"


# ============================================================================
# Milestone B: Work Thread fields and actions
# ============================================================================


def test_snapshot_work_threads_empty_when_no_work_thread_store_supplied(tmp_path) -> None:
    repository = ActivityRepository(tmp_path / "activity.db")
    snapshot = DashboardController(repository, _service_stub()).snapshot(monitoring_running=False)

    assert snapshot.work_threads == ()
    assert snapshot.latest_context_observation_id is None


def test_snapshot_work_threads_empty_when_store_empty(tmp_path) -> None:
    from app.ml.work_thread_store import WorkThreadStore

    db_path = tmp_path / "activity.db"
    repository = ActivityRepository(db_path)
    wt_store = WorkThreadStore(db_path)

    snapshot = DashboardController(repository, _service_stub(), work_thread_store=wt_store).snapshot(
        monitoring_running=False
    )

    assert snapshot.work_threads == ()


def test_snapshot_exposes_persisted_work_threads_in_order(tmp_path) -> None:
    from app.ml.work_thread_store import WorkThreadStore

    db_path = tmp_path / "activity.db"
    repository = ActivityRepository(db_path)
    wt_store = WorkThreadStore(db_path)

    now = _now()
    t1 = wt_store.create_work_thread("First Thread", created_at=now - timedelta(minutes=5))
    t2 = wt_store.create_work_thread("Second Thread", created_at=now)

    snapshot = DashboardController(repository, _service_stub(), work_thread_store=wt_store).snapshot(
        monitoring_running=False
    )

    assert len(snapshot.work_threads) == 2
    assert snapshot.work_threads[0].id == t1.id
    assert snapshot.work_threads[0].name == "First Thread"
    assert snapshot.work_threads[1].id == t2.id
    assert snapshot.work_threads[1].name == "Second Thread"


def test_snapshot_exposes_latest_context_observation_id(tmp_path) -> None:
    from app.ml.context_observation_store import ContextObservationStore

    db_path = tmp_path / "activity.db"
    repository = ActivityRepository(db_path)
    started_at = _now() - timedelta(minutes=30)
    repository.start_session(StoredSession("session-1", started_at, started_at + timedelta(minutes=10)))
    obs_store = ContextObservationStore(db_path)
    obs = obs_store.record_observation(
        "session-1", "Coding", {"Coding": 1.0}, "v1", observed_at=started_at + timedelta(minutes=10)
    )

    snapshot = DashboardController(repository, _service_stub(), obs_store).snapshot(monitoring_running=False)

    assert snapshot.latest_context_observation_id == obs.id


def test_controller_create_work_thread_and_associate_latest_context(tmp_path) -> None:
    from app.ml.context_observation_store import ContextObservationStore
    from app.ml.work_thread_store import WorkThreadStore

    db_path = tmp_path / "activity.db"
    repository = ActivityRepository(db_path)
    started_at = _now() - timedelta(minutes=30)
    repository.start_session(StoredSession("session-1", started_at, started_at + timedelta(minutes=10)))
    obs_store = ContextObservationStore(db_path)
    obs = obs_store.record_observation(
        "session-1", "Deep Work", {"Deep Work": 1.0}, "v1", observed_at=started_at + timedelta(minutes=10)
    )
    wt_store = WorkThreadStore(db_path)
    controller = DashboardController(
        repository,
        _service_stub(),
        context_observation_store=obs_store,
        work_thread_store=wt_store,
    )

    # Create thread through controller
    thread = controller.create_work_thread("Controller Created Thread")
    assert thread is not None
    assert thread.name == "Controller Created Thread"

    # Associate latest context
    assoc = controller.associate_latest_context(thread.id)
    assert assoc is not None
    assert assoc.work_thread_id == thread.id
    assert assoc.observation_id == obs.id

    # Verify through store
    assocs = wt_store.list_observations_for_work_thread(thread.id)
    assert len(assocs) == 1
    assert assocs[0].id == assoc.id


def test_controller_associate_latest_context_when_no_observation_available(tmp_path) -> None:
    from app.ml.context_observation_store import ContextObservationStore
    from app.ml.work_thread_store import WorkThreadStore

    db_path = tmp_path / "activity.db"
    repository = ActivityRepository(db_path)
    obs_store = ContextObservationStore(db_path)
    wt_store = WorkThreadStore(db_path)
    controller = DashboardController(
        repository,
        _service_stub(),
        context_observation_store=obs_store,
        work_thread_store=wt_store,
    )
    thread = wt_store.create_work_thread("Thread", created_at=_now())

    assoc = controller.associate_latest_context(thread.id)
    assert assoc is None


def test_controller_helpers_return_none_without_stores(tmp_path) -> None:
    repository = ActivityRepository(tmp_path / "activity.db")
    controller = DashboardController(repository, _service_stub())

    assert controller.create_work_thread("Test") is None
    assert controller.associate_latest_context(1) is None


def test_main_window_self_wires_work_thread_store_when_none_supplied(tmp_path) -> None:
    from app.ml.work_thread_store import WorkThreadStore
    from app.ui.main_window import MainWindow

    db_path = tmp_path / "activity.db"
    repository = ActivityRepository(db_path)
    controller = DashboardController(repository, _service_stub())

    assert controller.work_thread_store is None
    store = MainWindow._build_default_work_thread_store(controller)

    assert isinstance(store, WorkThreadStore)
    assert store.database_path == db_path
    assert controller.work_thread_store is store


def test_main_window_reuses_existing_work_thread_store_on_controller(tmp_path) -> None:
    from app.ml.work_thread_store import WorkThreadStore
    from app.ui.main_window import MainWindow

    db_path = tmp_path / "activity.db"
    repository = ActivityRepository(db_path)
    existing_store = WorkThreadStore(db_path)
    controller = DashboardController(repository, _service_stub(), work_thread_store=existing_store)

    store = MainWindow._build_default_work_thread_store(controller)
    assert store is existing_store


# ============================================================================
# Task integration tests
# ============================================================================


def test_controller_task_helpers_return_empty_or_none_without_task_store(tmp_path) -> None:
    repository = ActivityRepository(tmp_path / "activity.db")
    controller = DashboardController(repository, _service_stub())

    assert controller.list_tasks_for_work_thread(1) == ()
    assert controller.create_task(1, "Test") is None
    assert controller.toggle_task(1) is None
    controller.delete_task(1)  # must not raise


# ============================================================================
# Workspace Snapshot integration: explicit capture and best-effort restore
# ============================================================================


def test_controller_workspace_helpers_are_inert_without_workspace_services(tmp_path) -> None:
    repository = ActivityRepository(tmp_path / "activity.db")
    controller = DashboardController(repository, _service_stub())

    assert controller.list_workspace_snapshots(1) == ()
    assert controller.capture_workspace_snapshot(1) is None
    assert controller.restore_workspace_snapshot(1) is None


def test_controller_captures_and_restores_only_through_injected_workspace_ports(tmp_path) -> None:
    from app.ml.work_thread_store import WorkThreadStore
    from app.workspace.workspace_restoration import WorkspaceRestoreResult
    from app.workspace.workspace_snapshot_store import CapturedForegroundWindow, WorkspaceSnapshotStore

    class Probe:
        def capture_foreground_window(self):
            return CapturedForegroundWindow("Notepad", "notepad.exe", r"C:\\Windows\\notepad.exe", "Notes")

    class Restorer:
        def __init__(self):
            self.snapshots = []

        def restore(self, snapshot):
            self.snapshots.append(snapshot)
            return WorkspaceRestoreResult(True, "launch_executable", "Requested a no-argument application launch.")

    database_path = tmp_path / "activity.db"
    repository = ActivityRepository(database_path)
    thread = WorkThreadStore(database_path).create_work_thread("Workspace", created_at=_now())
    restorer = Restorer()
    controller = DashboardController(
        repository,
        _service_stub(),
        workspace_snapshot_store=WorkspaceSnapshotStore(database_path),
        workspace_probe=Probe(),
        workspace_restorer=restorer,
    )

    snapshot = controller.capture_workspace_snapshot(thread.id)
    assert snapshot is not None
    assert controller.list_workspace_snapshots(thread.id) == (snapshot,)
    assert controller.restore_workspace_snapshot(snapshot.id).success is True
    assert restorer.snapshots == [snapshot]


def test_main_window_self_wires_workspace_snapshot_store_when_none_supplied(tmp_path) -> None:
    from app.ui.main_window import MainWindow
    from app.workspace.workspace_snapshot_store import WorkspaceSnapshotStore

    database_path = tmp_path / "activity.db"
    repository = ActivityRepository(database_path)
    controller = DashboardController(repository, _service_stub())

    store = MainWindow._build_default_workspace_snapshot_store(controller)

    assert isinstance(store, WorkspaceSnapshotStore)
    assert store.database_path == database_path
    assert controller.workspace_snapshot_store is store


def test_controller_create_list_toggle_delete_tasks(tmp_path) -> None:
    from app.ml.task_store import TaskStore
    from app.ml.work_thread_store import WorkThreadStore

    db_path = tmp_path / "activity.db"
    repository = ActivityRepository(db_path)
    wt_store = WorkThreadStore(db_path)
    task_store = TaskStore(db_path)
    controller = DashboardController(
        repository,
        _service_stub(),
        work_thread_store=wt_store,
        task_store=task_store,
    )

    thread = wt_store.create_work_thread("Thread for tasks", created_at=_now())

    task = controller.create_task(thread.id, "Buy coffee")
    assert task is not None
    assert task.title == "Buy coffee"
    assert task.is_done is False

    tasks = controller.list_tasks_for_work_thread(thread.id)
    assert len(tasks) == 1
    assert tasks[0].id == task.id

    toggled = controller.toggle_task(task.id)
    assert toggled is not None
    assert toggled.is_done is True

    controller.delete_task(task.id)
    assert controller.list_tasks_for_work_thread(thread.id) == ()


def test_main_window_self_wires_task_store_when_none_supplied(tmp_path) -> None:
    from app.ml.task_store import TaskStore
    from app.ui.main_window import MainWindow

    db_path = tmp_path / "activity.db"
    repository = ActivityRepository(db_path)
    controller = DashboardController(repository, _service_stub())

    assert controller.task_store is None
    store = MainWindow._build_default_task_store(controller)

    assert isinstance(store, TaskStore)
    assert store.database_path == db_path
    assert controller.task_store is store


def test_main_window_reuses_existing_task_store_on_controller(tmp_path) -> None:
    from app.ml.task_store import TaskStore
    from app.ui.main_window import MainWindow

    db_path = tmp_path / "activity.db"
    repository = ActivityRepository(db_path)
    existing_store = TaskStore(db_path)
    controller = DashboardController(repository, _service_stub(), task_store=existing_store)

    store = MainWindow._build_default_task_store(controller)
    assert store is existing_store


# ============================================================================
# Unfinished Work: list_open_tasks() and snapshot.open_tasks
# ============================================================================


def _full_controller(tmp_path) -> tuple:
    """Return (controller, wt_store, task_store) all sharing one DB."""
    from app.ml.task_store import TaskStore
    from app.ml.work_thread_store import WorkThreadStore

    db_path = tmp_path / "activity.db"
    repository = ActivityRepository(db_path)
    wt_store = WorkThreadStore(db_path)
    task_store = TaskStore(db_path)
    controller = DashboardController(
        repository,
        _service_stub(),
        work_thread_store=wt_store,
        task_store=task_store,
    )
    return controller, wt_store, task_store


def test_list_open_tasks_returns_empty_without_stores(tmp_path) -> None:
    repository = ActivityRepository(tmp_path / "activity.db")
    controller = DashboardController(repository, _service_stub())

    assert controller.list_open_tasks() == ()


def test_list_open_tasks_returns_empty_when_no_threads(tmp_path) -> None:
    controller, _wts, _ts = _full_controller(tmp_path)

    assert controller.list_open_tasks() == ()


def test_list_open_tasks_returns_empty_when_all_tasks_done(tmp_path) -> None:
    controller, wt_store, task_store = _full_controller(tmp_path)
    thread = wt_store.create_work_thread("Done thread", created_at=_now())
    task = task_store.create_task(thread.id, "Already done", created_at=_now())
    task_store.complete_task(task.id)

    assert controller.list_open_tasks() == ()


def test_list_open_tasks_returns_only_incomplete_tasks(tmp_path) -> None:
    controller, wt_store, task_store = _full_controller(tmp_path)
    thread = wt_store.create_work_thread("Mixed thread", created_at=_now())
    open_task = task_store.create_task(thread.id, "Still open", created_at=_now())
    done_task = task_store.create_task(thread.id, "Finished", created_at=_now())
    task_store.complete_task(done_task.id)

    result = controller.list_open_tasks()

    assert len(result) == 1
    returned_thread, open_tasks = result[0]
    assert returned_thread.id == thread.id
    assert len(open_tasks) == 1
    assert open_tasks[0].id == open_task.id
    assert open_tasks[0].is_done is False


def test_list_open_tasks_omits_threads_with_no_open_tasks(tmp_path) -> None:
    controller, wt_store, task_store = _full_controller(tmp_path)
    done_thread = wt_store.create_work_thread("Done thread", created_at=_now() - timedelta(minutes=5))
    open_thread = wt_store.create_work_thread("Open thread", created_at=_now())

    t1 = task_store.create_task(done_thread.id, "Finished task", created_at=_now())
    task_store.complete_task(t1.id)
    task_store.create_task(open_thread.id, "Pending task", created_at=_now())

    result = controller.list_open_tasks()

    assert len(result) == 1
    assert result[0][0].id == open_thread.id


def test_list_open_tasks_groups_by_thread_in_creation_order(tmp_path) -> None:
    controller, wt_store, task_store = _full_controller(tmp_path)
    t1 = wt_store.create_work_thread("Alpha", created_at=_now() - timedelta(minutes=10))
    t2 = wt_store.create_work_thread("Beta", created_at=_now())
    task_store.create_task(t1.id, "Alpha task 1", created_at=_now())
    task_store.create_task(t1.id, "Alpha task 2", created_at=_now())
    task_store.create_task(t2.id, "Beta task", created_at=_now())

    result = controller.list_open_tasks()

    assert len(result) == 2
    assert result[0][0].id == t1.id
    assert len(result[0][1]) == 2
    assert result[1][0].id == t2.id
    assert len(result[1][1]) == 1


def test_snapshot_open_tasks_matches_list_open_tasks(tmp_path) -> None:
    controller, wt_store, task_store = _full_controller(tmp_path)
    thread = wt_store.create_work_thread("Snapshot thread", created_at=_now())
    task_store.create_task(thread.id, "Snapshot task", created_at=_now())

    snapshot = controller.snapshot(monitoring_running=False)
    direct = controller.list_open_tasks()

    assert len(snapshot.open_tasks) == len(direct)
    assert snapshot.open_tasks[0][0].id == direct[0][0].id
    assert snapshot.open_tasks[0][1][0].id == direct[0][1][0].id


def test_marking_task_done_removes_it_from_open_tasks(tmp_path) -> None:
    controller, wt_store, task_store = _full_controller(tmp_path)
    thread = wt_store.create_work_thread("Thread", created_at=_now())
    task = task_store.create_task(thread.id, "Will be done", created_at=_now())

    assert len(controller.list_open_tasks()) == 1

    task_store.complete_task(task.id)

    assert controller.list_open_tasks() == ()


def test_open_tasks_isolated_across_threads(tmp_path) -> None:
    """Open tasks from one thread must not bleed into another thread's group."""
    controller, wt_store, task_store = _full_controller(tmp_path)
    t1 = wt_store.create_work_thread("Thread A", created_at=_now() - timedelta(minutes=5))
    t2 = wt_store.create_work_thread("Thread B", created_at=_now())
    ta = task_store.create_task(t1.id, "A task", created_at=_now())
    task_store.create_task(t2.id, "B task", created_at=_now())

    result = controller.list_open_tasks()
    # Both threads have open tasks
    assert len(result) == 2
    id_map = {thread.id: tasks for thread, tasks in result}
    assert all(t.work_thread_id == t1.id for t in id_map[t1.id])
    assert all(t.work_thread_id == t2.id for t in id_map[t2.id])

    # Complete t1's task — t1 should disappear from unfinished
    task_store.complete_task(ta.id)
    result2 = controller.list_open_tasks()
    assert len(result2) == 1
    assert result2[0][0].id == t2.id
