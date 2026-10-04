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
