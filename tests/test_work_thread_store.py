"""
Comprehensive tests for Milestone B: WorkThreadStore (`app/ml/work_thread_store.py`).

Covers all required areas:
- Schema initialization / idempotency
- create / get / list / rename / delete
- Association create / remove / list
- Deterministic ordering
- Restart durability
- FK enforcement on every connection
- Many-to-many behavior
- Duplicate names
- Deletion blocking with WorkThreadHasAssociationsError
- Timestamp validation (timezone awareness)
- Real ContextObservation -> WorkThread association integration
- ContextObservation append-only / untouched verification
- Architecture guards
"""

from __future__ import annotations

import ast
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.database.activity_repository import ActivityRepository, StoredSession
from app.ml.context_observation_store import ContextObservationStore
from app.ml.work_thread_store import (
    AssociationNotFoundError,
    WorkThread,
    WorkThreadHasAssociationsError,
    WorkThreadNotFoundError,
    WorkThreadObservation,
    WorkThreadStore,
)

# ============================================================================
# Helpers
# ============================================================================


def _utc_now(offset_seconds: int = 0) -> datetime:
    return datetime(2026, 10, 4, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=offset_seconds)


def _setup_activity_and_observation_db(tmp_path: Path) -> tuple[Path, int]:
    """
    Creates a real database with a session and a context observation,
    returning (db_path, observation_id).
    """
    db_path = tmp_path / "activity.db"
    repo = ActivityRepository(db_path)
    started = _utc_now(-1800)
    ended = _utc_now(-60)
    repo.start_session(StoredSession("sess-1", started, None))
    repo.end_session(StoredSession("sess-1", started, ended))

    obs_store = ContextObservationStore(db_path)
    obs = obs_store.record_observation(
        session_id="sess-1",
        predicted_label="Software Development",
        class_probabilities={"Software Development": 0.85, "Browsing": 0.15},
        model_version="test-v1",
        observed_at=ended,
    )
    return db_path, obs.id


# ============================================================================
# 1. Schema initialization & idempotency
# ============================================================================


def test_schema_initialization_creates_tables_and_indexes(tmp_path: Path) -> None:
    db_path = tmp_path / "activity.db"
    WorkThreadStore(db_path)

    connection = sqlite3.connect(db_path)
    tables = {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    indexes = {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
    }
    connection.close()

    assert "work_threads" in tables
    assert "work_thread_observations" in tables
    assert "idx_work_thread_observations_thread_id" in indexes
    assert "idx_work_thread_observations_observation_id" in indexes


def test_schema_initialization_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "activity.db"
    WorkThreadStore(db_path)
    # Re-running initialization on an existing database must not error or alter data
    store2 = WorkThreadStore(db_path)
    assert store2.list_work_threads() == []


# ============================================================================
# 2. create / get / list / rename / delete
# ============================================================================


def test_create_work_thread_success(tmp_path: Path) -> None:
    store = WorkThreadStore(tmp_path / "activity.db")
    created_at = _utc_now()
    thread = store.create_work_thread("Refactoring ML Pipeline", created_at=created_at)

    assert isinstance(thread, WorkThread)
    assert thread.id is not None
    assert thread.name == "Refactoring ML Pipeline"
    assert thread.created_at == created_at
    assert thread.work_thread_id == thread.id


def test_create_work_thread_validates_name(tmp_path: Path) -> None:
    store = WorkThreadStore(tmp_path / "activity.db")
    created_at = _utc_now()

    with pytest.raises(ValueError, match="name must be a non-empty string"):
        store.create_work_thread("", created_at=created_at)

    with pytest.raises(ValueError, match="name must be a non-empty string"):
        store.create_work_thread("   ", created_at=created_at)

    with pytest.raises(ValueError, match="name must be a non-empty string"):
        store.create_work_thread(None, created_at=created_at)  # type: ignore[arg-type]


def test_create_work_thread_validates_timezone(tmp_path: Path) -> None:
    store = WorkThreadStore(tmp_path / "activity.db")
    naive_dt = datetime(2026, 10, 4, 12, 0, 0)  # no tzinfo

    with pytest.raises(ValueError, match="created_at must be timezone-aware"):
        store.create_work_thread("Test Thread", created_at=naive_dt)

    with pytest.raises(ValueError, match="created_at must be a datetime instance"):
        store.create_work_thread("Test Thread", created_at="2026-10-04T12:00:00Z")  # type: ignore[arg-type]


def test_create_work_thread_allows_duplicate_names(tmp_path: Path) -> None:
    store = WorkThreadStore(tmp_path / "activity.db")
    t1 = store.create_work_thread("Project Research", created_at=_utc_now(0))
    t2 = store.create_work_thread("Project Research", created_at=_utc_now(10))

    assert t1.id != t2.id
    assert t1.name == t2.name == "Project Research"
    threads = store.list_work_threads()
    assert len(threads) == 2


def test_get_work_thread_existing_and_non_existing(tmp_path: Path) -> None:
    store = WorkThreadStore(tmp_path / "activity.db")
    thread = store.create_work_thread("Testing", created_at=_utc_now())

    fetched = store.get_work_thread(thread.id)
    assert fetched == thread

    assert store.get_work_thread(999999) is None


def test_list_work_threads_deterministic_ordering(tmp_path: Path) -> None:
    store = WorkThreadStore(tmp_path / "activity.db")
    # Insert in scrambled order
    t2 = store.create_work_thread("Thread B", created_at=_utc_now(20))
    t1 = store.create_work_thread("Thread A", created_at=_utc_now(10))
    t3 = store.create_work_thread("Thread C", created_at=_utc_now(20))

    threads = store.list_work_threads()
    # Ordered by created_at ASC, tie-broken by id ASC
    assert [t.id for t in threads] == [t1.id, t2.id, t3.id]


def test_rename_work_thread_success(tmp_path: Path) -> None:
    store = WorkThreadStore(tmp_path / "activity.db")
    thread = store.create_work_thread("Original Name", created_at=_utc_now())

    updated = store.rename_work_thread(thread.id, "Renamed Name")
    assert updated.id == thread.id
    assert updated.name == "Renamed Name"
    assert updated.created_at == thread.created_at

    fetched = store.get_work_thread(thread.id)
    assert fetched is not None
    assert fetched.name == "Renamed Name"


def test_rename_work_thread_validation_and_not_found(tmp_path: Path) -> None:
    store = WorkThreadStore(tmp_path / "activity.db")
    thread = store.create_work_thread("Original", created_at=_utc_now())

    with pytest.raises(ValueError, match="new_name must be a non-empty string"):
        store.rename_work_thread(thread.id, "")

    with pytest.raises(WorkThreadNotFoundError):
        store.rename_work_thread(99999, "New Name")


def test_delete_work_thread_without_associations(tmp_path: Path) -> None:
    store = WorkThreadStore(tmp_path / "activity.db")
    thread = store.create_work_thread("Disposable Thread", created_at=_utc_now())

    store.delete_work_thread(thread.id)
    assert store.get_work_thread(thread.id) is None
    assert store.list_work_threads() == []


def test_delete_work_thread_not_found(tmp_path: Path) -> None:
    store = WorkThreadStore(tmp_path / "activity.db")
    with pytest.raises(WorkThreadNotFoundError):
        store.delete_work_thread(99999)


# ============================================================================
# 3. Association create / remove / list & deletion blocking
# ============================================================================


def test_associate_and_remove_association(tmp_path: Path) -> None:
    db_path, obs_id = _setup_activity_and_observation_db(tmp_path)
    store = WorkThreadStore(db_path)
    thread = store.create_work_thread("Feature Work", created_at=_utc_now())

    assoc_time = _utc_now(10)
    assoc = store.associate_observation(thread.id, obs_id, associated_at=assoc_time)

    assert isinstance(assoc, WorkThreadObservation)
    assert assoc.id is not None
    assert assoc.work_thread_id == thread.id
    assert assoc.observation_id == obs_id
    assert assoc.associated_at == assoc_time
    assert assoc.association_id == assoc.id

    # List observations for thread
    thread_obs = store.list_observations_for_work_thread(thread.id)
    assert thread_obs == [assoc]

    # Remove association
    store.remove_association(assoc.id)
    assert store.list_observations_for_work_thread(thread.id) == []


def test_associate_observation_validates_timezone(tmp_path: Path) -> None:
    db_path, obs_id = _setup_activity_and_observation_db(tmp_path)
    store = WorkThreadStore(db_path)
    thread = store.create_work_thread("Test Thread", created_at=_utc_now())

    with pytest.raises(ValueError, match="associated_at must be timezone-aware"):
        store.associate_observation(thread.id, obs_id, associated_at=datetime(2026, 10, 4, 12, 0))


def test_remove_association_not_found(tmp_path: Path) -> None:
    db_path, _ = _setup_activity_and_observation_db(tmp_path)
    store = WorkThreadStore(db_path)
    with pytest.raises(AssociationNotFoundError):
        store.remove_association(99999)


def test_delete_work_thread_blocked_when_has_associations(tmp_path: Path) -> None:
    db_path, obs_id = _setup_activity_and_observation_db(tmp_path)
    store = WorkThreadStore(db_path)
    thread = store.create_work_thread("Critical Thread", created_at=_utc_now())
    assoc = store.associate_observation(thread.id, obs_id, associated_at=_utc_now(5))

    # Deletion must be blocked
    with pytest.raises(WorkThreadHasAssociationsError) as exc_info:
        store.delete_work_thread(thread.id)

    assert f"Cannot delete work thread {thread.id}" in str(exc_info.value)
    assert "1 associated observation" in str(exc_info.value)

    # Thread still exists
    assert store.get_work_thread(thread.id) is not None

    # After removing the association, deletion must succeed
    store.remove_association(assoc.id)
    store.delete_work_thread(thread.id)
    assert store.get_work_thread(thread.id) is None


# ============================================================================
# 4. Foreign Key enforcement on every connection
# ============================================================================


def test_fk_enforcement_invalid_thread_id(tmp_path: Path) -> None:
    db_path, obs_id = _setup_activity_and_observation_db(tmp_path)
    store = WorkThreadStore(db_path)

    with pytest.raises(sqlite3.IntegrityError):
        store.associate_observation(99999, obs_id, associated_at=_utc_now())


def test_fk_enforcement_invalid_observation_id(tmp_path: Path) -> None:
    db_path, _ = _setup_activity_and_observation_db(tmp_path)
    store = WorkThreadStore(db_path)
    thread = store.create_work_thread("Thread", created_at=_utc_now())

    with pytest.raises(sqlite3.IntegrityError):
        store.associate_observation(thread.id, 99999, associated_at=_utc_now())


# ============================================================================
# 5. Many-to-many behavior and deterministic ordering
# ============================================================================


def test_many_to_many_and_deterministic_ordering(tmp_path: Path) -> None:
    db_path = tmp_path / "activity.db"
    repo = ActivityRepository(db_path)
    obs_store = ContextObservationStore(db_path)

    # Create 2 sessions & 2 observations
    for s_id in ("sess-1", "sess-2"):
        t = _utc_now(-1000)
        repo.start_session(StoredSession(s_id, t, None))
        repo.end_session(StoredSession(s_id, t, t + timedelta(minutes=10)))

    obs1 = obs_store.record_observation("sess-1", "Coding", {"Coding": 1.0}, "v1", observed_at=_utc_now(-500))
    obs2 = obs_store.record_observation("sess-2", "Browsing", {"Browsing": 1.0}, "v1", observed_at=_utc_now(-400))

    store = WorkThreadStore(db_path)
    t1 = store.create_work_thread("Thread Alpha", created_at=_utc_now(-300))
    t2 = store.create_work_thread("Thread Beta", created_at=_utc_now(-200))

    # Many-to-many:
    # t1 is associated with obs1 and obs2
    # t2 is associated with obs1
    assoc_t1_obs1 = store.associate_observation(t1.id, obs1.id, associated_at=_utc_now(-100))
    assoc_t1_obs2 = store.associate_observation(t1.id, obs2.id, associated_at=_utc_now(-50))
    assoc_t2_obs1 = store.associate_observation(t2.id, obs1.id, associated_at=_utc_now(-30))

    # Check observations for t1: ordered by associated_at ASC, id ASC
    t1_obs = store.list_observations_for_work_thread(t1.id)
    assert [a.id for a in t1_obs] == [assoc_t1_obs1.id, assoc_t1_obs2.id]
    assert [a.observation_id for a in t1_obs] == [obs1.id, obs2.id]

    # Check work threads for obs1: should list t1 and t2
    obs1_threads = store.list_work_threads_for_observation(obs1.id)
    assert [t.id for t in obs1_threads] == [t1.id, t2.id]
    assert [t.name for t in obs1_threads] == ["Thread Alpha", "Thread Beta"]

    # Check work threads for obs2: should list only t1
    obs2_threads = store.list_work_threads_for_observation(obs2.id)
    assert [t.id for t in obs2_threads] == [t1.id]


# ============================================================================
# 6. Restart durability
# ============================================================================


def test_restart_durability(tmp_path: Path) -> None:
    db_path, obs_id = _setup_activity_and_observation_db(tmp_path)

    store1 = WorkThreadStore(db_path)
    created_time = _utc_now(-100)
    thread = store1.create_work_thread("Durable Thread", created_at=created_time)
    assoc_time = _utc_now(-50)
    assoc = store1.associate_observation(thread.id, obs_id, associated_at=assoc_time)

    # Reopen with a completely new WorkThreadStore instance
    store2 = WorkThreadStore(db_path)
    fetched_thread = store2.get_work_thread(thread.id)
    assert fetched_thread is not None
    assert fetched_thread.id == thread.id
    assert fetched_thread.name == "Durable Thread"
    assert fetched_thread.created_at == created_time

    fetched_assocs = store2.list_observations_for_work_thread(thread.id)
    assert len(fetched_assocs) == 1
    assert fetched_assocs[0].id == assoc.id
    assert fetched_assocs[0].work_thread_id == thread.id
    assert fetched_assocs[0].observation_id == obs_id
    assert fetched_assocs[0].associated_at == assoc_time


# ============================================================================
# 7. Context observations remain untouched and append-only
# ============================================================================


def test_context_observations_remain_untouched_and_append_only(tmp_path: Path) -> None:
    db_path, obs_id = _setup_activity_and_observation_db(tmp_path)
    obs_store = ContextObservationStore(db_path)
    initial_obs = obs_store.get_observation(obs_id)
    assert initial_obs is not None

    wt_store = WorkThreadStore(db_path)
    thread = wt_store.create_work_thread("Research", created_at=_utc_now())
    assoc = wt_store.associate_observation(thread.id, obs_id, associated_at=_utc_now())

    # Verify context_observations row has not been modified
    obs_after = obs_store.get_observation(obs_id)
    assert obs_after == initial_obs

    # Removing the association does not affect context observation either
    wt_store.remove_association(assoc.id)
    obs_after_removal = obs_store.get_observation(obs_id)
    assert obs_after_removal == initial_obs


# ============================================================================
# 8. Architecture Guards
# ============================================================================


def test_architecture_guards_no_prohibited_imports() -> None:
    source_path = Path(__file__).resolve().parents[1] / "app" / "ml" / "work_thread_store.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    prohibited_modules = {
        "sklearn",
        "scipy",
        "numpy",
        "torch",
        "transformers",
        "app.ml.context_classifier",
        "app.ml.context_clustering",
        "app.ml.context_inference",
        "app.ml.training_pipeline",
        "app.core.monitoring_service",
        "app.core.activity_monitor",
    }

    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    for prohibited in prohibited_modules:
        assert prohibited not in imported_modules, f"Prohibited module {prohibited} was imported in work_thread_store.py"


def test_architecture_guards_strictly_user_created() -> None:
    """Ensure no automatic background threads, inference, or tasks are defined in work_thread_store."""
    source_path = Path(__file__).resolve().parents[1] / "app" / "ml" / "work_thread_store.py"
    content = source_path.read_text(encoding="utf-8").lower()

    prohibited_terms = [
        "task",
        "deadline",
        "cluster",
        "similarity",
        "embedding",
        "predict",
        "infer",
        "qthread",
        "qtimer",
    ]
    for term in prohibited_terms:
        # Check function and class definitions for these terms
        assert f"def {term}" not in content
        assert f"class {term}" not in content
