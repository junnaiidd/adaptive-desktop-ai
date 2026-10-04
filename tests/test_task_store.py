"""
Comprehensive tests for TaskStore (`app/ml/task_store.py`).

Covers:
- Schema initialization / idempotency
- create_task / get_task / list_tasks_for_work_thread / rename_task / delete_task
- toggle_task / complete_task / set_task_done
- Exactly one Work Thread association
- Foreign Key enforcement (PRAGMA foreign_keys = ON)
- Deterministic ordering
- Restart durability
- Blocking Work Thread deletion when it has tasks (WorkThreadHasTasksError)
- Unblocking Work Thread deletion after tasks are deleted
- Timestamp validation (timezone awareness)
- Architectural constraints (strictly user-created)
"""

from __future__ import annotations

import ast
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.ml.task_store import Task, TaskNotFoundError, TaskStore
from app.ml.work_thread_store import (
    WorkThreadHasTasksError,
    WorkThreadStore,
)


def _utc_now(offset_seconds: int = 0) -> datetime:
    return datetime(2026, 10, 4, 16, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=offset_seconds)


def _db_with_thread(tmp_path: Path, thread_name: str = "Main Thread") -> tuple[Path, int]:
    db_path = tmp_path / "activity.db"
    wt_store = WorkThreadStore(db_path)
    thread = wt_store.create_work_thread(thread_name, created_at=_utc_now(-3600))
    return db_path, thread.id


# ============================================================================
# 1. Schema initialization & idempotency
# ============================================================================


def test_schema_initialization_creates_tasks_table(tmp_path: Path) -> None:
    db_path = tmp_path / "activity.db"
    TaskStore(db_path)

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

    assert "tasks" in tables
    assert "idx_tasks_work_thread_id" in indexes


def test_schema_initialization_is_idempotent(tmp_path: Path) -> None:
    db_path, thread_id = _db_with_thread(tmp_path)
    TaskStore(db_path)
    store2 = TaskStore(db_path)
    assert store2.list_tasks_for_work_thread(thread_id) == []


# ============================================================================
# 2. create / get / list / rename / delete
# ============================================================================


def test_create_and_get_task(tmp_path: Path) -> None:
    db_path, thread_id = _db_with_thread(tmp_path)
    store = TaskStore(db_path)

    created_at = _utc_now()
    task = store.create_task(thread_id, "Write unit tests", created_at=created_at)

    assert isinstance(task, Task)
    assert task.id is not None
    assert task.work_thread_id == thread_id
    assert task.title == "Write unit tests"
    assert task.created_at == created_at
    assert task.is_done is False

    fetched = store.get_task(task.id)
    assert fetched == task


def test_create_task_validates_title(tmp_path: Path) -> None:
    db_path, thread_id = _db_with_thread(tmp_path)
    store = TaskStore(db_path)

    with pytest.raises(ValueError, match="title must be a non-empty string"):
        store.create_task(thread_id, "", created_at=_utc_now())

    with pytest.raises(ValueError, match="title must be a non-empty string"):
        store.create_task(thread_id, "   ", created_at=_utc_now())


def test_create_task_validates_timezone(tmp_path: Path) -> None:
    db_path, thread_id = _db_with_thread(tmp_path)
    store = TaskStore(db_path)

    naive = datetime(2026, 10, 4, 16, 0, 0)
    with pytest.raises(ValueError, match="created_at must be timezone-aware"):
        store.create_task(thread_id, "Task with naive dt", created_at=naive)


def test_list_tasks_deterministic_ordering(tmp_path: Path) -> None:
    db_path, thread_id = _db_with_thread(tmp_path)
    store = TaskStore(db_path)

    t2 = store.create_task(thread_id, "Task B", created_at=_utc_now(20))
    t1 = store.create_task(thread_id, "Task A", created_at=_utc_now(10))
    t3 = store.create_task(thread_id, "Task C", created_at=_utc_now(20))

    tasks = store.list_tasks_for_work_thread(thread_id)
    # Ordered by created_at_utc ASC, id ASC
    assert [t.id for t in tasks] == [t1.id, t2.id, t3.id]


def test_rename_task(tmp_path: Path) -> None:
    db_path, thread_id = _db_with_thread(tmp_path)
    store = TaskStore(db_path)

    task = store.create_task(thread_id, "Initial Title", created_at=_utc_now())
    renamed = store.rename_task(task.id, "Updated Title")

    assert renamed.id == task.id
    assert renamed.title == "Updated Title"
    assert store.get_task(task.id).title == "Updated Title"


def test_rename_task_validation_and_not_found(tmp_path: Path) -> None:
    db_path, thread_id = _db_with_thread(tmp_path)
    store = TaskStore(db_path)
    task = store.create_task(thread_id, "Valid", created_at=_utc_now())

    with pytest.raises(ValueError, match="new_title must be a non-empty string"):
        store.rename_task(task.id, "  ")

    with pytest.raises(TaskNotFoundError):
        store.rename_task(99999, "Non-existent")


def test_delete_task(tmp_path: Path) -> None:
    db_path, thread_id = _db_with_thread(tmp_path)
    store = TaskStore(db_path)

    task = store.create_task(thread_id, "Task to delete", created_at=_utc_now())
    store.delete_task(task.id)

    assert store.get_task(task.id) is None
    assert store.list_tasks_for_work_thread(thread_id) == []


def test_delete_task_not_found(tmp_path: Path) -> None:
    db_path, _ = _db_with_thread(tmp_path)
    store = TaskStore(db_path)
    with pytest.raises(TaskNotFoundError):
        store.delete_task(99999)


# ============================================================================
# 3. toggle / complete / set_task_done
# ============================================================================


def test_toggle_task(tmp_path: Path) -> None:
    db_path, thread_id = _db_with_thread(tmp_path)
    store = TaskStore(db_path)

    task = store.create_task(thread_id, "Toggle me", created_at=_utc_now())
    assert task.is_done is False

    t1 = store.toggle_task(task.id)
    assert t1.is_done is True
    assert store.get_task(task.id).is_done is True

    t2 = store.toggle_task(task.id)
    assert t2.is_done is False
    assert store.get_task(task.id).is_done is False


def test_complete_and_set_task_done(tmp_path: Path) -> None:
    db_path, thread_id = _db_with_thread(tmp_path)
    store = TaskStore(db_path)

    task = store.create_task(thread_id, "Complete me", created_at=_utc_now())
    completed = store.complete_task(task.id)
    assert completed.is_done is True

    reset = store.set_task_done(task.id, is_done=False)
    assert reset.is_done is False


# ============================================================================
# 4. Foreign Key enforcement & Exactly ONE Work Thread
# ============================================================================


def test_fk_enforcement_on_invalid_work_thread(tmp_path: Path) -> None:
    db_path = tmp_path / "activity.db"
    # Create tables by initializing stores
    WorkThreadStore(db_path)
    store = TaskStore(db_path)

    with pytest.raises(sqlite3.IntegrityError):
        store.create_task(99999, "Orphan task", created_at=_utc_now())


def test_tasks_isolated_to_their_own_work_thread(tmp_path: Path) -> None:
    db_path = tmp_path / "activity.db"
    wt_store = WorkThreadStore(db_path)
    t1 = wt_store.create_work_thread("Thread 1", created_at=_utc_now(-100))
    t2 = wt_store.create_work_thread("Thread 2", created_at=_utc_now(-50))

    store = TaskStore(db_path)
    task1 = store.create_task(t1.id, "Task for Thread 1", created_at=_utc_now(-30))
    task2 = store.create_task(t2.id, "Task for Thread 2", created_at=_utc_now(-10))

    t1_tasks = store.list_tasks_for_work_thread(t1.id)
    t2_tasks = store.list_tasks_for_work_thread(t2.id)

    assert [t.id for t in t1_tasks] == [task1.id]
    assert [t.id for t in t2_tasks] == [task2.id]


# ============================================================================
# 5. Blocking Work Thread deletion when tasks exist
# ============================================================================


def test_block_work_thread_deletion_when_tasks_exist(tmp_path: Path) -> None:
    db_path = tmp_path / "activity.db"
    wt_store = WorkThreadStore(db_path)
    thread = wt_store.create_work_thread("Thread with tasks", created_at=_utc_now(-100))

    task_store = TaskStore(db_path)
    task = task_store.create_task(thread.id, "Sub task", created_at=_utc_now(-50))

    # Deleting thread must be blocked
    with pytest.raises(WorkThreadHasTasksError) as exc_info:
        wt_store.delete_work_thread(thread.id)

    assert f"Cannot delete work thread {thread.id}" in str(exc_info.value)
    assert "1 task(s)" in str(exc_info.value)

    # Thread still exists
    assert wt_store.get_work_thread(thread.id) is not None

    # Deleting the task unblocks thread deletion
    task_store.delete_task(task.id)
    wt_store.delete_work_thread(thread.id)
    assert wt_store.get_work_thread(thread.id) is None


# ============================================================================
# 6. Restart durability
# ============================================================================


def test_task_restart_durability(tmp_path: Path) -> None:
    db_path, thread_id = _db_with_thread(tmp_path)
    store1 = TaskStore(db_path)
    created_at = _utc_now()
    task = store1.create_task(thread_id, "Durable Task", created_at=created_at)
    store1.complete_task(task.id)

    store2 = TaskStore(db_path)
    fetched = store2.get_task(task.id)
    assert fetched is not None
    assert fetched.id == task.id
    assert fetched.work_thread_id == thread_id
    assert fetched.title == "Durable Task"
    assert fetched.created_at == created_at
    assert fetched.is_done is True


# ============================================================================
# 7. Architectural guards
# ============================================================================


def test_architecture_guards_no_prohibited_imports() -> None:
    source_path = Path(__file__).resolve().parents[1] / "app" / "ml" / "task_store.py"
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
    }

    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    for prohibited in prohibited_modules:
        assert prohibited not in imported_modules, f"Prohibited module {prohibited} was imported in task_store.py"
