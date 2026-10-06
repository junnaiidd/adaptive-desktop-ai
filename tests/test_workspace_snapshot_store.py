"""Tests for the isolated, local Workspace Snapshot persistence boundary."""

from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from app.ml.work_thread_store import WorkThreadStore
from app.workspace.workspace_snapshot_store import (
    CapturedForegroundWindow,
    WorkspaceSnapshotNotFoundError,
    WorkspaceSnapshotStore,
)


def _now(offset: int = 0) -> datetime:
    return datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc) + timedelta(seconds=offset)


def _window(title: str | None = "Notes") -> CapturedForegroundWindow:
    return CapturedForegroundWindow("Notepad", "notepad.exe", r"C:\\Windows\\notepad.exe", title)


def _store_with_thread(tmp_path):
    database_path = tmp_path / "activity.db"
    thread = WorkThreadStore(database_path).create_work_thread("Thread", created_at=_now())
    return WorkspaceSnapshotStore(database_path), thread.id


def test_snapshot_schema_create_round_trip_and_restart_durability(tmp_path) -> None:
    store, thread_id = _store_with_thread(tmp_path)
    snapshot = store.create_snapshot(thread_id, _window(), captured_at=_now())

    reloaded = WorkspaceSnapshotStore(store.database_path).get_snapshot(snapshot.id)
    assert reloaded == snapshot


def test_snapshot_belongs_to_one_existing_work_thread_and_is_scoped(tmp_path) -> None:
    database_path = tmp_path / "activity.db"
    threads = WorkThreadStore(database_path)
    first = threads.create_work_thread("First", created_at=_now())
    second = threads.create_work_thread("Second", created_at=_now(1))
    store = WorkspaceSnapshotStore(database_path)
    first_snapshot = store.create_snapshot(first.id, _window(), captured_at=_now())
    store.create_snapshot(second.id, _window("Second"), captured_at=_now(1))

    assert store.list_snapshots_for_work_thread(first.id) == [first_snapshot]
    with pytest.raises(sqlite3.IntegrityError):
        store.create_snapshot(9999, _window(), captured_at=_now())


def test_snapshot_titles_are_sanitized_at_the_persistence_boundary(tmp_path) -> None:
    store, thread_id = _store_with_thread(tmp_path)
    snapshot = store.create_snapshot(thread_id, _window("Reset password - Notes"), captured_at=_now())

    assert snapshot.window_title == "[Sensitive window title hidden]"
    connection = sqlite3.connect(store.database_path)
    persisted_title = connection.execute(
        "SELECT window_title FROM workspace_snapshots WHERE id = ?", (snapshot.id,)
    ).fetchone()[0]
    connection.close()
    assert persisted_title == "[Sensitive window title hidden]"


def test_snapshots_are_deterministically_ordered_and_delete_is_explicit(tmp_path) -> None:
    store, thread_id = _store_with_thread(tmp_path)
    later = store.create_snapshot(thread_id, _window("Later"), captured_at=_now(20))
    earlier = store.create_snapshot(thread_id, _window("Earlier"), captured_at=_now(10))

    assert [item.id for item in store.list_snapshots_for_work_thread(thread_id)] == [earlier.id, later.id]
    store.delete_snapshot(earlier.id)
    assert store.get_snapshot(earlier.id) is None
    with pytest.raises(WorkspaceSnapshotNotFoundError):
        store.delete_snapshot(earlier.id)


def test_snapshot_rejects_naive_timestamps_and_incomplete_launch_identity(tmp_path) -> None:
    store, thread_id = _store_with_thread(tmp_path)
    with pytest.raises(ValueError, match="captured_at must be timezone-aware"):
        store.create_snapshot(thread_id, _window(), captured_at=datetime(2026, 10, 4, 12, 0))
    with pytest.raises(ValueError, match="executable_path must be a non-empty string"):
        store.create_snapshot(
            thread_id,
            CapturedForegroundWindow("Notepad", "notepad.exe", "", "Notes"),
            captured_at=_now(),
        )
