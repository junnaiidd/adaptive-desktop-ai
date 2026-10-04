"""
Task entity and persistence layer for Adaptive Desktop AI.

Allows users to create and manage concrete tasks inside an existing Work Thread.

What this module IS:
--------------------
A storage component for user-created tasks belonging to a single Work Thread.
It manages the `tasks` table in the application's shared SQLite database.

What this module is NOT:
------------------------
- No deadlines or priorities.
- No automatic task generation or detection.
- No ML, embeddings, similarity matching, or LLMs.
- No task/context-observation associations.
- Tasks are strictly and exclusively user-created.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


class TaskError(Exception):
    """Base exception for TaskStore operations."""


class TaskNotFoundError(TaskError, KeyError):
    """Raised when a task cannot be found by ID."""


@dataclass(frozen=True, slots=True)
class Task:
    """A user-created task within a WorkThread."""

    id: int
    work_thread_id: int
    title: str
    created_at: datetime  # UTC, timezone-aware
    is_done: bool


class TaskStore:
    """
    Persistence layer for tasks belonging to Work Threads, in the
    application's shared SQLite database.
    """

    def __init__(self, database_path: str | Path | None = None) -> None:
        default_path = Path(__file__).resolve().parents[2] / "data" / "activity.db"
        self.database_path = Path(database_path) if database_path else default_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        """Create the tasks table and its index if they do not exist."""
        connection = self._connect()
        try:
            with connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS tasks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        work_thread_id INTEGER NOT NULL,
                        title TEXT NOT NULL,
                        created_at_utc TEXT NOT NULL,
                        is_done INTEGER NOT NULL DEFAULT 0,
                        FOREIGN KEY (work_thread_id) REFERENCES work_threads(id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_tasks_work_thread_id
                        ON tasks (work_thread_id);
                    """
                )
        finally:
            connection.close()

    def create_task(
        self,
        work_thread_id: int,
        title: str,
        *,
        created_at: datetime,
        is_done: bool = False,
    ) -> Task:
        """
        Create a new persistent task within a Work Thread.

        Args:
            work_thread_id: The ID of the parent Work Thread.
            title: The human-readable title of the task.
            created_at: Timezone-aware UTC datetime.
            is_done: Whether the task is initially completed (default False).

        Returns:
            The created Task instance.

        Raises:
            ValueError: If title is empty/invalid or created_at is timezone-naive.
            sqlite3.IntegrityError: If work_thread_id does not exist in work_threads.
        """
        _validate_non_empty_string(title, "title")
        _require_timezone_aware(created_at, "created_at")

        title_clean = title.strip()
        utc_created_at = created_at.astimezone(timezone.utc)
        done_flag = 1 if is_done else 0

        connection = self._connect()
        try:
            with connection:
                cursor = connection.execute(
                    """
                    INSERT INTO tasks (work_thread_id, title, created_at_utc, is_done)
                    VALUES (?, ?, ?, ?)
                    """,
                    (work_thread_id, title_clean, _format_utc(utc_created_at), done_flag),
                )
                task_id = cursor.lastrowid
        finally:
            connection.close()

        return Task(
            id=task_id,
            work_thread_id=work_thread_id,
            title=title_clean,
            created_at=utc_created_at,
            is_done=bool(is_done),
        )

    def get_task(self, task_id: int) -> Task | None:
        """Return one Task by id, or None if it does not exist."""
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT id, work_thread_id, title, created_at_utc, is_done FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
        finally:
            connection.close()

        return _task_from_row(row) if row is not None else None

    def list_tasks_for_work_thread(self, work_thread_id: int) -> list[Task]:
        """
        Return all tasks for a Work Thread in deterministic order:
        oldest first (by created_at_utc ASC, tie-broken by id ASC).
        """
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT id, work_thread_id, title, created_at_utc, is_done
                FROM tasks
                WHERE work_thread_id = ?
                ORDER BY created_at_utc ASC, id ASC
                """,
                (work_thread_id,),
            ).fetchall()
        finally:
            connection.close()

        return [_task_from_row(row) for row in rows]

    def rename_task(self, task_id: int, new_title: str) -> Task:
        """
        Rename an existing task.

        Args:
            task_id: The ID of the task to rename.
            new_title: The new non-empty title string.

        Returns:
            The updated Task instance.

        Raises:
            ValueError: If new_title is empty or whitespace-only.
            TaskNotFoundError: If the task does not exist.
        """
        _validate_non_empty_string(new_title, "new_title")
        new_title_clean = new_title.strip()

        connection = self._connect()
        try:
            with connection:
                cursor = connection.execute(
                    "UPDATE tasks SET title = ? WHERE id = ?",
                    (new_title_clean, task_id),
                )
                if cursor.rowcount == 0:
                    raise TaskNotFoundError(f"Task with id {task_id} not found.")

                row = connection.execute(
                    "SELECT id, work_thread_id, title, created_at_utc, is_done FROM tasks WHERE id = ?",
                    (task_id,),
                ).fetchone()
        finally:
            connection.close()

        return _task_from_row(row)

    def set_task_done(self, task_id: int, is_done: bool = True) -> Task:
        """
        Set the completed status of a task.

        Args:
            task_id: The ID of the task.
            is_done: The new completion status.

        Returns:
            The updated Task instance.

        Raises:
            TaskNotFoundError: If the task does not exist.
        """
        done_flag = 1 if is_done else 0
        connection = self._connect()
        try:
            with connection:
                cursor = connection.execute(
                    "UPDATE tasks SET is_done = ? WHERE id = ?",
                    (done_flag, task_id),
                )
                if cursor.rowcount == 0:
                    raise TaskNotFoundError(f"Task with id {task_id} not found.")

                row = connection.execute(
                    "SELECT id, work_thread_id, title, created_at_utc, is_done FROM tasks WHERE id = ?",
                    (task_id,),
                ).fetchone()
        finally:
            connection.close()

        return _task_from_row(row)

    def complete_task(self, task_id: int) -> Task:
        """Convenience method to mark a task as completed."""
        return self.set_task_done(task_id, is_done=True)

    def toggle_task(self, task_id: int) -> Task:
        """
        Toggle the completion status of a task (done -> undone, undone -> done).

        Args:
            task_id: The ID of the task to toggle.

        Returns:
            The updated Task instance.

        Raises:
            TaskNotFoundError: If the task does not exist.
        """
        connection = self._connect()
        try:
            with connection:
                row = connection.execute(
                    "SELECT is_done FROM tasks WHERE id = ?",
                    (task_id,),
                ).fetchone()
                if row is None:
                    raise TaskNotFoundError(f"Task with id {task_id} not found.")

                new_done = 0 if row["is_done"] else 1
                connection.execute(
                    "UPDATE tasks SET is_done = ? WHERE id = ?",
                    (new_done, task_id),
                )

                updated_row = connection.execute(
                    "SELECT id, work_thread_id, title, created_at_utc, is_done FROM tasks WHERE id = ?",
                    (task_id,),
                ).fetchone()
        finally:
            connection.close()

        return _task_from_row(updated_row)

    def delete_task(self, task_id: int) -> None:
        """
        Delete a task by ID.

        Args:
            task_id: The ID of the task to delete.

        Raises:
            TaskNotFoundError: If the task does not exist.
        """
        connection = self._connect()
        try:
            with connection:
                cursor = connection.execute(
                    "DELETE FROM tasks WHERE id = ?",
                    (task_id,),
                )
                if cursor.rowcount == 0:
                    raise TaskNotFoundError(f"Task with id {task_id} not found.")
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


# ============================================================================
# Helpers
# ============================================================================


def _validate_non_empty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _require_timezone_aware(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a datetime instance.")
    if value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware.")


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def _task_from_row(row: sqlite3.Row) -> Task:
    return Task(
        id=row["id"],
        work_thread_id=row["work_thread_id"],
        title=row["title"],
        created_at=_parse_utc(row["created_at_utc"]),
        is_done=bool(row["is_done"]),
    )
