"""
Milestone B: Persistent, User-Created Work Threads and Observation Associations.

Introduces the first persistent, user-created `WorkThread` entity and allows
explicit association of existing `ContextObservation` records with it.

What this module IS:
--------------------
A narrow storage component for explicitly user-created Work Threads and explicit
associations to historical Context Observations. It manages two tables in the
application's existing SQLite database:
  - `work_threads`
  - `work_thread_observations`

What this module is NOT:
------------------------
- Not an inference or clustering engine: it never automatically detects,
  predicts, clusters, or infers a work thread.
- Not a task management system: no tasks, deadlines, completion statuses, or
  sub-tasks.
- Not a workspace restorer: no window/tab saving or restoration.
- Not connected to ML models, embeddings, similarity matching, or LLMs.
- Work threads are strictly and exclusively user-created.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


class WorkThreadError(Exception):
    """Base exception for WorkThreadStore operations."""


class WorkThreadHasAssociationsError(WorkThreadError):
    """Raised when attempting to delete a work thread that still has associated observations."""


class WorkThreadNotFoundError(WorkThreadError, KeyError):
    """Raised when a work thread cannot be found by ID."""


class AssociationNotFoundError(WorkThreadError, KeyError):
    """Raised when an association cannot be found by ID."""


@dataclass(frozen=True, slots=True)
class WorkThread:
    """A persistent, user-created work thread."""

    id: int
    name: str
    created_at: datetime  # UTC, timezone-aware

    @property
    def work_thread_id(self) -> int:
        """Alias for id to ensure ergonomic compatibility."""
        return self.id


@dataclass(frozen=True, slots=True)
class WorkThreadObservation:
    """An association record connecting a WorkThread to a ContextObservation."""

    id: int
    work_thread_id: int
    observation_id: int
    associated_at: datetime  # UTC, timezone-aware

    @property
    def association_id(self) -> int:
        """Alias for id to ensure ergonomic compatibility."""
        return self.id


class WorkThreadStore:
    """
    Persistence layer for user-created Work Threads and their associations
    with Context Observations, in the application's shared SQLite database.
    """

    def __init__(self, database_path: str | Path | None = None) -> None:
        default_path = Path(__file__).resolve().parents[2] / "data" / "activity.db"
        self.database_path = Path(database_path) if database_path else default_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        """Create the work_threads and work_thread_observations tables if they do not exist."""
        connection = self._connect()
        try:
            with connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS work_threads (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        created_at_utc TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS work_thread_observations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        work_thread_id INTEGER NOT NULL,
                        observation_id INTEGER NOT NULL,
                        associated_at_utc TEXT NOT NULL,
                        FOREIGN KEY (work_thread_id) REFERENCES work_threads(id),
                        FOREIGN KEY (observation_id) REFERENCES context_observations(id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_work_thread_observations_thread_id
                        ON work_thread_observations (work_thread_id);

                    CREATE INDEX IF NOT EXISTS idx_work_thread_observations_observation_id
                        ON work_thread_observations (observation_id);
                    """
                )
        finally:
            connection.close()

    def create_work_thread(self, name: str, *, created_at: datetime) -> WorkThread:
        """
        Create a new persistent, user-defined Work Thread.

        Args:
            name: The human-readable name for the work thread.
            created_at: Timezone-aware UTC datetime.

        Returns:
            The created WorkThread instance.

        Raises:
            ValueError: If name is empty/invalid or created_at is timezone-naive.
        """
        _validate_non_empty_string(name, "name")
        _require_timezone_aware(created_at, "created_at")

        name_clean = name.strip()
        utc_created_at = created_at.astimezone(timezone.utc)
        connection = self._connect()
        try:
            with connection:
                cursor = connection.execute(
                    """
                    INSERT INTO work_threads (name, created_at_utc)
                    VALUES (?, ?)
                    """,
                    (name_clean, _format_utc(utc_created_at)),
                )
                thread_id = cursor.lastrowid
        finally:
            connection.close()

        return WorkThread(id=thread_id, name=name_clean, created_at=utc_created_at)

    def get_work_thread(self, work_thread_id: int) -> WorkThread | None:
        """Return one WorkThread by id, or None if it does not exist."""
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT id, name, created_at_utc FROM work_threads WHERE id = ?",
                (work_thread_id,),
            ).fetchone()
        finally:
            connection.close()

        return _work_thread_from_row(row) if row is not None else None

    def list_work_threads(self) -> list[WorkThread]:
        """
        Return all work threads in deterministic order:
        oldest first (by created_at_utc, tie-broken by id).
        """
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT id, name, created_at_utc FROM work_threads ORDER BY created_at_utc ASC, id ASC"
            ).fetchall()
        finally:
            connection.close()

        return [_work_thread_from_row(row) for row in rows]

    def rename_work_thread(self, work_thread_id: int, new_name: str) -> WorkThread:
        """
        Rename an existing Work Thread.

        Args:
            work_thread_id: The ID of the thread to rename.
            new_name: The new human-readable name.

        Returns:
            The updated WorkThread instance.

        Raises:
            ValueError: If new_name is empty/invalid.
            WorkThreadNotFoundError: If the work thread does not exist.
        """
        _validate_non_empty_string(new_name, "new_name")
        new_name_clean = new_name.strip()

        connection = self._connect()
        try:
            with connection:
                cursor = connection.execute(
                    "UPDATE work_threads SET name = ? WHERE id = ?",
                    (new_name_clean, work_thread_id),
                )
                if cursor.rowcount == 0:
                    raise WorkThreadNotFoundError(f"Work thread with id {work_thread_id} not found.")

                row = connection.execute(
                    "SELECT id, name, created_at_utc FROM work_threads WHERE id = ?",
                    (work_thread_id,),
                ).fetchone()
        finally:
            connection.close()

        return _work_thread_from_row(row)

    def delete_work_thread(self, work_thread_id: int) -> None:
        """
        Delete a Work Thread, blocking deletion if it has any associated observations.

        Args:
            work_thread_id: The ID of the thread to delete.

        Raises:
            WorkThreadNotFoundError: If the thread does not exist.
            WorkThreadHasAssociationsError: If the thread has existing observation associations.
        """
        connection = self._connect()
        try:
            with connection:
                row = connection.execute(
                    "SELECT id FROM work_threads WHERE id = ?",
                    (work_thread_id,),
                ).fetchone()
                if row is None:
                    raise WorkThreadNotFoundError(f"Work thread with id {work_thread_id} not found.")

                assoc_count = connection.execute(
                    "SELECT COUNT(*) FROM work_thread_observations WHERE work_thread_id = ?",
                    (work_thread_id,),
                ).fetchone()[0]
                if assoc_count > 0:
                    raise WorkThreadHasAssociationsError(
                        f"Cannot delete work thread {work_thread_id}: it has {assoc_count} associated observation(s)."
                    )

                connection.execute("DELETE FROM work_threads WHERE id = ?", (work_thread_id,))
        finally:
            connection.close()

    def associate_observation(
        self,
        work_thread_id: int,
        observation_id: int,
        *,
        associated_at: datetime,
    ) -> WorkThreadObservation:
        """
        Explicitly associate an existing ContextObservation with a WorkThread.

        Foreign keys are strictly enforced on every connection; if either
        work_thread_id or observation_id does not exist, sqlite3.IntegrityError
        is raised by the database.

        Args:
            work_thread_id: The ID of the work thread.
            observation_id: The ID of the context observation.
            associated_at: Timezone-aware UTC datetime.

        Returns:
            The created WorkThreadObservation record.

        Raises:
            ValueError: If associated_at is timezone-naive.
            sqlite3.IntegrityError: If foreign key validation fails.
        """
        _require_timezone_aware(associated_at, "associated_at")
        utc_associated_at = associated_at.astimezone(timezone.utc)

        connection = self._connect()
        try:
            with connection:
                cursor = connection.execute(
                    """
                    INSERT INTO work_thread_observations (
                        work_thread_id, observation_id, associated_at_utc
                    ) VALUES (?, ?, ?)
                    """,
                    (work_thread_id, observation_id, _format_utc(utc_associated_at)),
                )
                assoc_id = cursor.lastrowid
        finally:
            connection.close()

        return WorkThreadObservation(
            id=assoc_id,
            work_thread_id=work_thread_id,
            observation_id=observation_id,
            associated_at=utc_associated_at,
        )

    def remove_association(self, association_id: int) -> None:
        """
        Remove an association between a work thread and an observation.

        Args:
            association_id: The ID of the association record in work_thread_observations.

        Raises:
            AssociationNotFoundError: If the association does not exist.
        """
        connection = self._connect()
        try:
            with connection:
                cursor = connection.execute(
                    "DELETE FROM work_thread_observations WHERE id = ?",
                    (association_id,),
                )
                if cursor.rowcount == 0:
                    raise AssociationNotFoundError(f"Association with id {association_id} not found.")
        finally:
            connection.close()

    def list_observations_for_work_thread(self, work_thread_id: int) -> list[WorkThreadObservation]:
        """
        Return all associations for a given work thread in deterministic order:
        chronological (by associated_at_utc ASC, tie-broken by id ASC).
        """
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT id, work_thread_id, observation_id, associated_at_utc
                FROM work_thread_observations
                WHERE work_thread_id = ?
                ORDER BY associated_at_utc ASC, id ASC
                """,
                (work_thread_id,),
            ).fetchall()
        finally:
            connection.close()

        return [_association_from_row(row) for row in rows]

    def list_work_threads_for_observation(self, observation_id: int) -> list[WorkThread]:
        """
        Return all work threads associated with a given observation in deterministic order:
        by association time (associated_at_utc ASC, tie-broken by association id ASC).
        """
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT wt.id, wt.name, wt.created_at_utc
                FROM work_threads wt
                JOIN work_thread_observations wto ON wt.id = wto.work_thread_id
                WHERE wto.observation_id = ?
                ORDER BY wto.associated_at_utc ASC, wto.id ASC
                """,
                (observation_id,),
            ).fetchall()
        finally:
            connection.close()

        return [_work_thread_from_row(row) for row in rows]

    def list_associations_for_work_thread(self, work_thread_id: int) -> list[WorkThreadObservation]:
        """Convenience alias for list_observations_for_work_thread."""
        return self.list_observations_for_work_thread(work_thread_id)

    def list_associations_for_observation(self, observation_id: int) -> list[WorkThreadObservation]:
        """Return the raw association records for a given observation."""
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT id, work_thread_id, observation_id, associated_at_utc
                FROM work_thread_observations
                WHERE observation_id = ?
                ORDER BY associated_at_utc ASC, id ASC
                """,
                (observation_id,),
            ).fetchall()
        finally:
            connection.close()

        return [_association_from_row(row) for row in rows]

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


def _work_thread_from_row(row: sqlite3.Row) -> WorkThread:
    return WorkThread(
        id=row["id"],
        name=row["name"],
        created_at=_parse_utc(row["created_at_utc"]),
    )


def _association_from_row(row: sqlite3.Row) -> WorkThreadObservation:
    return WorkThreadObservation(
        id=row["id"],
        work_thread_id=row["work_thread_id"],
        observation_id=row["observation_id"],
        associated_at=_parse_utc(row["associated_at_utc"]),
    )
