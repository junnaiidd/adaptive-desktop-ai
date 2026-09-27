"""
Phase 2H: Durable Context Observation Layer

Persists a `PredictionResult` (Phase 2D/2F/2G's ephemeral inference
output) as a durable, historical, append-only `ContextObservation` row in
the application's existing SQLite database.

    Phase 2G: Completed StoredSession -> ... -> PredictionResult  (ephemeral)
                                                     |
                                                     v
    Phase 2H: PredictionResult -> ContextObservationStore -> SQLite  (durable)

What this module IS
--------------------
A narrow storage component. It accepts already-computed inference output
(a session_id, a predicted label, a probability distribution, and a
caller-supplied model-version string) and durably records it. It reads
those records back. Nothing else.

What this module is NOT
------------------------
- Not an inference engine: it never imports, constructs, or calls
  `ContextClassifier`, `ContextInferenceEngine`, or `FeatureExtractor`.
  It has no way to compute a prediction -- only to store one it is given.
- Not a training/retraining system: it never touches `TrainingExampleStore`,
  `capture_labeled_examples`, `ContextLabelStore`, or
  `retrain_from_examples`. Recording an observation never feeds back into
  the training corpus.
- Not a session manager: it never imports `ActivityRepository`,
  `SessionManager`, or `ActivityMonitoringService`, and never looks up or
  validates a session's activities. It only stores the *result* of
  inference that Phase 2G already performed elsewhere.
- Not a "Context" entity: this module persists CONTEXT OBSERVATIONS (one
  row per inference event, tied to one session), never a global,
  session-independent "Context #17 = Software Development" concept.
  Aggregating observations into a higher-level semantic entity is an
  explicitly future, unbuilt concern (Work Threads).
- Not a live/background service: nothing here polls, schedules, or
  triggers on a timer. `record_observation` is only ever called
  explicitly by a caller that already has a `PredictionResult` in hand.
- Not connected to the UI in any way.

Why a second, independent SQLite schema owner (not ActivityRepository)
--------------------------------------------------------------------------
`ActivityRepository` remains the sole owner of the `sessions` and
`activity_segments` tables and their business logic (session lifecycle,
activity CRUD). This module owns a *different* table
(`context_observations`) and creates it itself, using its own connection
and its own `CREATE TABLE IF NOT EXISTS`, pointed at the SAME database
file `ActivityRepository` uses by default. This is safe and does not
require ActivityRepository to change at all: SQLite table creation is
independent per table (verified: `CREATE TABLE ... FOREIGN KEY
REFERENCES sessions(id)` succeeds even before `sessions` exists), and
foreign-key enforcement (`PRAGMA foreign_keys = ON`) is a per-connection
setting that this module sets for itself, exactly as `ActivityRepository`
already does for its own connections. No context-observation business
logic was added to `ActivityRepository`, and no second database file was
created.

Append-only / immutability
---------------------------
Context observations are historical facts about "what did the classifier
say at this moment", not "current context state". There is deliberately
no update/upsert-by-session_id method and no delete method: calling
`record_observation` for the same session twice creates two separate
rows, both retrievable. This leaves room for a future capability
(windowed/live/re-run inference producing multiple observations per
session) without the schema or API needing to change.

Model provenance -- deliberately lightweight, no Phase 2D changes
-----------------------------------------------------------------------
`model_version` is a plain, caller-supplied string. This module has NO
dependency on `context_classifier.py` and does not derive this value
itself -- doing so would require either constructing a `ContextClassifier`
here (explicitly forbidden) or modifying Phase 2D to expose a new
attribute (also explicitly out of scope; Phase 2D was not modified).

Practical guidance for callers (not enforced by this module): a stable,
deterministic version string can already be built entirely from
`ContextClassifier`'s existing, unmodified public attributes, e.g.:

    model_version = (
        f"schema{classifier.SCHEMA_VERSION}"
        f"-rs{classifier.random_state}"
        f"-n{classifier.n_estimators}"
        f"-depth{classifier.max_depth}"
    )

This identifies *the trained configuration* (schema version + hyperparameters)
deterministically without corpus hashing or an experiment-tracking system --
exactly the "smallest deterministic identifier" the frozen contract calls
for. See `tests/test_context_observation_store.py` for a worked example
used in the end-to-end integration test.

Probabilities are stored, not interpreted
-------------------------------------------
The complete `class_probabilities` distribution is persisted as
deterministic JSON (sorted keys). This module does not rename it
"confidence", does not calibrate it, does not threshold it, and does not
compute an "uncertain" label. It is raw `RandomForestClassifier.predict_proba`
output, stored as historical information for whatever a later phase
chooses to do with it.

Session validation
-------------------
`session_id` existence is enforced entirely by SQLite's own foreign key
constraint (`context_observations.session_id REFERENCES sessions(id)`,
with `PRAGMA foreign_keys = ON`) -- this module performs no separate
Python-side existence check before inserting. This mirrors
`ActivityRepository.insert_activity`, which relies on the identical
mechanism for `activity_segments.session_id` today, and keeps this
module from duplicating logic that already lives in the database engine.
A reference to a session that was never created (or whose row is
otherwise absent) raises `sqlite3.IntegrityError` from the underlying
`INSERT`, unwrapped -- consistent with `ActivityRepository`, which never
wraps SQLite errors of its own.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ContextObservation:
    """One durably-persisted, historical inference observation."""

    id: int
    session_id: str
    predicted_label: str
    class_probabilities: dict[str, float]
    observed_at: datetime  # UTC, tz-aware
    model_version: str


class ContextObservationStore:
    """
    Append-only persistence for `ContextObservation` rows, in the same
    SQLite database `ActivityRepository` uses (a dedicated table,
    independently owned by this class).
    """

    def __init__(self, database_path: str | Path | None = None) -> None:
        default_path = Path(__file__).resolve().parents[2] / "data" / "activity.db"
        self.database_path = Path(database_path) if database_path else default_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        """Create the context_observations table and its index if they do not already exist."""
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS context_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    predicted_label TEXT NOT NULL,
                    class_probabilities_json TEXT NOT NULL,
                    observed_at_utc TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );

                CREATE INDEX IF NOT EXISTS idx_context_observations_session_id
                    ON context_observations (session_id);
                """
            )

    def record_observation(
        self,
        session_id: str,
        predicted_label: str,
        class_probabilities: dict[str, float],
        model_version: str,
        *,
        observed_at: datetime,
    ) -> ContextObservation:
        """
        Durably record one inference observation. This always INSERTs a
        new row; it never updates or replaces an existing one, even if
        called again for the same `session_id`.

        Args:
            session_id: The session this observation is about. Must
                already exist in the `sessions` table (enforced by the
                database's own foreign key, not by this method).
            predicted_label: The classifier's predicted context label.
            class_probabilities: The complete probability distribution
                over all classes the classifier knows, e.g.
                `{"Coding": 0.8, "Browsing": 0.2}`. Stored in full, never
                reduced to just the winning label.
            model_version: A caller-supplied, stable, deterministic
                identifier for the model that produced this prediction
                (see the module docstring for how to derive one without
                modifying Phase 2D).
            observed_at: Timezone-aware UTC timestamp for when this
                observation was generated/persisted (not session start,
                session end, or model training time).

        Returns:
            The persisted ContextObservation, including its assigned id.

        Raises:
            ValueError: on invalid session_id/predicted_label/
                class_probabilities/model_version, or a timezone-naive
                `observed_at`.
            sqlite3.IntegrityError: if `session_id` does not exist in the
                `sessions` table (propagated unmodified from SQLite's own
                foreign key enforcement).
        """
        _validate_non_empty_string(session_id, "session_id")
        _validate_non_empty_string(predicted_label, "predicted_label")
        _validate_class_probabilities(class_probabilities)
        _validate_non_empty_string(model_version, "model_version")
        _require_timezone_aware(observed_at, "observed_at")

        serialized_probabilities = _serialize_probabilities(class_probabilities)

        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO context_observations (
                    session_id, predicted_label, class_probabilities_json,
                    observed_at_utc, model_version
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    predicted_label,
                    serialized_probabilities,
                    _format_utc(observed_at),
                    model_version,
                ),
            )

        return ContextObservation(
            id=cursor.lastrowid,
            session_id=session_id,
            predicted_label=predicted_label,
            class_probabilities=dict(class_probabilities),
            observed_at=observed_at,
            model_version=model_version,
        )

    def get_observation(self, observation_id: int) -> ContextObservation | None:
        """Return one observation by id, or None if it does not exist."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM context_observations WHERE id = ?",
                (observation_id,),
            ).fetchone()
        return _observation_from_row(row) if row is not None else None

    def list_observations(self) -> list[ContextObservation]:
        """
        Return every observation, oldest first (chronological by
        `observed_at`, tie-broken by `id`), matching the ordering
        convention `ActivityRepository.list_activities`/`list_sessions`
        already use.
        """
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM context_observations ORDER BY observed_at_utc, id"
            ).fetchall()
        return [_observation_from_row(row) for row in rows]

    def list_observations_for_session(self, session_id: str) -> list[ContextObservation]:
        """
        Return only the observations for one session, oldest first (same
        ordering convention as `list_observations`). Never includes
        observations belonging to any other session.
        """
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM context_observations WHERE session_id = ? ORDER BY observed_at_utc, id",
                (session_id,),
            ).fetchall()
        return [_observation_from_row(row) for row in rows]

    def list_recent_observations(self, limit: int = 50) -> list[ContextObservation]:
        """
        Return the newest observations first, matching the ordering
        convention `ActivityRepository.list_recent_activities` already
        uses for its analogous "most recent" query.
        """
        if limit <= 0:
            raise ValueError("The recent observation limit must be positive.")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM context_observations
                ORDER BY observed_at_utc DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [_observation_from_row(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


# ============================================================================
# Validation & small internal helpers
# ============================================================================


def _validate_non_empty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _validate_class_probabilities(class_probabilities: dict[str, float]) -> None:
    if not isinstance(class_probabilities, dict) or not class_probabilities:
        raise ValueError("class_probabilities must be a non-empty dict[str, float].")
    for label, probability in class_probabilities.items():
        if not isinstance(label, str) or not label:
            raise ValueError(f"class_probabilities keys must be non-empty strings, got {label!r}.")
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError(f"class_probabilities values must be numeric, got {probability!r} for {label!r}.")
        if not math.isfinite(float(probability)):
            raise ValueError(f"class_probabilities values must be finite, got {probability!r} for {label!r}.")


def _require_timezone_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware.")


def _serialize_probabilities(class_probabilities: dict[str, float]) -> str:
    """Deterministic JSON: sorted keys, plain float values, no exotic types."""
    return json.dumps({label: float(value) for label, value in class_probabilities.items()}, sort_keys=True)


def _deserialize_probabilities(serialized: str) -> dict[str, float]:
    """Parse persisted JSON. Malformed content raises json.JSONDecodeError, unmodified -- never silently accepted."""
    payload = json.loads(serialized)
    if not isinstance(payload, dict):
        raise ValueError(f"Corrupt class_probabilities_json: expected a JSON object, got {type(payload).__name__}.")
    return {str(key): float(value) for key, value in payload.items()}


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def _observation_from_row(row: sqlite3.Row) -> ContextObservation:
    return ContextObservation(
        id=row["id"],
        session_id=row["session_id"],
        predicted_label=row["predicted_label"],
        class_probabilities=_deserialize_probabilities(row["class_probabilities_json"]),
        observed_at=_parse_utc(row["observed_at_utc"]),
        model_version=row["model_version"],
    )
