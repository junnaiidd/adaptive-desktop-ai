"""
Comprehensive tests for Phase 2H: Durable Context Observation Layer
(`app/ml/context_observation_store.py`).

Covers all items from the frozen Phase 2H test requirements (1-35),
plus the mandatory Phase 2G -> Phase 2H integration test, plus
architectural regression guards in the established Phase 2E/2F/2G style.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.database.activity_repository import ActivityRepository, ActivitySegment, StoredSession
from app.ml.context_observation_store import ContextObservation, ContextObservationStore

# ============================================================================
# Test helpers
# ============================================================================


def _ts(seconds: int = 0, hour: int = 9, day: int = 1) -> datetime:
    return datetime(2026, 1, day, hour=hour, tzinfo=timezone.utc) + timedelta(seconds=seconds)


def _repo_with_session(tmp_path: Path, session_id: str = "sess-1", db_name: str = "activity.db") -> tuple[ActivityRepository, Path]:
    """A real ActivityRepository with one completed session already in it."""
    db_path = tmp_path / db_name
    repository = ActivityRepository(db_path)
    started = _ts()
    ended = started + timedelta(seconds=1800)
    repository.start_session(StoredSession(session_id, started, None))
    repository.end_session(StoredSession(session_id, started, ended))
    return repository, db_path


PROBS = {"Focused Work": 0.8, "Browsing": 0.2}


def _sample_model_version() -> str:
    """A worked example of the caller-side derivation the module docstring documents."""
    schema_version, random_state, n_estimators, max_depth = 1, 42, 100, None
    return f"schema{schema_version}-rs{random_state}-n{n_estimators}-depth{max_depth}"


# ============================================================================
# 1. Database initialization creates context_observations
# ============================================================================


def test_initialization_creates_context_observations_table(tmp_path: Path) -> None:
    db_path = tmp_path / "activity.db"
    ContextObservationStore(db_path)

    connection = sqlite3.connect(db_path)
    tables = {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    connection.close()

    assert "context_observations" in tables


def test_initialization_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "activity.db"
    ContextObservationStore(db_path)
    ContextObservationStore(db_path)  # must not raise or duplicate anything

    store = ContextObservationStore(db_path)
    assert store.list_observations() == []


# ============================================================================
# 2-4. Record / read / round-trip
# ============================================================================


def test_record_one_observation(tmp_path: Path) -> None:
    _, db_path = _repo_with_session(tmp_path)
    store = ContextObservationStore(db_path)

    observation = store.record_observation(
        "sess-1", "Focused Work", PROBS, _sample_model_version(), observed_at=_ts()
    )

    assert isinstance(observation, ContextObservation)
    assert observation.id is not None


def test_read_one_observation(tmp_path: Path) -> None:
    _, db_path = _repo_with_session(tmp_path)
    store = ContextObservationStore(db_path)
    recorded = store.record_observation("sess-1", "Focused Work", PROBS, _sample_model_version(), observed_at=_ts())

    fetched = store.get_observation(recorded.id)

    assert fetched == recorded


def test_all_fields_round_trip_correctly(tmp_path: Path) -> None:
    _, db_path = _repo_with_session(tmp_path)
    store = ContextObservationStore(db_path)
    observed_at = _ts(hour=14, day=3)

    recorded = store.record_observation(
        "sess-1", "Deep Coding", {"Deep Coding": 0.91, "Browsing": 0.09}, "rf-abc123", observed_at=observed_at
    )
    fetched = store.get_observation(recorded.id)

    assert fetched.session_id == "sess-1"
    assert fetched.predicted_label == "Deep Coding"
    assert fetched.class_probabilities == {"Deep Coding": 0.91, "Browsing": 0.09}
    assert fetched.observed_at == observed_at
    assert fetched.model_version == "rf-abc123"


# ============================================================================
# 5. Full probability distribution round-trips
# ============================================================================


def test_full_probability_distribution_round_trips(tmp_path: Path) -> None:
    _, db_path = _repo_with_session(tmp_path)
    store = ContextObservationStore(db_path)
    full_distribution = {"Coding": 0.5, "Browsing": 0.3, "Gaming": 0.15, "Research": 0.05}

    recorded = store.record_observation("sess-1", "Coding", full_distribution, "v1", observed_at=_ts())
    fetched = store.get_observation(recorded.id)

    assert fetched.class_probabilities == full_distribution
    assert len(fetched.class_probabilities) == 4  # every class preserved, not just the winner


# ============================================================================
# 6. Deterministic JSON serialization
# ============================================================================


def test_deterministic_json_serialization_in_the_database(tmp_path: Path) -> None:
    db_path_a = tmp_path / "a.db"
    db_path_b = tmp_path / "b.db"
    repo_a = ActivityRepository(db_path_a)
    repo_b = ActivityRepository(db_path_b)
    for repo in (repo_a, repo_b):
        repo.start_session(StoredSession("sess-1", _ts(), None))
        repo.end_session(StoredSession("sess-1", _ts(), _ts(1800)))

    store_a = ContextObservationStore(db_path_a)
    store_b = ContextObservationStore(db_path_b)
    probs = {"Zeta": 0.1, "Alpha": 0.9}  # intentionally out of alphabetical order

    store_a.record_observation("sess-1", "Alpha", probs, "v1", observed_at=_ts())
    store_b.record_observation("sess-1", "Alpha", probs, "v1", observed_at=_ts())

    connection_a = sqlite3.connect(db_path_a)
    connection_b = sqlite3.connect(db_path_b)
    raw_a = connection_a.execute("SELECT class_probabilities_json FROM context_observations").fetchone()[0]
    raw_b = connection_b.execute("SELECT class_probabilities_json FROM context_observations").fetchone()[0]
    connection_a.close()
    connection_b.close()

    assert raw_a == raw_b
    assert raw_a == json.dumps(probs, sort_keys=True)  # sorted-key canonical form, not insertion order


# ============================================================================
# 7. UTC timestamp round-trip
# ============================================================================


def test_utc_timestamp_round_trip(tmp_path: Path) -> None:
    _, db_path = _repo_with_session(tmp_path)
    store = ContextObservationStore(db_path)
    observed_at = datetime(2026, 6, 15, 13, 45, 30, tzinfo=timezone.utc)

    recorded = store.record_observation("sess-1", "X", {"X": 1.0}, "v1", observed_at=observed_at)

    assert recorded.observed_at == observed_at
    assert recorded.observed_at.tzinfo is not None


def test_non_utc_timezone_is_normalized_to_utc_on_round_trip(tmp_path: Path) -> None:
    from datetime import timezone as tz

    _, db_path = _repo_with_session(tmp_path)
    store = ContextObservationStore(db_path)
    plus_five = tz(timedelta(hours=5))
    observed_at = datetime(2026, 6, 15, 18, 45, 30, tzinfo=plus_five)  # same instant as 13:45:30 UTC

    recorded = store.record_observation("sess-1", "X", {"X": 1.0}, "v1", observed_at=observed_at)
    fetched = store.get_observation(recorded.id)

    assert fetched.observed_at == datetime(2026, 6, 15, 13, 45, 30, tzinfo=timezone.utc)


# ============================================================================
# 8. model_version persistence
# ============================================================================


def test_model_version_persists_exactly(tmp_path: Path) -> None:
    _, db_path = _repo_with_session(tmp_path)
    store = ContextObservationStore(db_path)

    recorded = store.record_observation("sess-1", "X", {"X": 1.0}, "schema1-rs42-n100-depthNone", observed_at=_ts())
    fetched = store.get_observation(recorded.id)

    assert fetched.model_version == "schema1-rs42-n100-depthNone"


# ============================================================================
# 9-11. Session existence / FK enforcement
# ============================================================================


def test_nonexistent_session_is_rejected(tmp_path: Path) -> None:
    db_path = tmp_path / "activity.db"
    ActivityRepository(db_path)  # creates sessions table, but no rows
    store = ContextObservationStore(db_path)

    with pytest.raises(sqlite3.IntegrityError):
        store.record_observation("does-not-exist", "X", {"X": 1.0}, "v1", observed_at=_ts())


def test_existing_session_accepts_observation(tmp_path: Path) -> None:
    _, db_path = _repo_with_session(tmp_path)
    store = ContextObservationStore(db_path)

    observation = store.record_observation("sess-1", "X", {"X": 1.0}, "v1", observed_at=_ts())

    assert observation.session_id == "sess-1"


def test_fk_relationship_is_actually_enforced_by_sqlite(tmp_path: Path) -> None:
    """Confirm PRAGMA foreign_keys=ON is genuinely in effect, not just assumed."""
    db_path = tmp_path / "activity.db"
    ActivityRepository(db_path)
    store = ContextObservationStore(db_path)

    connection = sqlite3.connect(db_path)
    fk_status = connection.execute("PRAGMA foreign_keys").fetchone()
    connection.close()
    # PRAGMA foreign_keys is per-connection, not persisted; the store's OWN
    # connections set it every time -- verify indirectly via rejection:
    with pytest.raises(sqlite3.IntegrityError):
        store.record_observation("ghost-session", "X", {"X": 1.0}, "v1", observed_at=_ts())


def test_no_orphaned_observation_is_created_on_rejection(tmp_path: Path) -> None:
    db_path = tmp_path / "activity.db"
    ActivityRepository(db_path)
    store = ContextObservationStore(db_path)

    with pytest.raises(sqlite3.IntegrityError):
        store.record_observation("ghost-session", "X", {"X": 1.0}, "v1", observed_at=_ts())

    assert store.list_observations() == []


# ============================================================================
# 12-14. Multiple observations per session, append-only, no upsert
# ============================================================================


def test_multiple_observations_for_the_same_session_are_allowed(tmp_path: Path) -> None:
    _, db_path = _repo_with_session(tmp_path)
    store = ContextObservationStore(db_path)

    first = store.record_observation("sess-1", "Coding", {"Coding": 0.9, "Browsing": 0.1}, "v1", observed_at=_ts(hour=9))
    second = store.record_observation("sess-1", "Browsing", {"Coding": 0.2, "Browsing": 0.8}, "v1", observed_at=_ts(hour=10))

    assert first.id != second.id
    all_for_session = store.list_observations_for_session("sess-1")
    assert len(all_for_session) == 2


def test_append_only_no_update_method_exists() -> None:
    """The store must expose no update/mutate method for existing observations."""
    for forbidden in ("update_observation", "upsert_observation", "replace_observation"):
        assert not hasattr(ContextObservationStore, forbidden)


def test_recording_twice_for_same_session_does_not_overwrite_the_first(tmp_path: Path) -> None:
    _, db_path = _repo_with_session(tmp_path)
    store = ContextObservationStore(db_path)

    first = store.record_observation("sess-1", "Coding", {"Coding": 1.0}, "v1", observed_at=_ts(hour=9))
    store.record_observation("sess-1", "Browsing", {"Browsing": 1.0}, "v1", observed_at=_ts(hour=10))

    still_there = store.get_observation(first.id)
    assert still_there == first  # untouched by the second call


# ============================================================================
# 15-18. Read semantics / ordering
# ============================================================================


def test_list_observations_deterministic_ordering(tmp_path: Path) -> None:
    _, db_path = _repo_with_session(tmp_path)
    store = ContextObservationStore(db_path)

    store.record_observation("sess-1", "C", {"C": 1.0}, "v1", observed_at=_ts(hour=12))
    store.record_observation("sess-1", "A", {"A": 1.0}, "v1", observed_at=_ts(hour=9))
    store.record_observation("sess-1", "B", {"B": 1.0}, "v1", observed_at=_ts(hour=10))

    labels_in_order = [o.predicted_label for o in store.list_observations()]
    assert labels_in_order == ["A", "B", "C"]  # oldest-first, as documented


def test_list_observations_for_session_filters_correctly(tmp_path: Path) -> None:
    repository, db_path = _repo_with_session(tmp_path, session_id="sess-1")
    repository.start_session(StoredSession("sess-2", _ts(), None))
    repository.end_session(StoredSession("sess-2", _ts(), _ts(1800)))
    store = ContextObservationStore(db_path)

    store.record_observation("sess-1", "Coding", {"Coding": 1.0}, "v1", observed_at=_ts())
    store.record_observation("sess-2", "Browsing", {"Browsing": 1.0}, "v1", observed_at=_ts())

    only_session_1 = store.list_observations_for_session("sess-1")
    assert len(only_session_1) == 1
    assert only_session_1[0].predicted_label == "Coding"


def test_no_cross_session_leakage(tmp_path: Path) -> None:
    repository, db_path = _repo_with_session(tmp_path, session_id="sess-1")
    repository.start_session(StoredSession("sess-2", _ts(), None))
    repository.end_session(StoredSession("sess-2", _ts(), _ts(1800)))
    store = ContextObservationStore(db_path)

    store.record_observation("sess-1", "A", {"A": 1.0}, "v1", observed_at=_ts())
    store.record_observation("sess-2", "B", {"B": 1.0}, "v1", observed_at=_ts())

    for observation in store.list_observations_for_session("sess-1"):
        assert observation.session_id == "sess-1"
    for observation in store.list_observations_for_session("sess-2"):
        assert observation.session_id == "sess-2"


def test_tie_timestamps_use_stable_id_ordering(tmp_path: Path) -> None:
    _, db_path = _repo_with_session(tmp_path)
    store = ContextObservationStore(db_path)
    same_instant = _ts(hour=9)

    first = store.record_observation("sess-1", "First", {"First": 1.0}, "v1", observed_at=same_instant)
    second = store.record_observation("sess-1", "Second", {"Second": 1.0}, "v1", observed_at=same_instant)

    ordered = store.list_observations_for_session("sess-1")
    assert [o.id for o in ordered] == [first.id, second.id]  # insertion/id order breaks the tie


def test_list_recent_observations_orders_newest_first(tmp_path: Path) -> None:
    _, db_path = _repo_with_session(tmp_path)
    store = ContextObservationStore(db_path)

    store.record_observation("sess-1", "Old", {"Old": 1.0}, "v1", observed_at=_ts(hour=9))
    store.record_observation("sess-1", "New", {"New": 1.0}, "v1", observed_at=_ts(hour=15))

    recent = store.list_recent_observations(limit=10)
    assert [o.predicted_label for o in recent] == ["New", "Old"]


def test_list_recent_observations_respects_limit(tmp_path: Path) -> None:
    _, db_path = _repo_with_session(tmp_path)
    store = ContextObservationStore(db_path)
    for i in range(5):
        store.record_observation("sess-1", f"label-{i}", {f"label-{i}": 1.0}, "v1", observed_at=_ts(hour=9, seconds=i))

    recent = store.list_recent_observations(limit=2)
    assert len(recent) == 2


def test_list_recent_observations_rejects_non_positive_limit(tmp_path: Path) -> None:
    _, db_path = _repo_with_session(tmp_path)
    store = ContextObservationStore(db_path)

    with pytest.raises(ValueError, match="positive"):
        store.list_recent_observations(limit=0)


# ============================================================================
# 19-20. Empty store / restart durability
# ============================================================================


def test_empty_store_behavior(tmp_path: Path) -> None:
    db_path = tmp_path / "activity.db"
    ActivityRepository(db_path)
    store = ContextObservationStore(db_path)

    assert store.list_observations() == []
    assert store.list_observations_for_session("anything") == []
    assert store.get_observation(999) is None


def test_persistence_survives_creating_a_new_store_instance(tmp_path: Path) -> None:
    _, db_path = _repo_with_session(tmp_path)
    store_a = ContextObservationStore(db_path)
    recorded = store_a.record_observation("sess-1", "Coding", PROBS, "v1", observed_at=_ts())

    store_b = ContextObservationStore(db_path)  # simulates a fresh process/restart

    fetched = store_b.get_observation(recorded.id)
    assert fetched == recorded
    assert store_b.list_observations_for_session("sess-1") == [recorded]


# ============================================================================
# 21-22. Malformed data / failure safety
# ============================================================================


def test_malformed_probability_json_is_handled_safely(tmp_path: Path) -> None:
    """A row corrupted at the raw SQL level must fail loudly on read, never silently return wrong data."""
    _, db_path = _repo_with_session(tmp_path)
    store = ContextObservationStore(db_path)
    recorded = store.record_observation("sess-1", "X", {"X": 1.0}, "v1", observed_at=_ts())

    connection = sqlite3.connect(db_path)
    connection.execute(
        "UPDATE context_observations SET class_probabilities_json = ? WHERE id = ?",
        ("{not valid json", recorded.id),
    )
    connection.commit()
    connection.close()

    with pytest.raises(json.JSONDecodeError):
        store.get_observation(recorded.id)


def test_failed_write_does_not_corrupt_existing_observations(tmp_path: Path) -> None:
    _, db_path = _repo_with_session(tmp_path)
    store = ContextObservationStore(db_path)
    good = store.record_observation("sess-1", "Good", {"Good": 1.0}, "v1", observed_at=_ts())

    with pytest.raises(sqlite3.IntegrityError):
        store.record_observation("ghost-session", "Bad", {"Bad": 1.0}, "v1", observed_at=_ts())

    assert store.get_observation(good.id) == good
    assert len(store.list_observations()) == 1


def test_failed_write_due_to_invalid_input_does_not_persist_anything(tmp_path: Path) -> None:
    _, db_path = _repo_with_session(tmp_path)
    store = ContextObservationStore(db_path)

    with pytest.raises(ValueError):
        store.record_observation("sess-1", "", {"X": 1.0}, "v1", observed_at=_ts())  # empty label

    assert store.list_observations() == []


# ============================================================================
# 23-24. Parameterized SQL safety / special characters
# ============================================================================


def test_parameterized_query_prevents_sql_injection_in_session_id(tmp_path: Path) -> None:
    _, db_path = _repo_with_session(tmp_path)
    store = ContextObservationStore(db_path)
    malicious_session_id = "sess-1'; DROP TABLE context_observations; --"

    # Not a real session, so this correctly fails via FK -- the point is
    # that it fails SAFELY (a clean IntegrityError) rather than executing
    # injected SQL and corrupting the schema.
    with pytest.raises(sqlite3.IntegrityError):
        store.record_observation(malicious_session_id, "X", {"X": 1.0}, "v1", observed_at=_ts())

    # The table must still exist and be queryable.
    assert store.list_observations() == []


def test_special_characters_in_labels_and_model_version_round_trip(tmp_path: Path) -> None:
    _, db_path = _repo_with_session(tmp_path)
    store = ContextObservationStore(db_path)
    tricky_label = "Coding / Research (50%) — \"quoted\" 🚀"
    tricky_version = "rf-v1.0'; DROP--"

    recorded = store.record_observation("sess-1", tricky_label, {tricky_label: 1.0}, tricky_version, observed_at=_ts())
    fetched = store.get_observation(recorded.id)

    assert fetched.predicted_label == tricky_label
    assert fetched.model_version == tricky_version
    assert fetched.class_probabilities == {tricky_label: 1.0}


# ============================================================================
# 25-26. No feature vector stored / no mutation of activity data
# ============================================================================


def test_no_feature_vector_columns_exist_in_the_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "activity.db"
    ActivityRepository(db_path)
    ContextObservationStore(db_path)

    connection = sqlite3.connect(db_path)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(context_observations)").fetchall()}
    connection.close()

    assert columns == {"id", "session_id", "predicted_label", "class_probabilities_json", "observed_at_utc", "model_version"}
    for forbidden in ("feature_names", "feature_values", "metadata", "cluster_id", "cluster_label", "context_id", "work_thread_id", "task_id", "confirmed", "confidence", "training_example_id"):
        assert forbidden not in columns


def test_recording_an_observation_does_not_mutate_session_or_activity_rows(tmp_path: Path) -> None:
    repository, db_path = _repo_with_session(tmp_path)
    repository.insert_activity(
        ActivitySegment(
            session_id="sess-1", started_at=_ts(), ended_at=_ts(600),
            application="VSCode", process_name="Code.exe", window_title="x", duration_seconds=600.0,
        )
    )
    sessions_before = repository.list_sessions()
    activities_before = repository.list_activities()
    store = ContextObservationStore(db_path)

    store.record_observation("sess-1", "Coding", PROBS, "v1", observed_at=_ts())

    assert repository.list_sessions() == sessions_before
    assert repository.list_activities() == activities_before


# ============================================================================
# 27-30. Store does not perform inference / does not train / no model artifacts
# ============================================================================


def test_store_has_no_inference_capability() -> None:
    for forbidden in ("predict", "predict_many", "infer", "classify"):
        assert not hasattr(ContextObservationStore, forbidden)


def test_store_has_no_training_capability() -> None:
    for forbidden in ("fit", "train", "retrain", "evaluate", "capture_labeled_examples"):
        assert not hasattr(ContextObservationStore, forbidden)


def test_recording_observations_creates_no_model_artifact(tmp_path: Path) -> None:
    _, db_path = _repo_with_session(tmp_path)
    store = ContextObservationStore(db_path)
    files_before = set(tmp_path.rglob("*"))

    store.record_observation("sess-1", "Coding", PROBS, "v1", observed_at=_ts())

    files_after = set(tmp_path.rglob("*"))
    assert files_after == files_before  # no new file created anywhere (only the existing db was written to)


# ============================================================================
# 31-32. Phase 2F/2G remain unchanged (behavioral proof, not just "we didn't touch the file")
# ============================================================================


def test_phase_2f_inference_engine_still_works_unmodified(tmp_path: Path) -> None:
    from app.ml.context_classifier import ContextClassifier
    from app.ml.context_inference import ContextInferenceEngine
    from app.ml.feature_engineering import FeatureVector

    names = ("f1", "f2")
    vectors = [FeatureVector(f"a{i}", names, (0.0 + i * 0.01, 0.0), {}) for i in range(4)] + [
        FeatureVector(f"b{i}", names, (50.0 + i * 0.01, 0.0), {}) for i in range(4)
    ]
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, ["A"] * 4 + ["B"] * 4)
    engine = ContextInferenceEngine(clf)

    result = engine.predict(FeatureVector("x", names, (0.01, 0.0), {}))
    assert result.predicted_label in {"A", "B"}


def test_phase_2g_integrator_still_works_unmodified(tmp_path: Path) -> None:
    from app.ml.context_classifier import ContextClassifier
    from app.ml.context_inference import ContextInferenceEngine
    from app.ml.feature_engineering import FeatureExtractor, FeatureVector
    from app.ml.session_context_integration import SessionContextIntegrator

    repository, db_path = _repo_with_session(tmp_path)
    repository.insert_activity(
        ActivitySegment(
            session_id="sess-1", started_at=_ts(), ended_at=_ts(1800),
            application="VSCode", process_name="Code.exe", window_title="x", duration_seconds=1800.0,
        )
    )
    vectors = [FeatureVector(f"a{i}", FeatureExtractor.FEATURE_NAMES, tuple(0.0 for _ in FeatureExtractor.FEATURE_NAMES), {}) for i in range(4)] + [
        FeatureVector(f"b{i}", FeatureExtractor.FEATURE_NAMES, tuple(1.0 for _ in FeatureExtractor.FEATURE_NAMES), {}) for i in range(4)
    ]
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, ["A"] * 4 + ["B"] * 4)
    engine = ContextInferenceEngine(clf)
    integrator = SessionContextIntegrator(repository, FeatureExtractor(), engine)

    result = integrator.predict_for_session("sess-1")
    assert result.predicted_label in {"A", "B"}


# ============================================================================
# 33-34. PredictionResult passed explicitly / persistence is explicit
# ============================================================================


def test_prediction_result_fields_can_be_passed_explicitly_into_persistence(tmp_path: Path) -> None:
    from app.ml.context_classifier import PredictionResult

    _, db_path = _repo_with_session(tmp_path)
    store = ContextObservationStore(db_path)
    prediction = PredictionResult(session_id="sess-1", predicted_label="Coding", class_probabilities={"Coding": 0.9, "Browsing": 0.1})

    recorded = store.record_observation(
        prediction.session_id, prediction.predicted_label, prediction.class_probabilities,
        _sample_model_version(), observed_at=_ts(),
    )

    assert recorded.session_id == prediction.session_id
    assert recorded.predicted_label == prediction.predicted_label
    assert recorded.class_probabilities == prediction.class_probabilities


def test_persistence_never_happens_automatically_from_inference_alone(tmp_path: Path) -> None:
    """Merely running inference must create zero rows unless record_observation is explicitly called."""
    from app.ml.context_classifier import ContextClassifier
    from app.ml.context_inference import ContextInferenceEngine
    from app.ml.feature_engineering import FeatureExtractor, FeatureVector
    from app.ml.session_context_integration import SessionContextIntegrator

    repository, db_path = _repo_with_session(tmp_path)
    repository.insert_activity(
        ActivitySegment(
            session_id="sess-1", started_at=_ts(), ended_at=_ts(1800),
            application="VSCode", process_name="Code.exe", window_title="x", duration_seconds=1800.0,
        )
    )
    vectors = [FeatureVector(f"a{i}", FeatureExtractor.FEATURE_NAMES, tuple(0.0 for _ in FeatureExtractor.FEATURE_NAMES), {}) for i in range(4)] + [
        FeatureVector(f"b{i}", FeatureExtractor.FEATURE_NAMES, tuple(1.0 for _ in FeatureExtractor.FEATURE_NAMES), {}) for i in range(4)
    ]
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, ["A"] * 4 + ["B"] * 4)
    integrator = SessionContextIntegrator(repository, FeatureExtractor(), ContextInferenceEngine(clf))

    integrator.predict_for_session("sess-1")  # inference only, no persistence call

    store = ContextObservationStore(db_path)
    assert store.list_observations() == []  # confirmed: nothing was auto-persisted


# ============================================================================
# 35. Architectural guard tests
# ============================================================================

import ast
import inspect


def _store_source() -> str:
    import app.ml.context_observation_store as module

    return inspect.getsource(module)


def _store_imported_names() -> set[str]:
    tree = ast.parse(_store_source())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            for alias in node.names:
                imported.add(f"{module_name}.{alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
    return imported


def test_guard_no_randomforestclassifier_construction() -> None:
    imported = _store_imported_names()
    assert not any(name.rsplit(".", 1)[-1] == "RandomForestClassifier" for name in imported)
    assert not any("sklearn" in name for name in imported)


def test_guard_no_context_classifier_dependency() -> None:
    imported = _store_imported_names()
    assert not any("context_classifier" in name for name in imported)


def test_guard_no_training_dependency() -> None:
    imported = _store_imported_names()
    assert not any("training_examples" in name for name in imported)
    assert not any("training_pipeline" in name for name in imported)


def test_guard_no_feature_extractor_dependency() -> None:
    imported = _store_imported_names()
    assert not any("feature_engineering" in name for name in imported)
    assert not any(name.rsplit(".", 1)[-1] == "FeatureExtractor" for name in imported)


def test_guard_no_activity_repository_dependency() -> None:
    imported = _store_imported_names()
    assert not any("activity_repository" in name for name in imported)


def test_guard_no_ui_dependency() -> None:
    imported = _store_imported_names()
    assert not any("app.ui" in name for name in imported)
    assert not any("PySide6" in name for name in imported)


def test_guard_no_background_monitoring_dependency() -> None:
    imported = _store_imported_names()
    assert not any("monitoring_service" in name for name in imported)
    assert not any("session_manager" in name for name in imported)
    assert not any("activity_monitor" in name for name in imported)


def test_guard_no_work_thread_or_task_concepts() -> None:
    source_lower = _store_source().lower()
    assert "workthread" not in source_lower
    assert "work_thread" not in source_lower
    assert "class task" not in source_lower


def test_guard_no_automatic_persistence_inside_context_inference_engine() -> None:
    """ContextInferenceEngine's source must not have been changed to write to a store."""
    import inspect as _inspect

    from app.ml.context_inference import ContextInferenceEngine

    predict_source = _inspect.getsource(ContextInferenceEngine.predict)
    predict_many_source = _inspect.getsource(ContextInferenceEngine.predict_many)

    for body in (predict_source, predict_many_source):
        assert "ContextObservationStore" not in body
        assert "record_observation" not in body


def test_guard_context_observation_is_the_only_locally_defined_dataclass() -> None:
    import app.ml.context_observation_store as module

    locally_defined = [
        name
        for name, obj in vars(module).items()
        if inspect.isclass(obj) and hasattr(obj, "__dataclass_fields__") and obj.__module__ == "app.ml.context_observation_store"
    ]
    assert locally_defined == ["ContextObservation"]


def test_guard_no_global_context_entity_concept() -> None:
    """Every observation stands alone -- no separate 'Context' table/entity with its own id space."""
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp())
    db_path = tmp_dir / "activity.db"
    ActivityRepository(db_path)
    ContextObservationStore(db_path)

    connection = sqlite3.connect(db_path)
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    connection.close()

    assert "contexts" not in tables
    assert "context" not in tables
    assert "work_threads" not in tables
    assert "tasks" not in tables


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
