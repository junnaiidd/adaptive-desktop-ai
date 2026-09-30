"""
Tests for Milestone A's `ContextObservationCoordinator`
(`app/ui/context_observation_coordinator.py`).

No PySide6 import anywhere in this file -- the coordinator is
Qt-independent by design, tested the same way as `DashboardController`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.database.activity_repository import ActivityRepository, ActivitySegment, StoredSession
from app.ml.context_classifier import ContextClassifier
from app.ml.context_observation_store import ContextObservationStore
from app.ml.feature_engineering import FeatureExtractor, FeatureVector
from app.ui.context_observation_coordinator import (
    ContextObservationCoordinator,
    CoordinatorRunResult,
    SessionObservationOutcome,
)

# ============================================================================
# Test helpers
# ============================================================================


def _ts(seconds: int = 0, hour: int = 9, day: int = 1) -> datetime:
    return datetime(2026, 1, day, hour=hour, tzinfo=timezone.utc) + timedelta(seconds=seconds)


def _repository_with_completed_session(
    tmp_path: Path, session_id: str = "sess-1", start_seconds: int = 0, db_name: str = "activity.db"
) -> tuple[ActivityRepository, Path]:
    db_path = tmp_path / db_name
    repository = ActivityRepository(db_path)
    started = _ts(start_seconds)
    ended = started + timedelta(seconds=1800)
    repository.start_session(StoredSession(session_id, started, None))
    repository.insert_activity(
        ActivitySegment(
            session_id=session_id, started_at=started, ended_at=ended,
            application="VSCode", process_name="Code.exe", window_title="x", duration_seconds=1800.0,
        )
    )
    repository.end_session(StoredSession(session_id, started, ended))
    return repository, db_path


def _train_and_save_model(tmp_path: Path, filename: str = "model.joblib") -> Path:
    names = FeatureExtractor.FEATURE_NAMES
    vectors = [FeatureVector(f"a{i}", names, tuple(0.0 for _ in names), {}) for i in range(4)] + [
        FeatureVector(f"b{i}", names, tuple(1.0 for _ in names), {}) for i in range(4)
    ]
    classifier = ContextClassifier(random_state=42)
    classifier.fit(vectors, ["Focused Work"] * 4 + ["Browsing"] * 4)
    model_path = tmp_path / filename
    classifier.save(model_path)
    return model_path


# ============================================================================
# Core happy path
# ============================================================================


def test_completed_session_with_no_prior_observation_gets_one_recorded(tmp_path: Path) -> None:
    repository, db_path = _repository_with_completed_session(tmp_path)
    store = ContextObservationStore(db_path)
    model_path = _train_and_save_model(tmp_path)
    coordinator = ContextObservationCoordinator(repository, store, model_path=model_path)

    result = coordinator.run_once()

    assert isinstance(result, CoordinatorRunResult)
    assert result.model_available is True
    assert result.recorded_count == 1
    assert result.failed_count == 0
    stored = store.list_observations_for_session("sess-1")
    assert len(stored) == 1
    assert stored[0].predicted_label in ("Focused Work", "Browsing")


def test_already_observed_session_is_skipped_not_duplicated(tmp_path: Path) -> None:
    repository, db_path = _repository_with_completed_session(tmp_path)
    store = ContextObservationStore(db_path)
    model_path = _train_and_save_model(tmp_path)
    coordinator = ContextObservationCoordinator(repository, store, model_path=model_path)

    first_result = coordinator.run_once()
    second_result = coordinator.run_once()

    assert first_result.recorded_count == 1
    assert second_result.recorded_count == 0
    assert len(second_result.outcomes) == 0  # skipped entirely, not attempted-and-failed
    assert len(store.list_observations_for_session("sess-1")) == 1  # never duplicated


def test_incomplete_session_is_never_touched(tmp_path: Path) -> None:
    db_path = tmp_path / "activity.db"
    repository = ActivityRepository(db_path)
    repository.start_session(StoredSession("active-session", _ts(), None))  # never ended
    store = ContextObservationStore(db_path)
    model_path = _train_and_save_model(tmp_path)
    coordinator = ContextObservationCoordinator(repository, store, model_path=model_path)

    result = coordinator.run_once()

    assert result.recorded_count == 0
    assert len(result.outcomes) == 0
    assert store.list_observations() == []


def test_multiple_completed_sessions_each_get_exactly_one_observation(tmp_path: Path) -> None:
    db_path = tmp_path / "activity.db"
    repository = ActivityRepository(db_path)
    for i in range(3):
        started = _ts(hour=9 + i)
        ended = started + timedelta(seconds=1800)
        repository.start_session(StoredSession(f"sess-{i}", started, None))
        repository.insert_activity(
            ActivitySegment(
                session_id=f"sess-{i}", started_at=started, ended_at=ended,
                application="VSCode", process_name="Code.exe", window_title="x", duration_seconds=1800.0,
            )
        )
        repository.end_session(StoredSession(f"sess-{i}", started, ended))
    store = ContextObservationStore(db_path)
    model_path = _train_and_save_model(tmp_path)
    coordinator = ContextObservationCoordinator(repository, store, model_path=model_path)

    result = coordinator.run_once()

    assert result.recorded_count == 3
    for i in range(3):
        assert len(store.list_observations_for_session(f"sess-{i}")) == 1


# ============================================================================
# Missing model
# ============================================================================


def test_no_model_artifact_returns_cleanly_with_no_observations(tmp_path: Path) -> None:
    repository, db_path = _repository_with_completed_session(tmp_path)
    store = ContextObservationStore(db_path)
    coordinator = ContextObservationCoordinator(repository, store, model_path=tmp_path / "does-not-exist.joblib")

    result = coordinator.run_once()

    assert result.model_available is False
    assert result.recorded_count == 0
    assert store.list_observations() == []


def test_missing_model_load_is_not_retried_on_every_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository, db_path = _repository_with_completed_session(tmp_path)
    store = ContextObservationStore(db_path)
    coordinator = ContextObservationCoordinator(repository, store, model_path=tmp_path / "does-not-exist.joblib")

    from app.ml.context_inference import ContextInferenceEngine

    calls: list[int] = []
    original_load = ContextInferenceEngine.load

    def spy_load(path=None):
        calls.append(1)
        return original_load(path)

    monkeypatch.setattr(ContextInferenceEngine, "load", staticmethod(spy_load))

    coordinator.run_once()
    coordinator.run_once()
    coordinator.run_once()

    assert len(calls) == 1  # attempted exactly once, never retried


# ============================================================================
# Per-session failure isolation
# ============================================================================


def test_one_sessions_failure_does_not_block_another_valid_session(tmp_path: Path) -> None:
    db_path = tmp_path / "activity.db"
    repository = ActivityRepository(db_path)
    # "bad" session: completed but with zero activities -> SessionHasNoActivitiesError
    started_bad = _ts(hour=9)
    repository.start_session(StoredSession("bad-session", started_bad, None))
    repository.end_session(StoredSession("bad-session", started_bad, started_bad + timedelta(seconds=60)))
    # "good" session: normal, complete, with an activity
    repository_good, _ = _repository_with_completed_session(tmp_path, session_id="good-session", start_seconds=5000, db_name="activity.db")

    store = ContextObservationStore(db_path)
    model_path = _train_and_save_model(tmp_path)
    coordinator = ContextObservationCoordinator(repository, store, model_path=model_path)

    result = coordinator.run_once()

    outcomes_by_session = {outcome.session_id: outcome for outcome in result.outcomes}
    assert outcomes_by_session["bad-session"].recorded is False
    assert outcomes_by_session["bad-session"].error is not None
    assert outcomes_by_session["good-session"].recorded is True
    assert len(store.list_observations_for_session("good-session")) == 1
    assert len(store.list_observations_for_session("bad-session")) == 0


def test_run_once_never_raises_for_a_per_session_failure(tmp_path: Path) -> None:
    db_path = tmp_path / "activity.db"
    repository = ActivityRepository(db_path)
    started = _ts()
    repository.start_session(StoredSession("no-activities", started, None))
    repository.end_session(StoredSession("no-activities", started, started + timedelta(seconds=60)))
    store = ContextObservationStore(db_path)
    model_path = _train_and_save_model(tmp_path)
    coordinator = ContextObservationCoordinator(repository, store, model_path=model_path)

    result = coordinator.run_once()  # must not raise

    assert isinstance(result, CoordinatorRunResult)
    assert result.recorded_count == 0
    assert result.failed_count == 1


# ============================================================================
# Result shape
# ============================================================================


def test_session_observation_outcome_fields() -> None:
    outcome = SessionObservationOutcome(session_id="s1", recorded=True)
    assert outcome.session_id == "s1"
    assert outcome.recorded is True
    assert outcome.error is None


def test_coordinator_run_result_counts_are_consistent(tmp_path: Path) -> None:
    repository, db_path = _repository_with_completed_session(tmp_path)
    store = ContextObservationStore(db_path)
    model_path = _train_and_save_model(tmp_path)
    coordinator = ContextObservationCoordinator(repository, store, model_path=model_path)

    result = coordinator.run_once()

    assert result.recorded_count + result.failed_count == len(result.outcomes)


# ============================================================================
# Architectural regression guards
# ============================================================================

import ast
import inspect


def _coordinator_source() -> str:
    import app.ui.context_observation_coordinator as module

    return inspect.getsource(module)


def _coordinator_imported_names() -> set[str]:
    tree = ast.parse(_coordinator_source())
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


def test_guard_no_qt_dependency() -> None:
    imported = _coordinator_imported_names()
    assert not any("PySide6" in name for name in imported)
    assert not any(name.rsplit(".", 1)[-1] in ("QThread", "QTimer", "Signal") for name in imported)


def test_guard_no_direct_randomforestclassifier_construction() -> None:
    imported = _coordinator_imported_names()
    assert not any(name.rsplit(".", 1)[-1] == "RandomForestClassifier" for name in imported)
    assert not any("sklearn" in name for name in imported)


def test_guard_does_not_import_phase_2b_2c_2e() -> None:
    imported = _coordinator_imported_names()
    assert not any("context_clustering" in name for name in imported)
    assert not any("context_labeling" in name for name in imported)
    assert not any("training_examples" in name for name in imported)
    assert not any("training_pipeline" in name for name in imported)


def test_guard_does_not_import_session_manager_or_monitoring_service() -> None:
    """SessionManager/MonitoringService remain ML-agnostic; the coordinator only reads via ActivityRepository."""
    imported = _coordinator_imported_names()
    assert not any("session_manager" in name for name in imported)
    assert not any("monitoring_service" in name for name in imported)
    assert not any("activity_monitor" in name for name in imported)


def test_guard_does_not_write_sessions_or_activity_segments(tmp_path: Path) -> None:
    """The coordinator must only read from ActivityRepository, never write."""
    coordinator_body = inspect.getsource(ContextObservationCoordinator)
    for forbidden in (".start_session(", ".end_session(", ".insert_activity("):
        assert forbidden not in coordinator_body


def test_guard_no_work_thread_or_task_concepts() -> None:
    lowered = _coordinator_source().lower()
    assert "workthread" not in lowered
    assert "work_thread" not in lowered
    assert "class task" not in lowered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
