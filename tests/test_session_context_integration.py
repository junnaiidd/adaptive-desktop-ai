"""
Comprehensive tests for Phase 2G: Session Context Integration
(`app/ml/session_context_integration.py`).

Tests cover:
- Completed session successfully produces a PredictionResult
- Real StoredSession/StoredActivity + real Phase 2A/2F components
- FeatureExtractor and ContextInferenceEngine are actually invoked
- Existing Phase 2D PredictionResult type is returned unchanged
- session_id preservation
- Feature values genuinely derive from the session's own activities
- Incomplete / missing / no-activity session rejection
- Untrained inference engine handling
- Feature-schema incompatibility propagation
- No duplicated classifier/model behavior
- No persistence, no training
- Adversarial / regression guards (Phase 2F style)
"""

from __future__ import annotations

import ast
import inspect
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.database.activity_repository import ActivityRepository, ActivitySegment, StoredSession
from app.ml.context_classifier import ContextClassifier, PredictionResult
from app.ml.context_inference import ContextInferenceEngine
from app.ml.feature_engineering import FeatureExtractor, FeatureVector
from app.ml.session_context_integration import (
    SessionContextIntegrator,
    SessionHasNoActivitiesError,
    SessionLookupError,
    SessionNotCompleteError,
    SessionNotFoundError,
)

# ============================================================================
# Test helpers -- all use the REAL repository, extractor, classifier, engine.
# ============================================================================


def _ts(seconds: int = 0, hour: int = 9, day: int = 1) -> datetime:
    return datetime(2026, 1, day, hour=hour, tzinfo=timezone.utc) + timedelta(seconds=seconds)


def _new_repository(tmp_path: Path) -> ActivityRepository:
    return ActivityRepository(tmp_path / "activity.db")


def _add_completed_session_with_activity(
    repository: ActivityRepository,
    session_id: str,
    application: str = "VSCode",
    start_seconds: int = 0,
    duration_seconds: float = 1800.0,
) -> None:
    started = _ts(start_seconds)
    ended = started + timedelta(seconds=duration_seconds)
    repository.start_session(StoredSession(session_id, started, None))
    repository.insert_activity(
        ActivitySegment(
            session_id=session_id,
            started_at=started,
            ended_at=ended,
            application=application,
            process_name=f"{application}.exe",
            window_title=application,
            duration_seconds=duration_seconds,
        )
    )
    repository.end_session(StoredSession(session_id, started, ended))


def _two_class_training_vectors() -> tuple[list[FeatureVector], list[str]]:
    """Small, well-separated synthetic training set spanning the real 14-feature schema."""
    names = FeatureExtractor.FEATURE_NAMES
    low_vectors = [
        FeatureVector(f"low-{i}", names, tuple(0.0 + i * 0.01 for _ in names), {}) for i in range(4)
    ]
    high_vectors = [
        FeatureVector(f"high-{i}", names, tuple(500.0 + i * 0.01 for _ in names), {}) for i in range(4)
    ]
    return low_vectors + high_vectors, ["Focused Work"] * 4 + ["Browsing"] * 4


def _trained_engine() -> ContextInferenceEngine:
    vectors, labels = _two_class_training_vectors()
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, labels)
    return ContextInferenceEngine(clf)


def _integrator(repository: ActivityRepository, engine: ContextInferenceEngine | None = None) -> SessionContextIntegrator:
    return SessionContextIntegrator(repository, FeatureExtractor(), engine or _trained_engine())


# ============================================================================
# Completed session -> PredictionResult (happy path, real components throughout)
# ============================================================================


def test_completed_session_produces_prediction_result(tmp_path: Path) -> None:
    repository = _new_repository(tmp_path)
    _add_completed_session_with_activity(repository, "sess-1")
    integrator = _integrator(repository)

    result = integrator.predict_for_session("sess-1")

    assert isinstance(result, PredictionResult)


def test_result_is_the_actual_phase_2d_prediction_result_type(tmp_path: Path) -> None:
    repository = _new_repository(tmp_path)
    _add_completed_session_with_activity(repository, "sess-1")
    integrator = _integrator(repository)

    result = integrator.predict_for_session("sess-1")

    assert type(result) is PredictionResult
    assert result.__class__.__module__ == "app.ml.context_classifier"


def test_session_id_is_preserved_in_the_result(tmp_path: Path) -> None:
    repository = _new_repository(tmp_path)
    _add_completed_session_with_activity(repository, "my-specific-session-id")
    integrator = _integrator(repository)

    result = integrator.predict_for_session("my-specific-session-id")

    assert result.session_id == "my-specific-session-id"


def test_class_probabilities_are_a_valid_distribution(tmp_path: Path) -> None:
    repository = _new_repository(tmp_path)
    _add_completed_session_with_activity(repository, "sess-1")
    integrator = _integrator(repository)

    result = integrator.predict_for_session("sess-1")

    assert sum(result.class_probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(0.0 <= p <= 1.0 for p in result.class_probabilities.values())


# ============================================================================
# FeatureExtractor and ContextInferenceEngine are actually invoked
# ============================================================================


def test_feature_extractor_is_actually_invoked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository = _new_repository(tmp_path)
    _add_completed_session_with_activity(repository, "sess-1")
    extractor = FeatureExtractor()
    integrator = SessionContextIntegrator(repository, extractor, _trained_engine())

    calls: list[str] = []
    original = extractor.extract_features

    def spy(*args, **kwargs):
        calls.append(kwargs.get("session_id") or (args[0] if args else None))
        return original(*args, **kwargs)

    monkeypatch.setattr(extractor, "extract_features", spy)
    integrator.predict_for_session("sess-1")

    assert calls == ["sess-1"]


def test_context_inference_engine_is_actually_invoked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository = _new_repository(tmp_path)
    _add_completed_session_with_activity(repository, "sess-1")
    engine = _trained_engine()
    integrator = SessionContextIntegrator(repository, FeatureExtractor(), engine)

    calls: list[FeatureVector] = []
    original = engine.predict

    def spy(feature_vector):
        calls.append(feature_vector)
        return original(feature_vector)

    monkeypatch.setattr(engine, "predict", spy)
    integrator.predict_for_session("sess-1")

    assert len(calls) == 1
    assert isinstance(calls[0], FeatureVector)
    assert calls[0].session_id == "sess-1"


# ============================================================================
# Feature values genuinely derive from the session's own activities
# ============================================================================


def test_feature_values_reflect_the_sessions_actual_activities(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two sessions with different real activity durations must produce different feature vectors."""
    repository = _new_repository(tmp_path)
    _add_completed_session_with_activity(repository, "short", duration_seconds=60.0)
    _add_completed_session_with_activity(repository, "long", start_seconds=10_000, duration_seconds=7200.0)

    extractor = FeatureExtractor()
    captured: dict[str, FeatureVector] = {}
    original = extractor.extract_features

    def spy(*args, **kwargs):
        vector = original(*args, **kwargs)
        captured[vector.session_id] = vector
        return vector

    monkeypatch.setattr(extractor, "extract_features", spy)
    integrator = SessionContextIntegrator(repository, extractor, _trained_engine())

    integrator.predict_for_session("short")
    integrator.predict_for_session("long")

    short_duration_feature = captured["short"].to_dict()["session_duration_minutes"]
    long_duration_feature = captured["long"].to_dict()["session_duration_minutes"]
    assert short_duration_feature == pytest.approx(1.0)
    assert long_duration_feature == pytest.approx(120.0)
    assert short_duration_feature != long_duration_feature


def test_activities_from_other_sessions_do_not_leak_into_feature_extraction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository = _new_repository(tmp_path)
    _add_completed_session_with_activity(repository, "sess-1", application="VSCode")
    _add_completed_session_with_activity(repository, "sess-2", start_seconds=5000, application="Chrome")

    extractor = FeatureExtractor()
    captured: dict[str, tuple] = {}
    original = extractor.extract_features

    def spy(*args, **kwargs):
        vector = original(*args, **kwargs)
        captured[vector.session_id] = vector.metadata["unique_apps"] if "unique_apps" in vector.metadata else None
        return vector

    monkeypatch.setattr(extractor, "extract_features", spy)
    integrator = SessionContextIntegrator(repository, extractor, _trained_engine())

    integrator.predict_for_session("sess-1")
    integrator.predict_for_session("sess-2")

    assert captured["sess-1"] == ["VSCode"]
    assert captured["sess-2"] == ["Chrome"]


# ============================================================================
# Incomplete / missing / no-activity session rejection
# ============================================================================


def test_incomplete_session_is_rejected(tmp_path: Path) -> None:
    repository = _new_repository(tmp_path)
    repository.start_session(StoredSession("active-session", _ts(), None))  # never ended
    integrator = _integrator(repository)

    with pytest.raises(SessionNotCompleteError, match="has not ended yet"):
        integrator.predict_for_session("active-session")


def test_missing_session_is_rejected(tmp_path: Path) -> None:
    repository = _new_repository(tmp_path)
    integrator = _integrator(repository)

    with pytest.raises(SessionNotFoundError, match="No session found"):
        integrator.predict_for_session("does-not-exist")


def test_no_activity_session_is_rejected_explicitly(tmp_path: Path) -> None:
    repository = _new_repository(tmp_path)
    repository.start_session(StoredSession("empty-session", _ts(), None))
    repository.end_session(StoredSession("empty-session", _ts(), _ts(60)))  # completed, zero activities
    integrator = _integrator(repository)

    with pytest.raises(SessionHasNoActivitiesError, match="no recorded activity"):
        integrator.predict_for_session("empty-session")


def test_all_three_session_errors_are_value_error_subclasses(tmp_path: Path) -> None:
    """Callers that only care about 'this input wasn't valid' can catch plain ValueError."""
    assert issubclass(SessionNotFoundError, SessionLookupError)
    assert issubclass(SessionNotCompleteError, SessionLookupError)
    assert issubclass(SessionHasNoActivitiesError, SessionLookupError)
    assert issubclass(SessionLookupError, ValueError)


def test_missing_session_error_can_be_caught_as_plain_value_error(tmp_path: Path) -> None:
    repository = _new_repository(tmp_path)
    integrator = _integrator(repository)

    with pytest.raises(ValueError):
        integrator.predict_for_session("does-not-exist")


# ============================================================================
# Untrained inference engine handling
# ============================================================================


def test_untrained_inference_engine_raises_runtime_error(tmp_path: Path) -> None:
    repository = _new_repository(tmp_path)
    _add_completed_session_with_activity(repository, "sess-1")
    untrained_engine = ContextInferenceEngine(ContextClassifier(random_state=1))
    integrator = SessionContextIntegrator(repository, FeatureExtractor(), untrained_engine)

    with pytest.raises(RuntimeError, match="not been trained"):
        integrator.predict_for_session("sess-1")


def test_untrained_engine_error_matches_phase_2f_error_exactly(tmp_path: Path) -> None:
    """The integrator must not wrap or alter Phase 2F's own error text."""
    repository = _new_repository(tmp_path)
    _add_completed_session_with_activity(repository, "sess-1")
    untrained_classifier = ContextClassifier(random_state=1)
    engine = ContextInferenceEngine(untrained_classifier)
    integrator = SessionContextIntegrator(repository, FeatureExtractor(), engine)

    with pytest.raises(RuntimeError) as integrator_error:
        integrator.predict_for_session("sess-1")
    with pytest.raises(RuntimeError) as direct_error:
        untrained_classifier.predict(FeatureVector("x", FeatureExtractor.FEATURE_NAMES, tuple(0.0 for _ in FeatureExtractor.FEATURE_NAMES), {}))

    assert str(integrator_error.value) == str(direct_error.value)


# ============================================================================
# Feature-schema incompatibility propagation
# ============================================================================


def test_feature_schema_incompatibility_propagates_as_value_error(tmp_path: Path) -> None:
    """
    A classifier trained on a DIFFERENT feature schema than the one
    FeatureExtractor actually produces must cause a ValueError to
    propagate, unmodified, from ContextInferenceEngine/ContextClassifier.
    """
    repository = _new_repository(tmp_path)
    _add_completed_session_with_activity(repository, "sess-1")

    wrong_schema_names = ("totally", "different", "feature", "schema")
    wrong_vectors = [
        FeatureVector("a", wrong_schema_names, (1.0, 2.0, 3.0, 4.0), {}),
        FeatureVector("b", wrong_schema_names, (5.0, 6.0, 7.0, 8.0), {}),
    ]
    mismatched_classifier = ContextClassifier(random_state=1)
    mismatched_classifier.fit(wrong_vectors, ["A", "B"])
    engine = ContextInferenceEngine(mismatched_classifier)
    integrator = SessionContextIntegrator(repository, FeatureExtractor(), engine)

    with pytest.raises(ValueError, match="Feature mismatch"):
        integrator.predict_for_session("sess-1")


# ============================================================================
# No duplicated classifier/model behavior, no persistence, no training
# ============================================================================


def test_prediction_does_not_train_the_classifier(tmp_path: Path) -> None:
    repository = _new_repository(tmp_path)
    _add_completed_session_with_activity(repository, "sess-1")
    vectors, labels = _two_class_training_vectors()
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, labels)
    classes_before = clf.classes
    model_object_before = clf._model
    engine = ContextInferenceEngine(clf)
    integrator = SessionContextIntegrator(repository, FeatureExtractor(), engine)

    integrator.predict_for_session("sess-1")

    assert clf.classes == classes_before
    assert clf._model is model_object_before  # never replaced/refit


def test_prediction_does_not_write_to_the_repository(tmp_path: Path) -> None:
    repository = _new_repository(tmp_path)
    _add_completed_session_with_activity(repository, "sess-1")
    sessions_before = repository.list_sessions()
    activities_before = repository.list_activities()
    integrator = _integrator(repository)

    integrator.predict_for_session("sess-1")

    assert repository.list_sessions() == sessions_before
    assert repository.list_activities() == activities_before


def test_prediction_does_not_create_any_new_files(tmp_path: Path) -> None:
    repository = _new_repository(tmp_path)
    _add_completed_session_with_activity(repository, "sess-1")
    integrator = _integrator(repository)

    files_before = set(tmp_path.rglob("*"))
    integrator.predict_for_session("sess-1")
    files_after = set(tmp_path.rglob("*"))

    assert files_after == files_before


def test_prediction_does_not_modify_saved_model_artifact(tmp_path: Path) -> None:
    repository = _new_repository(tmp_path)
    _add_completed_session_with_activity(repository, "sess-1")
    vectors, labels = _two_class_training_vectors()
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, labels)
    model_path = tmp_path / "model.joblib"
    clf.save(model_path)
    bytes_before = model_path.read_bytes()

    engine = ContextInferenceEngine.load(model_path)
    integrator = SessionContextIntegrator(repository, FeatureExtractor(), engine)
    integrator.predict_for_session("sess-1")

    assert model_path.read_bytes() == bytes_before


# ============================================================================
# Validation ordering: fail fast, don't touch later steps unnecessarily
# ============================================================================


def test_nonexistent_session_never_reaches_activity_lookup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository = _new_repository(tmp_path)
    integrator = _integrator(repository)
    calls: list[str] = []
    original = repository.list_activities_for_session

    def spy(session_id):
        calls.append(session_id)
        return original(session_id)

    monkeypatch.setattr(repository, "list_activities_for_session", spy)

    with pytest.raises(SessionNotFoundError):
        integrator.predict_for_session("does-not-exist")

    assert calls == []  # never even attempted once the session lookup failed


def test_incomplete_session_never_reaches_feature_extraction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository = _new_repository(tmp_path)
    repository.start_session(StoredSession("active", _ts(), None))
    extractor = FeatureExtractor()
    calls: list = []
    original = extractor.extract_features

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(extractor, "extract_features", spy)
    integrator = SessionContextIntegrator(repository, extractor, _trained_engine())

    with pytest.raises(SessionNotCompleteError):
        integrator.predict_for_session("active")

    assert calls == []


def test_no_activity_session_never_reaches_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository = _new_repository(tmp_path)
    repository.start_session(StoredSession("empty", _ts(), None))
    repository.end_session(StoredSession("empty", _ts(), _ts(60)))
    engine = _trained_engine()
    calls: list = []
    original = engine.predict

    def spy(feature_vector):
        calls.append(1)
        return original(feature_vector)

    monkeypatch.setattr(engine, "predict", spy)
    integrator = SessionContextIntegrator(repository, FeatureExtractor(), engine)

    with pytest.raises(SessionHasNoActivitiesError):
        integrator.predict_for_session("empty")

    assert calls == []


# ============================================================================
# Real Phase 1 repository objects + real Phase 2A/2F components, end to end
# (the mandatory "important architectural test": no mocks in the core path)
# ============================================================================


def test_full_real_pipeline_no_mocks(tmp_path: Path) -> None:
    """
    ActivityRepository -> StoredSession/StoredActivity -> FeatureExtractor
    -> FeatureVector -> ContextInferenceEngine -> PredictionResult, with
    every single component real: a real SQLite-backed repository, a real
    FeatureExtractor, a real fitted ContextClassifier wrapped in a real
    ContextInferenceEngine. Nothing in this test is mocked or patched.
    """
    repository = ActivityRepository(tmp_path / "activity.db")
    started = datetime(2026, 3, 5, hour=9, tzinfo=timezone.utc)
    ended = started + timedelta(seconds=2400)
    repository.start_session(StoredSession("real-session", started, None))
    repository.insert_activity(
        ActivitySegment(
            session_id="real-session", started_at=started, ended_at=started + timedelta(seconds=1200),
            application="VSCode", process_name="Code.exe", window_title="main.py", duration_seconds=1200.0,
        )
    )
    repository.insert_activity(
        ActivitySegment(
            session_id="real-session", started_at=started + timedelta(seconds=1200), ended_at=ended,
            application="Terminal", process_name="bash", window_title="terminal", duration_seconds=1200.0,
        )
    )
    repository.end_session(StoredSession("real-session", started, ended))

    real_extractor = FeatureExtractor()
    vectors, labels = _two_class_training_vectors()
    real_classifier = ContextClassifier(random_state=42)
    real_classifier.fit(vectors, labels)
    real_engine = ContextInferenceEngine(real_classifier)

    integrator = SessionContextIntegrator(repository, real_extractor, real_engine)
    result = integrator.predict_for_session("real-session")

    assert isinstance(result, PredictionResult)
    assert result.session_id == "real-session"
    assert result.predicted_label in real_classifier.classes
    assert set(result.class_probabilities.keys()) == set(real_classifier.classes)


# ============================================================================
# ADVERSARIAL / REGRESSION GUARDS (Phase 2F style)
# ============================================================================


def _integration_source() -> str:
    import app.ml.session_context_integration as module

    return inspect.getsource(module)


def _integration_imported_names() -> set[str]:
    """Parse actual import statements via AST, not raw text, to avoid docstring false positives."""
    tree = ast.parse(_integration_source())
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


def test_guard_no_direct_randomforestclassifier_or_sklearn_construction() -> None:
    imported = _integration_imported_names()
    assert not any(name.rsplit(".", 1)[-1] == "RandomForestClassifier" for name in imported)
    assert not any("sklearn" in name for name in imported)


def test_guard_no_kmeans_or_standardscaler_construction() -> None:
    imported = _integration_imported_names()
    assert not any(name.rsplit(".", 1)[-1] in ("KMeans", "StandardScaler") for name in imported)


def test_guard_does_not_import_phase_2b_clustering_or_2c_labeling() -> None:
    imported = _integration_imported_names()
    assert not any("context_clustering" in name for name in imported)
    assert not any("context_labeling" in name for name in imported)


def test_guard_does_not_import_training_examples_or_training_pipeline() -> None:
    imported = _integration_imported_names()
    assert not any("training_examples" in name for name in imported)
    assert not any("training_pipeline" in name for name in imported)


def test_guard_no_second_prediction_result_type() -> None:
    import app.ml.session_context_integration as module

    locally_defined_dataclasses = [
        name
        for name, obj in vars(module).items()
        if inspect.isclass(obj)
        and hasattr(obj, "__dataclass_fields__")
        and obj.__module__ == "app.ml.session_context_integration"
    ]

    assert locally_defined_dataclasses == []
    assert module.PredictionResult is PredictionResult
    assert module.PredictionResult.__module__ == "app.ml.context_classifier"


def test_guard_no_training_or_retraining_capability_exposed() -> None:
    forbidden_methods = ("fit", "train", "retrain", "evaluate", "partial_fit", "capture_labeled_examples")
    for name in forbidden_methods:
        assert not hasattr(SessionContextIntegrator, name)


def test_guard_no_prediction_persistence_in_source() -> None:
    source = _integration_source()
    lowered = source.lower()

    for forbidden_token in ("predictions.json", "prediction_history", "cursor(", "insert into", "create table"):
        assert forbidden_token not in lowered
    assert "write_text(" not in source
    assert "write_bytes(" not in source
    assert "joblib.dump" not in source


def test_guard_integrator_never_writes_to_the_repository_in_source() -> None:
    """
    The integrator must only call read methods on ActivityRepository
    (list_sessions/list_activities_for_session), never
    start_session/end_session/insert_activity.
    """
    predict_body = _method_body_source("SessionContextIntegrator", "predict_for_session")
    lookup_body = _method_body_source("SessionContextIntegrator", "_get_completed_session")
    combined = predict_body + lookup_body

    assert ".start_session(" not in combined
    assert ".end_session(" not in combined
    assert ".insert_activity(" not in combined


def _method_body_source(class_name: str, method_name: str) -> str:
    tree = ast.parse(_integration_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    body_nodes = item.body
                    if body_nodes and isinstance(body_nodes[0], ast.Expr) and isinstance(
                        getattr(body_nodes[0], "value", None), ast.Constant
                    ):
                        body_nodes = body_nodes[1:]
                    return "\n".join(ast.unparse(n) for n in body_nodes)
    raise AssertionError(f"{class_name}.{method_name} not found")


def test_guard_predict_for_session_delegates_to_the_real_components() -> None:
    """The method body must genuinely call the injected extractor/engine, not reimplement logic."""
    body = _method_body_source("SessionContextIntegrator", "predict_for_session")

    assert "self.feature_extractor.extract_features(" in body
    assert "self.inference_engine.predict(" in body
    assert "predict_proba" not in body
    assert ".classes_" not in body


def test_guard_no_feature_computation_reimplemented_in_source() -> None:
    """No local re-derivation of Phase 2A's entropy/transition/duration math."""
    source = _integration_source()
    lowered = source.lower()

    for forbidden_token in ("entropy", "shannon", "transition_count", "app_durations"):
        assert forbidden_token not in lowered


def test_guard_does_not_swallow_exceptions() -> None:
    source = _integration_source()
    assert "except Exception" not in source
    assert "except:" not in source


def test_guard_session_error_types_are_minimal_and_value_error_based() -> None:
    """Exactly the three documented precondition errors, all deriving from ValueError."""
    import app.ml.session_context_integration as module

    exception_classes = [
        name
        for name, obj in vars(module).items()
        if inspect.isclass(obj) and issubclass(obj, BaseException) and obj.__module__ == "app.ml.session_context_integration"
    ]

    assert set(exception_classes) == {
        "SessionLookupError",
        "SessionNotFoundError",
        "SessionNotCompleteError",
        "SessionHasNoActivitiesError",
    }
    assert issubclass(module.SessionLookupError, ValueError)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
