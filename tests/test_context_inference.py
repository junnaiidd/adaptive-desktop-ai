"""
Comprehensive tests for Phase 2F: Context Inference
(`app/ml/context_inference.py`).

Tests cover:
A. Engine construction
B. Prediction
C. Feature validation (delegated to Phase 2D, not duplicated)
D. Model loading
E. Save/load consistency
F. No-training behavior
G. No leakage
H. Existing PredictionResult reuse
I. Integration (real 2A/2D APIs)
+ Adversarial / regression-guard tests
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ml.context_classifier import ContextClassifier, PredictionResult
from app.ml.context_inference import ContextInferenceEngine
from app.ml.feature_engineering import FeatureVector

# ============================================================================
# Test helpers
# ============================================================================

FEATURE_NAMES = ("f1", "f2")


def _vector(session_id: str, values: tuple[float, ...], names: tuple[str, ...] = FEATURE_NAMES) -> FeatureVector:
    return FeatureVector(session_id=session_id, feature_names=names, feature_values=values, metadata={})


def _two_class_dataset(n_per_class: int = 6) -> tuple[list[FeatureVector], list[str]]:
    vectors, labels = [], []
    for i in range(n_per_class):
        vectors.append(_vector(f"low-{i}", (0.0 + i * 0.01, 0.0 + i * 0.01)))
        labels.append("Focused Work")
    for i in range(n_per_class):
        vectors.append(_vector(f"high-{i}", (50.0 + i * 0.01, 50.0 + i * 0.01)))
        labels.append("Browsing")
    return vectors, labels


def _trained_classifier(random_state: int = 42) -> ContextClassifier:
    vectors, labels = _two_class_dataset()
    clf = ContextClassifier(random_state=random_state)
    clf.fit(vectors, labels)
    return clf


# ============================================================================
# A. Engine construction
# ============================================================================


def test_construction_with_trained_classifier_succeeds() -> None:
    clf = _trained_classifier()

    engine = ContextInferenceEngine(clf)

    assert engine.classifier is clf
    assert engine.is_trained is True


def test_construction_with_untrained_classifier_succeeds_but_reports_untrained() -> None:
    """
    Construction itself does not eagerly re-validate trained state (that
    would duplicate ContextClassifier's own _require_trained check) --
    but the wrapped engine correctly reports it via is_trained, and
    predicting on it fails through the classifier's own error (see
    test_predict_on_untrained_classifier_raises_same_runtime_error).
    """
    untrained = ContextClassifier(random_state=1)

    engine = ContextInferenceEngine(untrained)

    assert engine.is_trained is False


def test_construction_with_invalid_non_classifier_object_raises_type_error() -> None:
    with pytest.raises(TypeError, match="ContextClassifier instance"):
        ContextInferenceEngine("not a classifier")  # type: ignore[arg-type]


def test_construction_with_none_raises_type_error() -> None:
    with pytest.raises(TypeError, match="ContextClassifier instance"):
        ContextInferenceEngine(None)  # type: ignore[arg-type]


# ============================================================================
# B. Prediction
# ============================================================================


def test_predict_valid_feature_vector_returns_prediction_result() -> None:
    engine = ContextInferenceEngine(_trained_classifier())

    result = engine.predict(_vector("session-123", (0.02, 0.02)))

    assert isinstance(result, PredictionResult)


def test_predict_result_preserves_session_id() -> None:
    engine = ContextInferenceEngine(_trained_classifier())

    result = engine.predict(_vector("my-unique-session-id", (0.02, 0.02)))

    assert result.session_id == "my-unique-session-id"


def test_predict_result_predicted_label_is_a_trained_class() -> None:
    clf = _trained_classifier()
    engine = ContextInferenceEngine(clf)

    result = engine.predict(_vector("s", (0.02, 0.02)))

    assert result.predicted_label in clf.classes


def test_predict_result_class_probabilities_preserved_and_valid() -> None:
    clf = _trained_classifier()
    engine = ContextInferenceEngine(clf)

    result = engine.predict(_vector("s", (0.02, 0.02)))

    assert set(result.class_probabilities.keys()) == set(clf.classes)
    assert all(0.0 <= p <= 1.0 for p in result.class_probabilities.values())
    assert sum(result.class_probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert result.class_probabilities[result.predicted_label] == max(result.class_probabilities.values())


def test_predict_is_deterministic() -> None:
    engine = ContextInferenceEngine(_trained_classifier())
    vector = _vector("s", (0.02, 0.02))

    result_a = engine.predict(vector)
    result_b = engine.predict(vector)

    assert result_a == result_b


def test_predict_many_returns_one_result_per_input_in_order() -> None:
    engine = ContextInferenceEngine(_trained_classifier())
    vectors = [_vector(f"q{i}", (0.0 + i * 0.01, 0.0)) for i in range(3)]

    results = engine.predict_many(vectors)

    assert len(results) == 3
    assert [r.session_id for r in results] == ["q0", "q1", "q2"]


def test_predict_many_empty_input_returns_empty_tuple() -> None:
    engine = ContextInferenceEngine(_trained_classifier())

    assert engine.predict_many([]) == ()


# ============================================================================
# C. Feature validation (delegated, not duplicated)
# ============================================================================


def test_predict_with_wrong_feature_names_raises_value_error() -> None:
    engine = ContextInferenceEngine(_trained_classifier())
    wrong_vector = _vector("s", (1.0, 2.0), names=("other1", "other2"))

    with pytest.raises(ValueError, match="Feature mismatch"):
        engine.predict(wrong_vector)


def test_predict_with_reordered_feature_names_raises_value_error() -> None:
    engine = ContextInferenceEngine(_trained_classifier())
    reordered_vector = _vector("s", (1.0, 2.0), names=("f2", "f1"))

    with pytest.raises(ValueError, match="Feature mismatch"):
        engine.predict(reordered_vector)


def test_predict_with_wrong_feature_count_raises_value_error() -> None:
    engine = ContextInferenceEngine(_trained_classifier())
    wrong_count_vector = _vector("s", (1.0, 2.0, 3.0), names=("f1", "f2", "f3"))

    with pytest.raises(ValueError, match="Feature mismatch"):
        engine.predict(wrong_count_vector)


def test_predict_many_with_one_incompatible_vector_raises() -> None:
    engine = ContextInferenceEngine(_trained_classifier())
    vectors = [_vector("good", (0.02, 0.02)), _vector("bad", (1.0,), names=("only_one",))]

    with pytest.raises(ValueError, match="Feature mismatch"):
        engine.predict_many(vectors)


def test_feature_validation_error_message_matches_classifier_exactly() -> None:
    """The engine must not wrap/alter the classifier's own error text."""
    clf = _trained_classifier()
    engine = ContextInferenceEngine(clf)
    wrong_vector = _vector("s", (1.0, 2.0), names=("other1", "other2"))

    with pytest.raises(ValueError) as engine_error:
        engine.predict(wrong_vector)
    with pytest.raises(ValueError) as direct_error:
        clf.predict(wrong_vector)

    assert str(engine_error.value) == str(direct_error.value)


def test_predict_on_untrained_classifier_raises_same_runtime_error() -> None:
    untrained = ContextClassifier(random_state=1)
    engine = ContextInferenceEngine(untrained)

    with pytest.raises(RuntimeError) as engine_error:
        engine.predict(_vector("s", (1.0, 2.0)))
    with pytest.raises(RuntimeError) as direct_error:
        untrained.predict(_vector("s", (1.0, 2.0)))

    assert str(engine_error.value) == str(direct_error.value)


# ============================================================================
# D. Model loading
# ============================================================================


def test_load_real_saved_classifier_and_predict(tmp_path: Path) -> None:
    clf = _trained_classifier()
    target = tmp_path / "model.joblib"
    clf.save(target)

    engine = ContextInferenceEngine.load(target)
    result = engine.predict(_vector("s", (0.02, 0.02)))

    assert isinstance(result, PredictionResult)
    assert engine.is_trained is True


def test_load_missing_model_raises_file_not_found_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ContextInferenceEngine.load(tmp_path / "does-not-exist.joblib")


def test_load_invalid_model_artifact_raises(tmp_path: Path) -> None:
    """A file that exists but is not a valid joblib payload must fail clearly, not silently."""
    bad_path = tmp_path / "corrupt.joblib"
    bad_path.write_bytes(b"this is not a valid joblib artifact")

    with pytest.raises(Exception):  # exact exception type is joblib's/pickle's, not engineered by us
        ContextInferenceEngine.load(bad_path)


def test_load_default_path_matches_context_classifier_default() -> None:
    assert ContextInferenceEngine.load.__func__ is not None  # sanity: classmethod exists
    # Resolve without actually requiring a file to exist at the default location.
    assert ContextClassifier.DEFAULT_MODEL_PATH.name == "context_classifier.joblib"


# ============================================================================
# E. Save/load consistency
# ============================================================================


def test_train_save_load_through_2f_predict_consistency(tmp_path: Path) -> None:
    vectors, labels = _two_class_dataset()
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, labels)

    direct_predictions = [clf.predict(v) for v in vectors]

    target = tmp_path / "model.joblib"
    clf.save(target)
    engine = ContextInferenceEngine.load(target)
    engine_predictions = engine.predict_many(vectors)

    assert [p.predicted_label for p in direct_predictions] == [p.predicted_label for p in engine_predictions]
    for direct, via_engine in zip(direct_predictions, engine_predictions):
        assert direct.class_probabilities == via_engine.class_probabilities


# ============================================================================
# F. No-training behavior
# ============================================================================


def test_engine_has_no_fit_or_train_method() -> None:
    """The engine must not expose any training/retraining capability at all."""
    for forbidden in ("fit", "train", "retrain", "evaluate"):
        assert not hasattr(ContextInferenceEngine, forbidden), f"engine unexpectedly exposes {forbidden}()"


def test_prediction_does_not_modify_classifier_state() -> None:
    clf = _trained_classifier()
    engine = ContextInferenceEngine(clf)
    classes_before = clf.classes
    feature_names_before = clf.feature_names

    engine.predict(_vector("s", (0.02, 0.02)))
    engine.predict_many([_vector(f"q{i}", (0.0 + i * 0.01, 0.0)) for i in range(3)])

    assert clf.classes == classes_before
    assert clf.feature_names == feature_names_before


def test_prediction_does_not_modify_saved_model_file(tmp_path: Path) -> None:
    clf = _trained_classifier()
    target = tmp_path / "model.joblib"
    clf.save(target)
    bytes_before = target.read_bytes()

    engine = ContextInferenceEngine.load(target)
    engine.predict(_vector("s", (0.02, 0.02)))
    engine.predict_many([_vector("q", (0.01, 0.01))])

    assert target.read_bytes() == bytes_before


# ============================================================================
# G. No leakage
# ============================================================================


def test_metadata_and_session_id_do_not_affect_prediction() -> None:
    """
    Two vectors with identical feature_values but different session_id
    and metadata must produce identical predictions and probabilities.
    """
    engine = ContextInferenceEngine(_trained_classifier())
    same_features = (0.02, 0.02)

    vector_a = FeatureVector(
        session_id="totally-different-A",
        feature_names=FEATURE_NAMES,
        feature_values=same_features,
        metadata={"cluster_label": 999, "run_id": "fake-run", "label": "LeakedLabel"},
    )
    vector_b = FeatureVector(
        session_id="unrelated-B",
        feature_names=FEATURE_NAMES,
        feature_values=same_features,
        metadata={},
    )

    result_a = engine.predict(vector_a)
    result_b = engine.predict(vector_b)

    assert result_a.predicted_label == result_b.predicted_label
    assert result_a.class_probabilities == result_b.class_probabilities
    assert result_a.session_id == "totally-different-A"  # still carried through for attribution only
    assert result_b.session_id == "unrelated-B"


# ============================================================================
# H. Existing PredictionResult reuse
# ============================================================================


def test_predict_returns_actual_phase_2d_prediction_result_type() -> None:
    engine = ContextInferenceEngine(_trained_classifier())

    result = engine.predict(_vector("s", (0.02, 0.02)))

    assert type(result) is PredictionResult
    assert result.__class__.__module__ == "app.ml.context_classifier"


def test_predict_many_returns_tuple_of_actual_phase_2d_prediction_results() -> None:
    engine = ContextInferenceEngine(_trained_classifier())

    results = engine.predict_many([_vector("s", (0.02, 0.02))])

    assert all(type(r) is PredictionResult for r in results)


# ============================================================================
# I. Integration (real project APIs, minimal mocking)
# ============================================================================


class _MockActivity:
    def __init__(self, started_at, ended_at, duration_seconds, application, process_name):
        self.started_at = started_at
        self.ended_at = ended_at
        self.duration_seconds = duration_seconds
        self.application = application
        self.process_name = process_name


def test_end_to_end_real_feature_extractor_to_inference(tmp_path: Path) -> None:
    """Real Phase 2A FeatureExtractor -> real Phase 2D fit/save -> Phase 2F load/predict."""
    from datetime import datetime, timedelta, timezone

    from app.ml.feature_engineering import FeatureExtractor

    extractor = FeatureExtractor()

    coding_vectors = []
    for day in range(4):
        start = datetime(2026, 1, 1 + day, hour=9, tzinfo=timezone.utc)
        activities = [_MockActivity(start, start + timedelta(seconds=1800), 1800.0, "VSCode", "Code.exe")]
        coding_vectors.append(
            extractor.extract_features(
                session_id=f"coding-{day}",
                session_started_at=start,
                session_ended_at=start + timedelta(seconds=1800),
                activities=activities,
            )
        )

    browsing_vectors = []
    for day in range(4):
        start = datetime(2026, 1, 1 + day, hour=20, tzinfo=timezone.utc)
        activities = [
            _MockActivity(
                start + timedelta(seconds=i * 100),
                start + timedelta(seconds=(i + 1) * 100),
                100.0,
                f"App{i % 3}",
                f"app{i % 3}.exe",
            )
            for i in range(6)
        ]
        browsing_vectors.append(
            extractor.extract_features(
                session_id=f"browsing-{day}",
                session_started_at=start,
                session_ended_at=start + timedelta(seconds=600),
                activities=activities,
            )
        )

    all_vectors = coding_vectors + browsing_vectors
    labels = ["Focused Coding"] * len(coding_vectors) + ["Evening Browsing"] * len(browsing_vectors)

    clf = ContextClassifier(random_state=42)
    clf.fit(all_vectors, labels)
    target = tmp_path / "model.joblib"
    clf.save(target)

    engine = ContextInferenceEngine.load(target)
    prediction = engine.predict(coding_vectors[0])

    assert prediction.predicted_label in {"Focused Coding", "Evening Browsing"}
    assert prediction.session_id == "coding-0"
    assert set(prediction.class_probabilities.keys()) == {"Focused Coding", "Evening Browsing"}


# ============================================================================
# ADVERSARIAL / REGRESSION GUARDS
#
# These do not test "normal" behavior -- they exist to FAIL if a future
# change reintroduces a forbidden pattern, even one that might otherwise
# look behaviorally fine. Several inspect the actual module source/imports
# rather than only behavior.
# ============================================================================

import ast
import inspect


def _inference_source() -> str:
    import app.ml.context_inference as module

    return inspect.getsource(module)


def _inference_imported_names() -> set[str]:
    """Parse actual import statements via AST, not raw text search, to avoid docstring false positives."""
    tree = ast.parse(_inference_source())
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


def test_guard_no_direct_randomforestclassifier_construction() -> None:
    """Fails if context_inference.py is ever changed to build its own model."""
    imported = _inference_imported_names()

    assert not any(name.rsplit(".", 1)[-1] == "RandomForestClassifier" for name in imported)
    assert not any("sklearn" in name for name in imported)


def _inference_method_body_source(class_name: str, method_name: str) -> str:
    """
    Extract just the executable body of one method (excluding its
    docstring) via AST, so prose in docstrings can never be mistaken for
    actual code -- the same false-positive class caught and fixed during
    Phase 2E's guard-test development.
    """
    tree = ast.parse(_inference_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    body_nodes = item.body
                    # Skip a leading docstring expression, if present.
                    if body_nodes and isinstance(body_nodes[0], ast.Expr) and isinstance(
                        getattr(body_nodes[0], "value", None), (ast.Constant,)
                    ):
                        body_nodes = body_nodes[1:]
                    return "\n".join(ast.unparse(n) for n in body_nodes)
    raise AssertionError(f"{class_name}.{method_name} not found in context_inference.py")


def test_guard_no_second_prediction_implementation() -> None:
    """
    Fails if predict/predict_many are ever reimplemented instead of
    delegating. Checks the actual executable body of each method (not
    docstrings/prose) for any direct use of the raw sklearn prediction
    surface, which would indicate the engine started talking to a model
    directly instead of exclusively through ContextClassifier.
    """
    predict_body = _inference_method_body_source("ContextInferenceEngine", "predict")
    predict_many_body = _inference_method_body_source("ContextInferenceEngine", "predict_many")

    for body in (predict_body, predict_many_body):
        assert "predict_proba" not in body
        assert ".classes_" not in body
        assert "np." not in body and "numpy" not in body  # no raw array math re-implemented here

    # And the bodies genuinely delegate to the wrapped classifier.
    assert "self.classifier.predict" in predict_body
    assert "self.classifier.predict_many" in predict_many_body


def test_guard_no_second_prediction_result_type() -> None:
    """Fails if a competing PredictionResult-like dataclass is introduced in this module."""
    import app.ml.context_inference as module

    locally_defined_dataclasses = [
        name
        for name, obj in vars(module).items()
        if inspect.isclass(obj)
        and hasattr(obj, "__dataclass_fields__")
        and obj.__module__ == "app.ml.context_inference"  # defined HERE, not merely imported
    ]

    assert locally_defined_dataclasses == [], (
        f"unexpected dataclass(es) defined in context_inference.py: {locally_defined_dataclasses}"
    )

    # The PredictionResult symbol available in this module is Phase 2D's actual type, re-exported/used as-is.
    assert module.PredictionResult is PredictionResult
    assert module.PredictionResult.__module__ == "app.ml.context_classifier"


def test_guard_no_training_or_retraining_capability() -> None:
    """Fails if fit/train/retrain/evaluate is ever added to the engine (duplicated in F above, kept for grouping)."""
    forbidden_methods = ("fit", "train", "retrain", "evaluate", "partial_fit")
    for name in forbidden_methods:
        assert not hasattr(ContextInferenceEngine, name)


def test_guard_does_not_import_phase_2b_clustering_or_2c_labeling() -> None:
    """Fails if a runtime dependency on context_clustering.py/context_labeling.py is ever introduced."""
    imported = _inference_imported_names()

    assert not any("context_clustering" in name for name in imported)
    assert not any("context_labeling" in name for name in imported)

    import app.ml.context_inference as module

    module_names = {
        obj.__module__
        for name, obj in vars(module).items()
        if not name.startswith("_") and hasattr(obj, "__module__")
    }
    assert "app.ml.context_clustering" not in module_names
    assert "app.ml.context_labeling" not in module_names


def test_guard_does_not_import_training_examples_or_training_pipeline() -> None:
    """Fails if inference is ever wired to touch the training corpus or retraining pipeline."""
    imported = _inference_imported_names()

    assert not any("training_examples" in name for name in imported)
    assert not any("training_pipeline" in name for name in imported)


def test_guard_no_prediction_persistence_in_source() -> None:
    """Fails if this module starts writing predictions to disk/DB/cache."""
    source = _inference_source()
    lowered = source.lower()

    for forbidden_token in ("sqlite", "predictions.json", "prediction_history", ".db", "insert into", "cursor("):
        assert forbidden_token not in lowered

    # No filesystem-write calls of any kind belong in an inference-only module.
    assert "write_text(" not in source
    assert "write_bytes(" not in source
    assert "open(" not in source


def test_guard_no_hashed_or_timestamped_model_filenames() -> None:
    """Fails if content-hashed or timestamped model filenames are ever introduced."""
    source = _inference_source()
    lowered = source.lower()

    assert "hashlib" not in lowered
    assert "sha256" not in lowered
    assert "strftime" not in lowered
    assert "utcnow().isoformat" not in source


def test_guard_default_model_path_is_exactly_the_2d_default() -> None:
    """Fails if a competing default path/filename scheme is introduced."""
    assert ContextClassifier.DEFAULT_MODEL_PATH.name == "context_classifier.joblib"
    assert ContextClassifier.DEFAULT_MODEL_PATH.parent.name == "models"

    source = _inference_source()
    assert "context_classifier_" not in source  # no hash/timestamp-suffixed filename pattern


def test_guard_engine_defines_no_alternative_model_path_constant() -> None:
    """Fails if the engine starts defining its own DEFAULT_MODEL_PATH instead of using 2D's."""
    import app.ml.context_inference as module

    assert not hasattr(ContextInferenceEngine, "DEFAULT_MODEL_PATH")
    assert not hasattr(module, "DEFAULT_MODEL_PATH")


def test_guard_no_silent_fallback_when_model_unavailable(tmp_path: Path) -> None:
    """Fails if a missing model ever produces a fabricated prediction instead of raising."""
    with pytest.raises(FileNotFoundError):
        ContextInferenceEngine.load(tmp_path / "nonexistent.joblib")


def test_guard_predict_does_not_swallow_exceptions() -> None:
    """
    Fails if predict()/predict_many() is ever wrapped in a broad
    try/except that could turn a real error into a fake fallback result.
    """
    source = _inference_source()

    assert "except Exception" not in source
    assert "except:" not in source
    assert "pass  # ignore" not in source


def test_guard_no_automatic_retraining_after_prediction() -> None:
    """Fails if predict() is ever changed to trigger fit()/retrain_from_examples() as a side effect."""
    clf = _trained_classifier()
    engine = ContextInferenceEngine(clf)
    model_object_before = clf._model  # the actual fitted sklearn estimator identity

    engine.predict(_vector("s", (0.02, 0.02)))
    engine.predict_many([_vector("q", (0.01, 0.01))])

    assert clf._model is model_object_before  # never replaced by a retrain


def test_guard_engine_exposes_no_speculative_prediction_aliases() -> None:
    """
    Fails if speculative alias methods (predict_context, classify_context,
    infer_context, run_inference, predict_session, ...) are ever added
    alongside predict/predict_many.
    """
    forbidden_aliases = (
        "predict_context", "classify_context", "infer_context",
        "run_inference", "predict_session", "classify", "infer",
    )
    for alias in forbidden_aliases:
        assert not hasattr(ContextInferenceEngine, alias), f"unexpected speculative alias: {alias}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
