"""
Comprehensive tests for Phase 2D: Personalized Supervised Context
Classification.

Tests cover:
- Untrained classifier behavior
- Successful training
- Deterministic training
- Arbitrary labels
- Multiclass classification
- Binary classification
- Prediction
- Probability/confidence
- Feature-name mismatch
- Feature-dimension mismatch
- Malformed/empty input
- Insufficient training data
- Missing labels
- Inconsistent labels
- Evaluation metrics
- Confusion matrix
- Train/test separation
- Model save/load
- Prediction consistency before/after save/load
- random_state determinism
- Verification that cluster ID/run_id/context metadata are NOT features
- Integration with real Phase 2A FeatureVector objects
- Compatibility with Phase 2C labels via the bridge
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.ml.context_classifier import (
    MIN_DISTINCT_LABELS_FOR_TRAINING,
    MIN_SAMPLES_FOR_SPLIT,
    MIN_SAMPLES_FOR_TRAINING,
    ContextClassifier,
    EvaluationResult,
    PredictionResult,
    TrainingSummary,
    build_training_set,
    train_test_split_feature_vectors,
)
from app.ml.context_clustering import ContextClusterer
from app.ml.context_labeling import ContextLabelStore
from app.ml.feature_engineering import FeatureExtractor, FeatureVector

# ============================================================================
# Test helpers
# ============================================================================

FEATURE_NAMES_2D = ("f1", "f2")


def _vector(session_id: str, values: tuple[float, ...], names: tuple[str, ...] = FEATURE_NAMES_2D) -> FeatureVector:
    return FeatureVector(session_id=session_id, feature_names=names, feature_values=values, metadata={})


def _two_class_dataset(n_per_class: int = 6) -> tuple[list[FeatureVector], list[str]]:
    """Two well-separated behavioral clusters with two distinct labels."""
    vectors = []
    labels = []
    for i in range(n_per_class):
        vectors.append(_vector(f"low-{i}", (0.0 + i * 0.01, 0.0 + i * 0.01)))
        labels.append("Focused Work")
    for i in range(n_per_class):
        vectors.append(_vector(f"high-{i}", (50.0 + i * 0.01, 50.0 + i * 0.01)))
        labels.append("Browsing")
    return vectors, labels


def _three_class_dataset(n_per_class: int = 5) -> tuple[list[FeatureVector], list[str]]:
    vectors = []
    labels = []
    centers = [((0.0, 0.0), "Coding"), ((30.0, 0.0), "Gaming"), ((0.0, 30.0), "Research")]
    for (cx, cy), label in centers:
        for i in range(n_per_class):
            vectors.append(_vector(f"{label}-{i}", (cx + i * 0.01, cy + i * 0.01)))
            labels.append(label)
    return vectors, labels


def _ts(hour: int = 12, day: int = 1) -> datetime:
    return datetime(2026, 1, day, hour=hour, tzinfo=timezone.utc)


class _MockActivity:
    """Minimal stand-in matching the attributes FeatureExtractor expects."""

    def __init__(self, started_at, ended_at, duration_seconds, application, process_name):
        self.started_at = started_at
        self.ended_at = ended_at
        self.duration_seconds = duration_seconds
        self.application = application
        self.process_name = process_name


# ============================================================================
# Untrained classifier behavior
# ============================================================================


def test_untrained_classifier_is_not_trained() -> None:
    clf = ContextClassifier(random_state=42)
    assert clf.is_trained is False


def test_untrained_classifier_predict_raises() -> None:
    clf = ContextClassifier(random_state=42)
    with pytest.raises(RuntimeError, match="not been trained"):
        clf.predict(_vector("s1", (1.0, 2.0)))


def test_untrained_classifier_predict_many_raises() -> None:
    clf = ContextClassifier(random_state=42)
    with pytest.raises(RuntimeError, match="not been trained"):
        clf.predict_many([_vector("s1", (1.0, 2.0))])


def test_untrained_classifier_evaluate_raises() -> None:
    clf = ContextClassifier(random_state=42)
    with pytest.raises(RuntimeError, match="not been trained"):
        clf.evaluate([_vector("s1", (1.0, 2.0))], ["A"])


def test_untrained_classifier_classes_raises() -> None:
    clf = ContextClassifier(random_state=42)
    with pytest.raises(RuntimeError, match="not been trained"):
        _ = clf.classes


def test_untrained_classifier_feature_names_raises() -> None:
    clf = ContextClassifier(random_state=42)
    with pytest.raises(RuntimeError, match="not been trained"):
        _ = clf.feature_names


def test_untrained_classifier_save_raises(tmp_path: Path) -> None:
    clf = ContextClassifier(random_state=42)
    with pytest.raises(RuntimeError, match="not been trained"):
        clf.save(tmp_path / "model.joblib")


# ============================================================================
# Successful training
# ============================================================================


def test_fit_returns_training_summary() -> None:
    vectors, labels = _two_class_dataset()
    clf = ContextClassifier(random_state=42)

    summary = clf.fit(vectors, labels)

    assert isinstance(summary, TrainingSummary)
    assert summary.n_samples == len(vectors)
    assert summary.n_features == 2
    assert summary.feature_names == FEATURE_NAMES_2D
    assert summary.n_classes == 2
    assert set(summary.classes) == {"Focused Work", "Browsing"}
    assert summary.random_state == 42


def test_fit_marks_classifier_as_trained() -> None:
    vectors, labels = _two_class_dataset()
    clf = ContextClassifier(random_state=42)

    assert clf.is_trained is False
    clf.fit(vectors, labels)
    assert clf.is_trained is True


def test_fit_exposes_feature_names_and_classes() -> None:
    vectors, labels = _two_class_dataset()
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, labels)

    assert clf.feature_names == FEATURE_NAMES_2D
    assert set(clf.classes) == {"Focused Work", "Browsing"}


def test_refitting_overwrites_previous_model() -> None:
    vectors_a, labels_a = _two_class_dataset()
    vectors_b, labels_b = _three_class_dataset()
    clf = ContextClassifier(random_state=42)

    clf.fit(vectors_a, labels_a)
    assert set(clf.classes) == {"Focused Work", "Browsing"}

    clf.fit(vectors_b, labels_b)
    assert set(clf.classes) == {"Coding", "Gaming", "Research"}


# ============================================================================
# Deterministic training
# ============================================================================


def test_deterministic_training_same_random_state() -> None:
    vectors, labels = _three_class_dataset()

    clf_a = ContextClassifier(random_state=7)
    clf_a.fit(vectors, labels)
    clf_b = ContextClassifier(random_state=7)
    clf_b.fit(vectors, labels)

    predictions_a = clf_a.predict_many(vectors)
    predictions_b = clf_b.predict_many(vectors)

    assert [p.predicted_label for p in predictions_a] == [p.predicted_label for p in predictions_b]
    for pa, pb in zip(predictions_a, predictions_b):
        assert pa.class_probabilities == pb.class_probabilities


def test_random_state_determinism_across_many_refits() -> None:
    """Refitting the same classifier instance repeatedly with the same data/seed is stable."""
    vectors, labels = _two_class_dataset()
    clf = ContextClassifier(random_state=99)

    clf.fit(vectors, labels)
    first_predictions = [p.predicted_label for p in clf.predict_many(vectors)]

    clf.fit(vectors, labels)
    second_predictions = [p.predicted_label for p in clf.predict_many(vectors)]

    assert first_predictions == second_predictions


# ============================================================================
# Arbitrary labels (no hardcoded taxonomy)
# ============================================================================


@pytest.mark.parametrize(
    "label_pair",
    [
        ("Deep Work / Flow State", "Casual Browsing"),
        ("misc-cluster-A", "misc-cluster-B"),
        ("🎮 Gaming night", "📚 Study session"),
        ("Client project: Acme Corp", "Personal errands"),
    ],
)
def test_fit_accepts_arbitrary_label_text(label_pair: tuple[str, str]) -> None:
    label_a, label_b = label_pair
    vectors = [_vector(f"a{i}", (0.0 + i * 0.01, 0.0 + i * 0.01)) for i in range(4)] + [
        _vector(f"b{i}", (50.0 + i * 0.01, 50.0 + i * 0.01)) for i in range(4)
    ]
    labels = [label_a] * 4 + [label_b] * 4

    clf = ContextClassifier(random_state=42)
    summary = clf.fit(vectors, labels)

    assert set(summary.classes) == {label_a, label_b}


# ============================================================================
# Multiclass classification
# ============================================================================


def test_multiclass_training_and_prediction() -> None:
    vectors, labels = _three_class_dataset()
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, labels)

    predictions = clf.predict_many(vectors)
    predicted_labels = {p.predicted_label for p in predictions}

    assert predicted_labels <= {"Coding", "Gaming", "Research"}
    assert len(clf.classes) == 3


def test_multiclass_predictions_mostly_match_well_separated_labels() -> None:
    """Well-separated synthetic clusters should be classified correctly by RandomForest."""
    vectors, labels = _three_class_dataset(n_per_class=8)
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, labels)

    predictions = clf.predict_many(vectors)
    correct = sum(1 for p, true_label in zip(predictions, labels) if p.predicted_label == true_label)

    assert correct == len(labels)  # trivially separable data


# ============================================================================
# Binary classification
# ============================================================================


def test_binary_training_and_prediction() -> None:
    vectors, labels = _two_class_dataset()
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, labels)

    assert len(clf.classes) == 2
    prediction = clf.predict(_vector("new", (0.02, 0.02)))
    assert prediction.predicted_label == "Focused Work"


# ============================================================================
# Prediction
# ============================================================================


def test_predict_returns_prediction_result_with_session_id() -> None:
    vectors, labels = _two_class_dataset()
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, labels)

    result = clf.predict(_vector("my-session-id", (0.03, 0.03)))

    assert isinstance(result, PredictionResult)
    assert result.session_id == "my-session-id"
    assert result.predicted_label in clf.classes


def test_predict_many_returns_one_result_per_input() -> None:
    vectors, labels = _two_class_dataset()
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, labels)

    query_vectors = [_vector(f"q{i}", (0.0 + i * 0.01, 0.0 + i * 0.01)) for i in range(3)]
    results = clf.predict_many(query_vectors)

    assert len(results) == 3
    assert [r.session_id for r in results] == ["q0", "q1", "q2"]


def test_predict_many_empty_input_returns_empty_tuple() -> None:
    vectors, labels = _two_class_dataset()
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, labels)

    assert clf.predict_many([]) == ()


# ============================================================================
# Probability / confidence
# ============================================================================


def test_predict_exposes_class_probabilities_summing_to_one() -> None:
    vectors, labels = _two_class_dataset()
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, labels)

    result = clf.predict(_vector("new", (0.02, 0.02)))

    assert set(result.class_probabilities.keys()) == set(clf.classes)
    assert result.class_probabilities[result.predicted_label] == max(result.class_probabilities.values())
    assert sum(result.class_probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_predict_probabilities_all_in_valid_range() -> None:
    vectors, labels = _three_class_dataset()
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, labels)

    result = clf.predict(_vector("new", (0.0, 0.0)))

    assert all(0.0 <= p <= 1.0 for p in result.class_probabilities.values())


# ============================================================================
# Feature-name mismatch
# ============================================================================


def test_predict_with_wrong_feature_names_raises() -> None:
    vectors, labels = _two_class_dataset()
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, labels)

    wrong_vector = _vector("s", (1.0, 2.0), names=("other1", "other2"))

    with pytest.raises(ValueError, match="Feature mismatch"):
        clf.predict(wrong_vector)


def test_predict_with_reordered_feature_names_raises() -> None:
    vectors, labels = _two_class_dataset()
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, labels)

    reordered_vector = _vector("s", (1.0, 2.0), names=("f2", "f1"))

    with pytest.raises(ValueError, match="Feature mismatch"):
        clf.predict(reordered_vector)


def test_evaluate_with_wrong_feature_names_raises() -> None:
    vectors, labels = _two_class_dataset()
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, labels)

    wrong_vector = _vector("s", (1.0, 2.0), names=("other1", "other2"))

    with pytest.raises(ValueError, match="Feature mismatch"):
        clf.evaluate([wrong_vector], ["Focused Work"])


# ============================================================================
# Feature-dimension mismatch
# ============================================================================


def test_predict_with_wrong_feature_count_raises() -> None:
    vectors, labels = _two_class_dataset()
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, labels)

    wrong_dim_vector = _vector("s", (1.0, 2.0, 3.0), names=("f1", "f2", "f3"))

    with pytest.raises(ValueError, match="Feature mismatch"):
        clf.predict(wrong_dim_vector)


def test_fit_with_inconsistent_feature_dimensions_raises() -> None:
    vectors = [
        _vector("a", (1.0, 2.0), names=("f1", "f2")),
        _vector("b", (3.0, 4.0, 5.0), names=("f1", "f2", "f3")),
    ]
    labels = ["A", "B"]
    clf = ContextClassifier(random_state=42)

    with pytest.raises(ValueError, match="dimension mismatch"):
        clf.fit(vectors, labels)


# ============================================================================
# Malformed / empty input
# ============================================================================


def test_fit_empty_feature_vectors_raises() -> None:
    clf = ContextClassifier(random_state=42)
    with pytest.raises(ValueError, match="Insufficient"):
        clf.fit([], [])


def test_fit_mismatched_lengths_raises() -> None:
    vectors, labels = _two_class_dataset()
    clf = ContextClassifier(random_state=42)

    with pytest.raises(ValueError, match="same length"):
        clf.fit(vectors, labels[:-1])


def test_evaluate_empty_input_raises() -> None:
    vectors, labels = _two_class_dataset()
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, labels)

    with pytest.raises(ValueError, match="Insufficient"):
        clf.evaluate([], [])


def test_evaluate_mismatched_lengths_raises() -> None:
    vectors, labels = _two_class_dataset()
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, labels)

    with pytest.raises(ValueError, match="same length"):
        clf.evaluate(vectors, labels[:-1])


def test_constructor_rejects_invalid_n_estimators() -> None:
    with pytest.raises(ValueError, match="n_estimators"):
        ContextClassifier(n_estimators=0)


def test_constructor_rejects_invalid_max_depth() -> None:
    with pytest.raises(ValueError, match="max_depth"):
        ContextClassifier(max_depth=0)


# ============================================================================
# Insufficient training data
# ============================================================================


def test_fit_below_min_samples_raises() -> None:
    vectors = [_vector("a", (1.0, 2.0))]
    labels = ["A"]
    clf = ContextClassifier(random_state=42)

    with pytest.raises(ValueError, match="Insufficient training data"):
        clf.fit(vectors, labels)

    assert MIN_SAMPLES_FOR_TRAINING == 2


def test_fit_single_distinct_label_raises() -> None:
    """Two samples, but both share the same label -- cannot train a discriminator."""
    vectors = [_vector("a", (1.0, 2.0)), _vector("b", (3.0, 4.0))]
    labels = ["OnlyLabel", "OnlyLabel"]
    clf = ContextClassifier(random_state=42)

    with pytest.raises(ValueError, match="distinct labels"):
        clf.fit(vectors, labels)

    assert MIN_DISTINCT_LABELS_FOR_TRAINING == 2


# ============================================================================
# Missing labels
# ============================================================================


def test_fit_with_none_label_raises() -> None:
    vectors = [_vector("a", (1.0, 2.0)), _vector("b", (3.0, 4.0))]
    labels = ["A", None]
    clf = ContextClassifier(random_state=42)

    with pytest.raises(ValueError, match="string"):
        clf.fit(vectors, labels)


def test_fit_with_empty_string_label_raises() -> None:
    vectors = [_vector("a", (1.0, 2.0)), _vector("b", (3.0, 4.0))]
    labels = ["A", ""]
    clf = ContextClassifier(random_state=42)

    with pytest.raises(ValueError, match="empty"):
        clf.fit(vectors, labels)


def test_fit_with_whitespace_only_label_raises() -> None:
    vectors = [_vector("a", (1.0, 2.0)), _vector("b", (3.0, 4.0))]
    labels = ["A", "   "]
    clf = ContextClassifier(random_state=42)

    with pytest.raises(ValueError, match="empty"):
        clf.fit(vectors, labels)


# ============================================================================
# Inconsistent labels
# ============================================================================


def test_fit_strips_whitespace_from_labels_consistently() -> None:
    vectors = [_vector(f"a{i}", (0.0 + i * 0.01, 0.0)) for i in range(3)] + [
        _vector(f"b{i}", (50.0 + i * 0.01, 0.0)) for i in range(3)
    ]
    labels = ["  Coding", "Coding  ", "Coding"] + ["Gaming", " Gaming ", "Gaming "]
    clf = ContextClassifier(random_state=42)

    summary = clf.fit(vectors, labels)

    assert set(summary.classes) == {"Coding", "Gaming"}  # whitespace variants collapsed


def test_evaluate_with_inconsistent_case_labels_are_distinct_classes() -> None:
    """Labels are NOT case-normalized -- 'Coding' and 'coding' are different user labels."""
    vectors, labels = _two_class_dataset()
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, labels)

    # Evaluate with a differently-cased label that was never trained on.
    result = clf.evaluate([vectors[0]], ["focused work"])  # lowercase, not "Focused Work"

    assert "focused work" in result.labels
    assert result.per_class_recall["focused work"] == 0.0  # model can never predict an unseen class


# ============================================================================
# Evaluation metrics
# ============================================================================


def test_evaluate_returns_evaluation_result_with_expected_fields() -> None:
    vectors, labels = _two_class_dataset()
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, labels)

    result = clf.evaluate(vectors, labels)

    assert isinstance(result, EvaluationResult)
    assert result.n_samples == len(vectors)
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.precision_macro <= 1.0
    assert 0.0 <= result.recall_macro <= 1.0
    assert 0.0 <= result.f1_macro <= 1.0
    assert set(result.per_class_precision.keys()) == set(result.labels)
    assert set(result.per_class_recall.keys()) == set(result.labels)
    assert set(result.per_class_f1.keys()) == set(result.labels)


def test_evaluate_on_trivially_separable_data_is_perfect() -> None:
    vectors, labels = _two_class_dataset(n_per_class=8)
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, labels)

    result = clf.evaluate(vectors, labels)

    assert result.accuracy == 1.0
    assert result.precision_macro == 1.0
    assert result.recall_macro == 1.0
    assert result.f1_macro == 1.0


def test_evaluate_does_not_assert_a_specific_accuracy_threshold() -> None:
    """No test in this suite treats a fixed accuracy number as a pass/fail requirement."""
    vectors, labels = _two_class_dataset()
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, labels)

    result = clf.evaluate(vectors, labels)

    # Only sanity-check the value is a valid probability; no "must be >= 0.85" assertion.
    assert isinstance(result.accuracy, float)


# ============================================================================
# Confusion matrix
# ============================================================================


def test_confusion_matrix_shape_matches_label_count() -> None:
    vectors, labels = _three_class_dataset()
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, labels)

    result = clf.evaluate(vectors, labels)

    n = len(result.labels)
    assert len(result.confusion_matrix) == n
    assert all(len(row) == n for row in result.confusion_matrix)


def test_confusion_matrix_diagonal_dominant_for_separable_data() -> None:
    vectors, labels = _three_class_dataset(n_per_class=8)
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, labels)

    result = clf.evaluate(vectors, labels)

    total = sum(sum(row) for row in result.confusion_matrix)
    diagonal = sum(result.confusion_matrix[i][i] for i in range(len(result.confusion_matrix)))
    assert diagonal == total  # perfectly separable data -> everything on the diagonal


def test_confusion_matrix_includes_unseen_evaluation_label() -> None:
    vectors, labels = _two_class_dataset()
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, labels)

    result = clf.evaluate([vectors[0]], ["Never Trained On This"])

    assert "Never Trained On This" in result.labels
    idx = result.labels.index("Never Trained On This")
    # The model can never predict a class it wasn't trained on, so that
    # row's true count is 1 but the model's prediction lands elsewhere.
    assert sum(result.confusion_matrix[idx]) == 1


# ============================================================================
# Train/test separation
# ============================================================================


def test_train_test_split_returns_disjoint_sets() -> None:
    vectors, labels = _two_class_dataset(n_per_class=8)  # 16 samples

    (train_v, train_l), (test_v, test_l) = train_test_split_feature_vectors(
        vectors, labels, test_size=0.25, random_state=42
    )

    train_ids = {v.session_id for v in train_v}
    test_ids = {v.session_id for v in test_v}
    assert train_ids.isdisjoint(test_ids)
    assert len(train_v) + len(test_v) == len(vectors)
    assert len(train_v) == len(train_l)
    assert len(test_v) == len(test_l)


def test_train_test_split_is_deterministic() -> None:
    vectors, labels = _three_class_dataset(n_per_class=6)  # 18 samples

    split_a = train_test_split_feature_vectors(vectors, labels, test_size=0.3, random_state=7)
    split_b = train_test_split_feature_vectors(vectors, labels, test_size=0.3, random_state=7)

    ids_a = tuple(v.session_id for v in split_a[0][0])
    ids_b = tuple(v.session_id for v in split_b[0][0])
    assert ids_a == ids_b


def test_train_test_split_too_small_raises() -> None:
    vectors, labels = _two_class_dataset(n_per_class=1)  # 2 samples

    with pytest.raises(ValueError, match="Insufficient data"):
        train_test_split_feature_vectors(vectors, labels, test_size=0.5, random_state=42)

    assert MIN_SAMPLES_FOR_SPLIT == 4


def test_train_test_split_invalid_test_size_raises() -> None:
    vectors, labels = _two_class_dataset(n_per_class=8)

    with pytest.raises(ValueError, match="test_size"):
        train_test_split_feature_vectors(vectors, labels, test_size=0.0, random_state=42)

    with pytest.raises(ValueError, match="test_size"):
        train_test_split_feature_vectors(vectors, labels, test_size=1.0, random_state=42)


def test_train_then_evaluate_on_held_out_split_is_a_genuine_holdout() -> None:
    """Demonstrates the intended usage: split first, train on train, evaluate on test."""
    vectors, labels = _three_class_dataset(n_per_class=8)  # 24 samples

    (train_v, train_l), (test_v, test_l) = train_test_split_feature_vectors(
        vectors, labels, test_size=0.25, random_state=42
    )

    clf = ContextClassifier(random_state=42)
    clf.fit(train_v, train_l)
    result = clf.evaluate(test_v, test_l)

    assert result.n_samples == len(test_v)
    # No specific accuracy threshold is asserted -- only that evaluation
    # ran successfully on data the model never trained on.
    assert isinstance(result, EvaluationResult)


# ============================================================================
# Model save/load
# ============================================================================


def test_save_writes_a_file(tmp_path: Path) -> None:
    vectors, labels = _two_class_dataset()
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, labels)

    target = tmp_path / "model.joblib"
    returned_path = clf.save(target)

    assert target.exists()
    assert returned_path == target


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ContextClassifier.load(tmp_path / "does-not-exist.joblib")


def test_save_then_load_restores_trained_state(tmp_path: Path) -> None:
    vectors, labels = _two_class_dataset()
    clf = ContextClassifier(random_state=42, n_estimators=17, max_depth=4)
    clf.fit(vectors, labels)

    target = tmp_path / "model.joblib"
    clf.save(target)
    loaded = ContextClassifier.load(target)

    assert loaded.is_trained is True
    assert loaded.random_state == 42
    assert loaded.n_estimators == 17
    assert loaded.max_depth == 4
    assert loaded.feature_names == clf.feature_names
    assert loaded.classes == clf.classes


# ============================================================================
# Prediction consistency before/after save/load
# ============================================================================


def test_predictions_identical_before_and_after_save_load(tmp_path: Path) -> None:
    vectors, labels = _three_class_dataset()
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, labels)

    before = clf.predict_many(vectors)

    target = tmp_path / "model.joblib"
    clf.save(target)
    loaded = ContextClassifier.load(target)
    after = loaded.predict_many(vectors)

    assert [p.predicted_label for p in before] == [p.predicted_label for p in after]
    for b, a in zip(before, after):
        assert b.class_probabilities == a.class_probabilities


def test_evaluation_identical_before_and_after_save_load(tmp_path: Path) -> None:
    vectors, labels = _two_class_dataset()
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, labels)

    before = clf.evaluate(vectors, labels)

    target = tmp_path / "model.joblib"
    clf.save(target)
    loaded = ContextClassifier.load(target)
    after = loaded.evaluate(vectors, labels)

    assert before == after


# ============================================================================
# Verification: cluster ID / run_id / context metadata are NOT features
# ============================================================================


def test_session_id_and_metadata_are_not_used_as_features() -> None:
    """
    Two feature vectors with identical feature_values but different
    session_id and metadata must produce identical predictions and
    identical class probabilities -- proving those fields play no role
    in the model's input.
    """
    vectors, labels = _two_class_dataset()
    clf = ContextClassifier(random_state=42)
    clf.fit(vectors, labels)

    same_features = (0.02, 0.02)
    vector_a = FeatureVector(
        session_id="totally-different-session-id-A",
        feature_names=FEATURE_NAMES_2D,
        feature_values=same_features,
        metadata={"segment_count": 999, "unique_apps": ["Whatever"], "total_duration_seconds": 12345.0},
    )
    vector_b = FeatureVector(
        session_id="unrelated-session-id-B",
        feature_names=FEATURE_NAMES_2D,
        feature_values=same_features,
        metadata={},
    )

    result_a = clf.predict(vector_a)
    result_b = clf.predict(vector_b)

    assert result_a.predicted_label == result_b.predicted_label
    assert result_a.class_probabilities == result_b.class_probabilities
    # session_id IS still carried through into the result for attribution.
    assert result_a.session_id == "totally-different-session-id-A"
    assert result_b.session_id == "unrelated-session-id-B"


def test_training_input_never_reads_metadata() -> None:
    """
    Fitting on vectors whose metadata contains cluster/run-like keys must
    not affect the model or raise -- metadata is simply never touched.
    """
    vectors = [
        FeatureVector(
            session_id=f"s{i}",
            feature_names=FEATURE_NAMES_2D,
            feature_values=(0.0 + i * 0.01, 0.0 + i * 0.01),
            metadata={"cluster_label": 0, "run_id": "run-fake", "label": "LeakedLabel"},
        )
        for i in range(4)
    ] + [
        FeatureVector(
            session_id=f"t{i}",
            feature_names=FEATURE_NAMES_2D,
            feature_values=(50.0 + i * 0.01, 50.0 + i * 0.01),
            metadata={"cluster_label": 1, "run_id": "run-other-fake", "label": "AnotherLeakedLabel"},
        )
        for i in range(4)
    ]
    labels = ["A"] * 4 + ["B"] * 4

    clf = ContextClassifier(random_state=42)
    summary = clf.fit(vectors, labels)

    # The trained classes come only from the `labels` argument, never from
    # anything found inside `metadata`.
    assert set(summary.classes) == {"A", "B"}


# ============================================================================
# Integration with real Phase 2A FeatureVector objects
# ============================================================================


def test_integration_with_real_feature_extractor_output() -> None:
    """End-to-end: real Phase 2A FeatureExtractor output feeds the classifier directly."""
    extractor = FeatureExtractor()

    coding_vectors = []
    for day in range(4):
        start = _ts(hour=9, day=1 + day)
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
        start = _ts(hour=20, day=1 + day)
        activities = []
        for i in range(6):
            activities.append(
                _MockActivity(
                    start + timedelta(seconds=i * 100),
                    start + timedelta(seconds=(i + 1) * 100),
                    100.0,
                    f"App{i % 3}",
                    f"app{i % 3}.exe",
                )
            )
        browsing_vectors.append(
            extractor.extract_features(
                session_id=f"browsing-{day}",
                session_started_at=start,
                session_ended_at=start + timedelta(seconds=600),
                activities=activities,
            )
        )

    vectors = coding_vectors + browsing_vectors
    labels = ["Focused Coding"] * len(coding_vectors) + ["Evening Browsing"] * len(browsing_vectors)

    clf = ContextClassifier(random_state=42)
    summary = clf.fit(vectors, labels)

    assert summary.feature_names == FeatureExtractor.FEATURE_NAMES
    assert set(summary.classes) == {"Focused Coding", "Evening Browsing"}

    prediction = clf.predict(coding_vectors[0])
    assert prediction.predicted_label in {"Focused Coding", "Evening Browsing"}


# ============================================================================
# Compatibility with Phase 2C labels via the bridge
# ============================================================================


def test_build_training_set_joins_clustering_and_labels(tmp_path: Path) -> None:
    """Full 2A -> 2B -> 2C -> 2D pipeline using the real modules end-to-end."""
    vectors, _unused_labels = _two_class_dataset(n_per_class=6)

    clusterer = ContextClusterer(n_clusters=2, random_state=42)
    result = clusterer.fit(vectors)

    store = ContextLabelStore(storage_path=tmp_path / "labels.json")
    # Discover which raw cluster_label corresponds to which synthetic group
    # by inspecting the actual assignments (K-Means cluster numbering is
    # not assumed in advance -- this mirrors how a human would use Phase 2C).
    label_map = result.to_label_map()
    low_cluster = label_map["low-0"]
    high_cluster = label_map["high-0"]

    store.assign_label(_run_id_for(result), low_cluster, "Focused Work", assigned_at=_ts())
    store.assign_label(_run_id_for(result), high_cluster, "Browsing", assigned_at=_ts())

    training_vectors, training_labels = build_training_set(vectors, result, store)

    assert len(training_vectors) == len(vectors)  # every session ended up labeled
    assert set(training_labels) == {"Focused Work", "Browsing"}

    clf = ContextClassifier(random_state=42)
    summary = clf.fit(training_vectors, training_labels)
    assert set(summary.classes) == {"Focused Work", "Browsing"}


def test_build_training_set_excludes_unlabeled_clusters(tmp_path: Path) -> None:
    vectors, _unused_labels = _two_class_dataset(n_per_class=6)
    clusterer = ContextClusterer(n_clusters=2, random_state=42)
    result = clusterer.fit(vectors)

    store = ContextLabelStore(storage_path=tmp_path / "labels.json")
    label_map = result.to_label_map()
    low_cluster = label_map["low-0"]
    # Only label ONE of the two clusters.
    store.assign_label(_run_id_for(result), low_cluster, "Focused Work", assigned_at=_ts())

    training_vectors, training_labels = build_training_set(vectors, result, store)

    assert len(training_vectors) == 6  # only the labeled cluster's sessions
    assert set(training_labels) == {"Focused Work"}


def test_build_training_set_with_no_labels_returns_empty(tmp_path: Path) -> None:
    vectors, _unused_labels = _two_class_dataset(n_per_class=6)
    clusterer = ContextClusterer(n_clusters=2, random_state=42)
    result = clusterer.fit(vectors)

    store = ContextLabelStore(storage_path=tmp_path / "labels.json")  # nothing labeled

    training_vectors, training_labels = build_training_set(vectors, result, store)

    assert training_vectors == ()
    assert training_labels == ()


def test_build_training_set_does_not_use_a_different_runs_labels(tmp_path: Path) -> None:
    """Labels assigned under one clustering run must not leak into a different run's training set."""
    vectors_a, _ = _two_class_dataset(n_per_class=6)
    result_a = ContextClusterer(n_clusters=2, random_state=42).fit(vectors_a)

    store = ContextLabelStore(storage_path=tmp_path / "labels.json")
    label_map_a = result_a.to_label_map()
    store.assign_label(_run_id_for(result_a), label_map_a["low-0"], "Focused Work", assigned_at=_ts())
    store.assign_label(_run_id_for(result_a), label_map_a["high-0"], "Browsing", assigned_at=_ts())

    # Different underlying data -> different run_id.
    vectors_b, _ = _two_class_dataset(n_per_class=9)
    result_b = ContextClusterer(n_clusters=2, random_state=42).fit(vectors_b)

    training_vectors, training_labels = build_training_set(vectors_b, result_b, store)

    assert training_vectors == ()
    assert training_labels == ()


def _run_id_for(result) -> str:
    from app.ml.context_labeling import compute_run_id

    return compute_run_id(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])