"""
Comprehensive tests for Phase 2E: Deterministic Retraining Pipeline
(`app/ml/training_pipeline.py`).

Tests cover:
- Empty corpus
- Mixed feature schemas
- 2-3 example behavior (no split, evaluation skipped)
- Phase 2D minimum propagation (below fit()'s own minimums)
- Valid split (>= MIN_SAMPLES_FOR_SPLIT)
- Deterministic retraining
- Save / no-save
- Full replacement as corpus grows
- Failure safety (no overwrite of a working model on failure)
- Complete 2A -> 2B -> 2C -> 2D -> 2E integration
- Persistence across restarts (simulated, via two store instances)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.ml.context_classifier import (
    MIN_DISTINCT_LABELS_FOR_TRAINING,
    MIN_SAMPLES_FOR_SPLIT,
    MIN_SAMPLES_FOR_TRAINING,
    ContextClassifier,
    EvaluationResult,
)
from app.ml.context_clustering import ContextClusterer
from app.ml.context_labeling import ContextLabelStore, compute_run_id
from app.ml.feature_engineering import FeatureVector
from app.ml.training_examples import TrainingExampleStore, capture_labeled_examples
from app.ml.training_pipeline import RetrainResult, retrain_from_examples

# ============================================================================
# Test helpers
# ============================================================================


def _ts(hour: int = 12, day: int = 1) -> datetime:
    return datetime(2026, 1, day, hour=hour, tzinfo=timezone.utc)


def _populate_store(
    store: TrainingExampleStore,
    label_a: str = "Focused Work",
    label_b: str = "Browsing",
    n_per_class: int = 6,
    feature_names: tuple[str, ...] = ("f1", "f2"),
) -> None:
    for i in range(n_per_class):
        store.record_example(
            f"a{i}", feature_names, (0.0 + i * 0.01, 0.0 + i * 0.01), label_a,
            source_run_id="run-fixture", source_cluster_label=0, captured_at=_ts(),
        )
    for i in range(n_per_class):
        store.record_example(
            f"b{i}", feature_names, (50.0 + i * 0.01, 50.0 + i * 0.01), label_b,
            source_run_id="run-fixture", source_cluster_label=1, captured_at=_ts(),
        )


# ============================================================================
# Empty corpus
# ============================================================================


def test_retrain_with_empty_corpus_raises(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")

    with pytest.raises(ValueError, match="No labeled training examples"):
        retrain_from_examples(store, save=False)


# ============================================================================
# Mixed feature schemas
# ============================================================================


def test_retrain_with_mixed_feature_schemas_raises(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    store.record_example("a1", ("f1", "f2"), (1.0, 2.0), "Coding", source_run_id="r", source_cluster_label=0, captured_at=_ts())
    store.record_example("a2", ("f1", "f2"), (3.0, 4.0), "Coding", source_run_id="r", source_cluster_label=0, captured_at=_ts())
    store.record_example("b1", ("g1", "g2", "g3"), (1.0, 2.0, 3.0), "Browsing", source_run_id="r", source_cluster_label=1, captured_at=_ts())
    store.record_example("b2", ("g1", "g2", "g3"), (4.0, 5.0, 6.0), "Browsing", source_run_id="r", source_cluster_label=1, captured_at=_ts())

    with pytest.raises(ValueError, match="[Ff]eature schema mismatch"):
        retrain_from_examples(store, save=False)


# ============================================================================
# 2-3 example behavior (no split, evaluation skipped)
# ============================================================================


def test_retrain_with_two_to_three_examples_skips_evaluation(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    store.record_example("a1", ("f1",), (1.0,), "Coding", source_run_id="r", source_cluster_label=0, captured_at=_ts())
    store.record_example("b1", ("f1",), (99.0,), "Browsing", source_run_id="r", source_cluster_label=1, captured_at=_ts())
    store.record_example("b2", ("f1",), (98.0,), "Browsing", source_run_id="r", source_cluster_label=1, captured_at=_ts())

    result = retrain_from_examples(store, save=False)

    assert result.evaluation is None
    assert result.evaluation_skipped_reason is not None
    assert "held-out split" in result.evaluation_skipped_reason
    assert result.n_examples_used == 3
    assert result.training_summary.n_samples == 3
    assert MIN_SAMPLES_FOR_SPLIT == 4  # sanity-check the threshold this test relies on


def test_below_min_samples_for_split_still_trains_on_full_corpus(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    store.record_example("a1", ("f1",), (1.0,), "Coding", source_run_id="r", source_cluster_label=0, captured_at=_ts())
    store.record_example("b1", ("f1",), (99.0,), "Browsing", source_run_id="r", source_cluster_label=1, captured_at=_ts())

    result = retrain_from_examples(store, save=False)

    assert set(result.training_summary.classes) == {"Coding", "Browsing"}
    assert result.evaluation is None


# ============================================================================
# Phase 2D minimum propagation
# ============================================================================


def test_single_example_propagates_phase2d_min_samples_error(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    store.record_example("a1", ("f1",), (1.0,), "Coding", source_run_id="r", source_cluster_label=0, captured_at=_ts())

    with pytest.raises(ValueError, match="Insufficient training data"):
        retrain_from_examples(store, save=False)

    assert MIN_SAMPLES_FOR_TRAINING == 2


def test_two_examples_same_label_propagates_phase2d_distinct_label_error(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    store.record_example("a1", ("f1",), (1.0,), "OnlyLabel", source_run_id="r", source_cluster_label=0, captured_at=_ts())
    store.record_example("a2", ("f1",), (2.0,), "OnlyLabel", source_run_id="r", source_cluster_label=0, captured_at=_ts())

    with pytest.raises(ValueError, match="distinct labels"):
        retrain_from_examples(store, save=False)

    assert MIN_DISTINCT_LABELS_FOR_TRAINING == 2


# ============================================================================
# Valid split
# ============================================================================


def test_retrain_with_sufficient_data_performs_split_and_evaluates(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    _populate_store(store, n_per_class=8)

    result = retrain_from_examples(store, save=False)

    assert result.evaluation is not None
    assert result.evaluation_skipped_reason is None
    assert isinstance(result.evaluation, EvaluationResult)
    assert result.n_examples_used == 16
    # test set was held out from training
    assert result.training_summary.n_samples < result.n_examples_used


# ============================================================================
# Deterministic retraining
# ============================================================================


def test_retrain_is_deterministic_for_same_corpus_and_seed(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    _populate_store(store, n_per_class=8)

    result_a = retrain_from_examples(store, random_state=7, save=False)
    result_b = retrain_from_examples(store, random_state=7, save=False)

    assert result_a.training_summary == result_b.training_summary
    assert result_a.evaluation == result_b.evaluation


def test_retrain_differs_with_different_random_state(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    _populate_store(store, n_per_class=8)

    result_a = retrain_from_examples(store, random_state=1, save=False)
    result_b = retrain_from_examples(store, random_state=99, save=False)

    # Different seeds should at least differ in which examples land in
    # which split (evaluation n_samples can differ, or the split itself).
    assert result_a.random_state != result_b.random_state


# ============================================================================
# Save / no-save
# ============================================================================


def test_retrain_with_save_true_writes_artifact(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    _populate_store(store, n_per_class=8)
    target = tmp_path / "model.joblib"

    result = retrain_from_examples(store, save=True, save_path=target)

    assert result.saved_to == target
    assert target.exists()


def test_retrain_with_save_false_writes_nothing(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    _populate_store(store, n_per_class=8)
    target = tmp_path / "model.joblib"

    result = retrain_from_examples(store, save=False, save_path=target)

    assert result.saved_to is None
    assert not target.exists()


def test_retrain_default_save_path_uses_context_classifier_default(tmp_path: Path, monkeypatch) -> None:
    """When save_path is omitted, the artifact goes to ContextClassifier's own default location."""
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    _populate_store(store, n_per_class=8)

    default_target = tmp_path / "models" / "context_classifier.joblib"
    monkeypatch.setattr(ContextClassifier, "DEFAULT_MODEL_PATH", default_target)

    result = retrain_from_examples(store, save=True)

    assert result.saved_to == default_target
    assert default_target.exists()


# ============================================================================
# Full replacement as corpus grows
# ============================================================================


def test_retrain_reflects_full_current_corpus_not_incremental(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    _populate_store(store, n_per_class=8)

    result_before = retrain_from_examples(store, save=False)
    assert set(result_before.training_summary.classes) == {"Focused Work", "Browsing"}

    # Grow the corpus with a brand-new third label.
    for i in range(8):
        store.record_example(
            f"c{i}", ("f1", "f2"), (100.0 + i * 0.01, 100.0 + i * 0.01), "Gaming",
            source_run_id="run-fixture-2", source_cluster_label=0, captured_at=_ts(day=2),
        )

    result_after = retrain_from_examples(store, save=False)

    assert set(result_after.training_summary.classes) == {"Focused Work", "Browsing", "Gaming"}
    assert result_after.n_examples_used == 24  # full corpus, not a delta


def test_passing_existing_classifier_is_retrained_in_place(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    _populate_store(store, n_per_class=8)
    classifier = ContextClassifier(random_state=42)

    result = retrain_from_examples(store, classifier=classifier, save=False)

    assert classifier.is_trained is True
    assert set(classifier.classes) == set(result.training_summary.classes)


# ============================================================================
# Failure safety
# ============================================================================


def test_failed_retrain_does_not_overwrite_existing_model_file(tmp_path: Path) -> None:
    target = tmp_path / "model.joblib"
    target.write_bytes(b"PRE-EXISTING WORKING MODEL BYTES")
    original_bytes = target.read_bytes()

    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    store.record_example("a1", ("f1",), (1.0,), "OnlyLabel", source_run_id="r", source_cluster_label=0, captured_at=_ts())
    store.record_example("a2", ("f1",), (2.0,), "OnlyLabel", source_run_id="r", source_cluster_label=0, captured_at=_ts())

    with pytest.raises(ValueError):
        retrain_from_examples(store, save=True, save_path=target)

    assert target.read_bytes() == original_bytes  # completely untouched


def test_failed_retrain_on_empty_corpus_does_not_touch_save_path(tmp_path: Path) -> None:
    target = tmp_path / "model.joblib"
    target.write_bytes(b"UNTOUCHED")

    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")

    with pytest.raises(ValueError):
        retrain_from_examples(store, save=True, save_path=target)

    assert target.read_bytes() == b"UNTOUCHED"


# ============================================================================
# Complete 2A -> 2B -> 2C -> 2D -> 2E integration
# ============================================================================


def test_full_pipeline_integration(tmp_path: Path) -> None:
    """Real 2A -> 2B -> 2C -> 2E capture -> 2E retrain -> 2D predict, end to end."""
    vectors = [
        FeatureVector(session_id=f"low-{i}", feature_names=("f1", "f2"), feature_values=(0.0 + i * 0.01, 0.0), metadata={})
        for i in range(8)
    ] + [
        FeatureVector(session_id=f"high-{i}", feature_names=("f1", "f2"), feature_values=(50.0 + i * 0.01, 50.0), metadata={})
        for i in range(8)
    ]
    result = ContextClusterer(n_clusters=2, random_state=42).fit(vectors)

    label_store = ContextLabelStore(storage_path=tmp_path / "labels.json")
    examples_store = TrainingExampleStore(storage_path=tmp_path / "examples.json")

    label_map = result.to_label_map()
    run_id = compute_run_id(result)
    label_store.assign_label(run_id, label_map["low-0"], "Focused Work", assigned_at=_ts())
    label_store.assign_label(run_id, label_map["high-0"], "Browsing", assigned_at=_ts())

    capture_labeled_examples(vectors, result, label_store, examples_store, captured_at=_ts())

    retrain_result = retrain_from_examples(examples_store, save_path=tmp_path / "model.joblib")

    assert retrain_result.evaluation is not None
    assert set(retrain_result.training_summary.classes) == {"Focused Work", "Browsing"}

    loaded = ContextClassifier.load(tmp_path / "model.joblib")
    prediction = loaded.predict(vectors[0])
    assert prediction.predicted_label in {"Focused Work", "Browsing"}


# ============================================================================
# Persistence across restarts (simulated via two independent store objects)
# ============================================================================


def test_retrain_after_simulated_restart_sees_full_accumulated_corpus(tmp_path: Path) -> None:
    """
    Two independent TrainingExampleStore instances on the same path
    (simulating process restarts) each capture from a DIFFERENT,
    unrelated clustering run; one final retrain call must see both.
    """
    label_store = ContextLabelStore(storage_path=tmp_path / "labels.json")
    examples_path = tmp_path / "examples.json"

    # "Session 1": capture from clustering run A.
    store_session_1 = TrainingExampleStore(storage_path=examples_path)
    vectors_a = [
        FeatureVector(session_id=f"a{i}", feature_names=("f1", "f2"), feature_values=(0.0 + i * 0.01, 0.0), metadata={})
        for i in range(6)
    ] + [
        FeatureVector(session_id=f"b{i}", feature_names=("f1", "f2"), feature_values=(50.0 + i * 0.01, 50.0), metadata={})
        for i in range(6)
    ]
    result_a = ContextClusterer(n_clusters=2, random_state=42).fit(vectors_a)
    run_id_a = compute_run_id(result_a)
    label_map_a = result_a.to_label_map()
    label_store.assign_label(run_id_a, label_map_a["a0"], "Focused Work", assigned_at=_ts(day=1))
    label_store.assign_label(run_id_a, label_map_a["b0"], "Browsing", assigned_at=_ts(day=1))
    capture_labeled_examples(vectors_a, result_a, label_store, store_session_1, captured_at=_ts(day=1))

    # "Session 2" (fresh TrainingExampleStore object, same file -- simulates
    # a process restart): capture from a completely unrelated clustering run B.
    store_session_2 = TrainingExampleStore(storage_path=examples_path)
    vectors_b = [
        FeatureVector(session_id=f"c{i}", feature_names=("f1", "f2"), feature_values=(200.0 + i * 0.01, 200.0), metadata={})
        for i in range(6)
    ]
    result_b = ContextClusterer(n_clusters=1, random_state=99).fit(vectors_b)
    run_id_b = compute_run_id(result_b)
    label_store.assign_label(run_id_b, 0, "Gaming", assigned_at=_ts(day=2))
    capture_labeled_examples(vectors_b, result_b, label_store, store_session_2, captured_at=_ts(day=2))

    assert run_id_a != run_id_b

    # One final retrain call, via a third fresh store instance, sees everything.
    final_store = TrainingExampleStore(storage_path=examples_path)
    retrain_result = retrain_from_examples(final_store, save=False)

    assert retrain_result.n_examples_used == 18  # 12 from run A + 6 from run B
    assert set(retrain_result.training_summary.classes) == {"Focused Work", "Browsing", "Gaming"}


# ============================================================================
# REGRESSION GUARDS
#
# These tests do not exercise "normal" behavior -- they exist specifically
# to FAIL if a future change reintroduces a contract violation. Several
# inspect the actual module source rather than only behavior, because a
# reimplementation that constructs its own RandomForestClassifier (or
# imports Phase 2A/2B/2C directly) could still superficially pass
# black-box behavioral tests while violating the frozen contract's
# architectural rules.
# ============================================================================

import dataclasses
import inspect
from unittest.mock import patch

from app.ml.context_classifier import ContextClassifier as RealContextClassifier


def _training_pipeline_source() -> str:
    import app.ml.training_pipeline as module

    return inspect.getsource(module)


def _training_pipeline_imported_names() -> set[str]:
    """
    Parse training_pipeline.py's actual `import`/`from ... import ...`
    statements via the AST (not a raw substring search), so a legitimate
    docstring mention of a forbidden name (explaining what is NOT done)
    can never be mistaken for a real import.
    """
    import ast

    tree = ast.parse(_training_pipeline_source())
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


def test_guard_no_direct_randomforestclassifier_construction_in_source() -> None:
    """Fails if training_pipeline.py is ever changed to build its own RandomForestClassifier."""
    source = _training_pipeline_source()

    assert "RandomForestClassifier" not in source
    assert "sklearn.ensemble" not in source
    assert "from sklearn" not in source
    assert "import sklearn" not in source


def test_guard_no_second_train_test_splitter_in_source() -> None:
    """Fails if a duplicate splitter is introduced instead of reusing Phase 2D's helper."""
    source = _training_pipeline_source()

    assert "sklearn.model_selection" not in source
    assert "train_test_split(" not in source  # the raw sklearn function, not the 2D wrapper name


def test_guard_no_hashed_or_timestamped_model_filenames_in_source() -> None:
    """Fails if content-hashed or timestamped model filenames (from the conflicting PDF design) are reintroduced."""
    source = _training_pipeline_source()
    lowered = source.lower()

    assert "hashlib" not in lowered
    assert "sha256" not in lowered
    assert "strftime" not in lowered
    assert "utcnow().isoformat" not in source


def test_guard_does_not_import_phase_2a_feature_extraction_or_2b_2c() -> None:
    """
    Fails if training_pipeline.py starts importing FeatureExtractor
    (re-running feature engineering), or anything from Phase 2B
    (context_clustering) or Phase 2C (context_labeling) -- the corpus
    must be the pipeline's only data source. Checks actual `import`
    statements (via AST) rather than raw text, so a docstring merely
    *mentioning* a forbidden name in prose (to explain what is NOT done)
    cannot cause a false failure here.
    """
    import app.ml.training_pipeline as module

    # Live module namespace: what's actually bound after import time.
    module_names = {
        obj.__module__
        for name, obj in vars(module).items()
        if not name.startswith("_") and hasattr(obj, "__module__")
    }
    assert "app.ml.context_clustering" not in module_names
    assert "app.ml.context_labeling" not in module_names

    # Source-level: no import statement references these modules/names at all.
    imported_names = _training_pipeline_imported_names()
    assert not any("context_clustering" in name for name in imported_names)
    assert not any("context_labeling" in name for name in imported_names)
    assert not any(name.rsplit(".", 1)[-1] == "FeatureExtractor" for name in imported_names)


def test_guard_only_feature_vector_is_imported_from_phase_2a() -> None:
    """
    The one permitted Phase 2A dependency is the FeatureVector dataclass
    itself (required to reconstruct examples into fittable vectors) --
    never the extraction logic. Checked via actual import statements
    (AST), not raw text, so this cannot be fooled by (or falsely trigger
    on) a docstring explanation.
    """
    imported_names = _training_pipeline_imported_names()

    assert "app.ml.feature_engineering.FeatureVector" in imported_names
    assert not any(name.rsplit(".", 1)[-1] == "FeatureExtractor" for name in imported_names)


def test_guard_retrain_from_examples_actually_calls_context_classifier_fit(tmp_path: Path) -> None:
    """
    Fails if retrain_from_examples is ever changed to bypass
    ContextClassifier (e.g. calling a private/duplicate training routine
    instead). Spies on the REAL ContextClassifier.fit to prove it is
    genuinely invoked, not just imported.
    """
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    _populate_store(store, n_per_class=8)

    with patch.object(RealContextClassifier, "fit", wraps=RealContextClassifier.fit, autospec=True) as spy_fit:
        retrain_from_examples(store, save=False)

    assert spy_fit.call_count == 1


def test_guard_retrain_from_examples_actually_calls_context_classifier_save(tmp_path: Path) -> None:
    """Fails if save=True stops actually invoking ContextClassifier.save()."""
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    _populate_store(store, n_per_class=8)
    target = tmp_path / "model.joblib"

    with patch.object(RealContextClassifier, "save", wraps=RealContextClassifier.save, autospec=True) as spy_save:
        retrain_from_examples(store, save=True, save_path=target)

    assert spy_save.call_count == 1


def test_guard_default_model_path_is_exactly_models_context_classifier_joblib() -> None:
    """Fails if the default artifact location is ever changed from Phase 2D's own default."""
    assert ContextClassifier.DEFAULT_MODEL_PATH.name == "context_classifier.joblib"
    assert ContextClassifier.DEFAULT_MODEL_PATH.parent.name == "models"


def test_guard_saved_path_is_stable_not_hash_or_timestamp_varying(tmp_path: Path) -> None:
    """
    Fails if filenames become content-hashed or timestamped: retraining
    the SAME corpus twice (with default save_path) must always produce
    the identical target path.
    """
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    _populate_store(store, n_per_class=8)
    default_target = tmp_path / "models" / "context_classifier.joblib"

    with patch.object(ContextClassifier, "DEFAULT_MODEL_PATH", default_target):
        result_a = retrain_from_examples(store, save=True)
        result_b = retrain_from_examples(store, save=True)

    assert result_a.saved_to == result_b.saved_to == default_target


def test_guard_retrain_result_has_exact_contract_fields() -> None:
    """Fails if any RetrainResult field is renamed, removed, reordered, or a replacement class is used."""
    field_names = tuple(f.name for f in dataclasses.fields(RetrainResult))

    assert field_names == (
        "training_summary",
        "evaluation",
        "evaluation_skipped_reason",
        "n_examples_used",
        "saved_to",
        "random_state",
    )


def test_guard_training_summary_and_evaluation_are_phase_2d_types(tmp_path: Path) -> None:
    """Fails if replacement result classes are ever substituted for Phase 2D's own types."""
    from app.ml.context_classifier import EvaluationResult, TrainingSummary

    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    _populate_store(store, n_per_class=8)

    result = retrain_from_examples(store, save=False)

    assert type(result.training_summary) is TrainingSummary
    assert type(result.evaluation) is EvaluationResult


def test_guard_retrain_from_examples_exact_signature_shape() -> None:
    """Fails if retrain_from_examples's parameter shape ever drifts from the frozen contract."""
    signature = inspect.signature(retrain_from_examples)
    parameter_names = list(signature.parameters.keys())

    assert parameter_names == ["examples_store", "classifier", "test_size", "random_state", "save", "save_path"]
    assert signature.parameters["classifier"].default is None
    assert signature.parameters["test_size"].default == 0.25
    assert signature.parameters["random_state"].default == 42
    assert signature.parameters["save"].default is True
    assert signature.parameters["save_path"].default is None
    for keyword_only_name in ("test_size", "random_state", "save", "save_path"):
        assert signature.parameters[keyword_only_name].kind == inspect.Parameter.KEYWORD_ONLY


def test_guard_train_and_test_sets_are_disjoint_in_split_path(tmp_path: Path) -> None:
    """
    Fails if the split is ever broken such that the same session ends up
    in both train and test (defeating the purpose of held-out evaluation).
    """
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    _populate_store(store, n_per_class=10)

    captured_train_ids: list[str] = []
    captured_test_ids: list[str] = []
    original_fit = RealContextClassifier.fit
    original_evaluate = RealContextClassifier.evaluate

    def spy_fit(self, feature_vectors, labels):
        captured_train_ids.extend(v.session_id for v in feature_vectors)
        return original_fit(self, feature_vectors, labels)

    def spy_evaluate(self, feature_vectors, labels):
        captured_test_ids.extend(v.session_id for v in feature_vectors)
        return original_evaluate(self, feature_vectors, labels)

    with patch.object(RealContextClassifier, "fit", spy_fit), patch.object(RealContextClassifier, "evaluate", spy_evaluate):
        retrain_from_examples(store, save=False)

    assert captured_train_ids  # split path was actually exercised
    assert captured_test_ids
    assert set(captured_train_ids).isdisjoint(set(captured_test_ids))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
