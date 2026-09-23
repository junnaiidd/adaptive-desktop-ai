"""
Comprehensive tests for Phase 2E: Durable Training-Example Corpus
(`app/ml/training_examples.py`).

Tests cover:
- CRUD (record/get/remove/list)
- Deterministic serialization
- Persistence across restarts (new object, same path)
- Validation (malformed/empty input)
- Labeled capture via capture_labeled_examples
- Unlabeled-session skipping
- Missing-session skipping
- Upsert behavior (dedup by session_id)
- Accumulation across independent clustering runs
- Anti-leakage (provenance never enters feature_values)
- Provenance fields stored correctly
- Schema handling (feature_schema_version)
- Real Phase 2A -> 2B -> 2C integration
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.ml.context_clustering import ContextClusterer
from app.ml.context_labeling import ContextLabelStore, compute_run_id
from app.ml.feature_engineering import FeatureVector
from app.ml.training_examples import (
    CURRENT_FEATURE_SCHEMA_VERSION,
    LabeledExample,
    TrainingExampleStore,
    capture_labeled_examples,
)

# ============================================================================
# Test helpers
# ============================================================================

FEATURE_NAMES_2D = ("f1", "f2")


def _vector(session_id: str, values: tuple[float, ...], names: tuple[str, ...] = FEATURE_NAMES_2D) -> FeatureVector:
    return FeatureVector(session_id=session_id, feature_names=names, feature_values=values, metadata={})


def _ts(hour: int = 12, day: int = 1) -> datetime:
    return datetime(2026, 1, day, hour=hour, tzinfo=timezone.utc)


def _two_well_separated_groups(n_per_group: int = 6) -> list[FeatureVector]:
    vectors = []
    for i in range(n_per_group):
        vectors.append(_vector(f"low-{i}", (0.0 + i * 0.01, 0.0 + i * 0.01)))
    for i in range(n_per_group):
        vectors.append(_vector(f"high-{i}", (50.0 + i * 0.01, 50.0 + i * 0.01)))
    return vectors


def _fit_two_clusters(vectors=None):
    vectors = vectors if vectors is not None else _two_well_separated_groups()
    return ContextClusterer(n_clusters=2, random_state=42).fit(vectors)


# ============================================================================
# CRUD
# ============================================================================


def test_record_example_stores_and_returns_entry(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")

    entry = store.record_example(
        "s1", ("f1", "f2"), (1.0, 2.0), "Coding",
        source_run_id="run-abc", source_cluster_label=0, captured_at=_ts(),
    )

    assert isinstance(entry, LabeledExample)
    assert entry.session_id == "s1"
    assert entry.feature_names == ("f1", "f2")
    assert entry.feature_values == (1.0, 2.0)
    assert entry.label == "Coding"
    assert entry.source_run_id == "run-abc"
    assert entry.source_cluster_label == 0
    assert entry.feature_schema_version == CURRENT_FEATURE_SCHEMA_VERSION
    assert entry.created_at == _ts()
    assert entry.updated_at == _ts()


def test_get_example_returns_stored_entry(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    store.record_example("s1", ("f1",), (1.0,), "Coding", source_run_id="r", source_cluster_label=0, captured_at=_ts())

    entry = store.get_example("s1")

    assert entry is not None
    assert entry.label == "Coding"


def test_get_example_unknown_session_returns_none(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")

    assert store.get_example("nonexistent") is None


def test_remove_example_deletes_existing(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    store.record_example("s1", ("f1",), (1.0,), "Coding", source_run_id="r", source_cluster_label=0, captured_at=_ts())

    removed = store.remove_example("s1")

    assert removed is True
    assert store.get_example("s1") is None


def test_remove_example_unknown_returns_false(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")

    assert store.remove_example("nonexistent") is False


def test_remove_example_does_not_affect_others(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    store.record_example("s1", ("f1",), (1.0,), "Coding", source_run_id="r", source_cluster_label=0, captured_at=_ts())
    store.record_example("s2", ("f1",), (2.0,), "Browsing", source_run_id="r", source_cluster_label=1, captured_at=_ts())

    store.remove_example("s1")

    assert store.get_example("s1") is None
    assert store.get_example("s2").label == "Browsing"


def test_list_examples_returns_all_sorted_by_session_id(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    store.record_example("zeta", ("f1",), (1.0,), "A", source_run_id="r", source_cluster_label=0, captured_at=_ts())
    store.record_example("alpha", ("f1",), (2.0,), "B", source_run_id="r", source_cluster_label=1, captured_at=_ts())

    examples = store.list_examples()

    assert [e.session_id for e in examples] == ["alpha", "zeta"]


def test_list_examples_by_label_filters_correctly(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    store.record_example("s1", ("f1",), (1.0,), "Coding", source_run_id="r", source_cluster_label=0, captured_at=_ts())
    store.record_example("s2", ("f1",), (2.0,), "Browsing", source_run_id="r", source_cluster_label=1, captured_at=_ts())
    store.record_example("s3", ("f1",), (3.0,), "Coding", source_run_id="r", source_cluster_label=0, captured_at=_ts())

    coding = store.list_examples_by_label("Coding")

    assert {e.session_id for e in coding} == {"s1", "s3"}


def test_list_examples_empty_store_returns_empty_tuple(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")

    assert store.list_examples() == ()
    assert store.list_examples_by_label("anything") == ()


def test_store_creates_file_on_first_use(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "examples.json"
    assert not path.exists()

    TrainingExampleStore(storage_path=path)

    assert path.exists()


# ============================================================================
# Deterministic serialization
# ============================================================================


def test_repeated_writes_of_same_content_produce_identical_bytes(tmp_path: Path) -> None:
    path_a = tmp_path / "examples_a.json"
    path_b = tmp_path / "examples_b.json"
    store_a = TrainingExampleStore(storage_path=path_a)
    store_b = TrainingExampleStore(storage_path=path_b)

    store_a.record_example("s1", ("f1", "f2"), (1.0, 2.0), "Coding", source_run_id="r", source_cluster_label=0, captured_at=_ts())
    store_a.record_example("s2", ("f1", "f2"), (3.0, 4.0), "Browsing", source_run_id="r", source_cluster_label=1, captured_at=_ts(hour=13))
    store_b.record_example("s1", ("f1", "f2"), (1.0, 2.0), "Coding", source_run_id="r", source_cluster_label=0, captured_at=_ts())
    store_b.record_example("s2", ("f1", "f2"), (3.0, 4.0), "Browsing", source_run_id="r", source_cluster_label=1, captured_at=_ts(hour=13))

    assert path_a.read_text(encoding="utf-8") == path_b.read_text(encoding="utf-8")


# ============================================================================
# Persistence across restarts
# ============================================================================


def test_examples_persist_across_new_store_instance_same_path(tmp_path: Path) -> None:
    path = tmp_path / "examples.json"
    store_a = TrainingExampleStore(storage_path=path)
    store_a.record_example("s1", ("f1",), (1.0,), "Coding", source_run_id="r", source_cluster_label=0, captured_at=_ts())

    # Simulate a fresh process/module reload: brand-new object, same file.
    store_b = TrainingExampleStore(storage_path=path)

    entry = store_b.get_example("s1")
    assert entry is not None
    assert entry.label == "Coding"


def test_removal_persists_across_restart(tmp_path: Path) -> None:
    path = tmp_path / "examples.json"
    store_a = TrainingExampleStore(storage_path=path)
    store_a.record_example("s1", ("f1",), (1.0,), "Coding", source_run_id="r", source_cluster_label=0, captured_at=_ts())
    store_a.remove_example("s1")

    store_b = TrainingExampleStore(storage_path=path)

    assert store_b.get_example("s1") is None
    assert store_b.list_examples() == ()


# ============================================================================
# Validation
# ============================================================================


def test_record_example_rejects_mismatched_lengths(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")

    with pytest.raises(ValueError, match="same length"):
        store.record_example("s1", ("f1", "f2"), (1.0,), "Coding", source_run_id="r", source_cluster_label=0, captured_at=_ts())


def test_record_example_rejects_empty_feature_names(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")

    with pytest.raises(ValueError, match="empty"):
        store.record_example("s1", (), (), "Coding", source_run_id="r", source_cluster_label=0, captured_at=_ts())


def test_record_example_rejects_non_finite_values(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")

    with pytest.raises(ValueError, match="finite"):
        store.record_example("s1", ("f1",), (float("nan"),), "Coding", source_run_id="r", source_cluster_label=0, captured_at=_ts())


def test_record_example_rejects_empty_session_id(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")

    with pytest.raises(ValueError, match="session_id"):
        store.record_example("", ("f1",), (1.0,), "Coding", source_run_id="r", source_cluster_label=0, captured_at=_ts())


def test_record_example_rejects_empty_label(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")

    with pytest.raises(ValueError, match="empty"):
        store.record_example("s1", ("f1",), (1.0,), "", source_run_id="r", source_cluster_label=0, captured_at=_ts())


def test_record_example_rejects_whitespace_only_label(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")

    with pytest.raises(ValueError, match="empty"):
        store.record_example("s1", ("f1",), (1.0,), "   ", source_run_id="r", source_cluster_label=0, captured_at=_ts())


def test_record_example_rejects_negative_cluster_label(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")

    with pytest.raises(ValueError, match="source_cluster_label"):
        store.record_example("s1", ("f1",), (1.0,), "Coding", source_run_id="r", source_cluster_label=-1, captured_at=_ts())


def test_record_example_rejects_empty_run_id(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")

    with pytest.raises(ValueError, match="source_run_id"):
        store.record_example("s1", ("f1",), (1.0,), "Coding", source_run_id="", source_cluster_label=0, captured_at=_ts())


def test_record_example_rejects_timezone_naive_timestamp(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    naive = datetime(2026, 1, 1, 12, 0, 0)  # no tzinfo

    with pytest.raises(ValueError, match="timezone-aware"):
        store.record_example("s1", ("f1",), (1.0,), "Coding", source_run_id="r", source_cluster_label=0, captured_at=naive)


def test_nothing_persisted_after_rejected_record(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")

    with pytest.raises(ValueError):
        store.record_example("s1", ("f1",), (1.0,), "", source_run_id="r", source_cluster_label=0, captured_at=_ts())

    assert store.list_examples() == ()


# ============================================================================
# Upsert behavior (dedup by session_id)
# ============================================================================


def test_record_example_upserts_by_session_id(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    store.record_example("s1", ("f1",), (1.0,), "Coding", source_run_id="r1", source_cluster_label=0, captured_at=_ts(hour=9))

    updated = store.record_example("s1", ("f1",), (2.0,), "Browsing", source_run_id="r2", source_cluster_label=1, captured_at=_ts(hour=10))

    assert len(store.list_examples()) == 1
    assert updated.feature_values == (2.0,)
    assert updated.label == "Browsing"
    assert updated.source_run_id == "r2"


def test_upsert_preserves_original_created_at(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    first = store.record_example("s1", ("f1",), (1.0,), "Coding", source_run_id="r", source_cluster_label=0, captured_at=_ts(hour=9))

    second = store.record_example("s1", ("f1",), (2.0,), "Browsing", source_run_id="r", source_cluster_label=1, captured_at=_ts(hour=15))

    assert second.created_at == first.created_at == _ts(hour=9)
    assert second.updated_at == _ts(hour=15)


# ============================================================================
# capture_labeled_examples: labeled capture
# ============================================================================


def test_capture_labeled_examples_persists_all_labeled_sessions(tmp_path: Path) -> None:
    vectors = _two_well_separated_groups(n_per_group=6)
    result = _fit_two_clusters(vectors)
    label_store = ContextLabelStore(storage_path=tmp_path / "labels.json")
    examples_store = TrainingExampleStore(storage_path=tmp_path / "examples.json")

    label_map = result.to_label_map()
    run_id = compute_run_id(result)
    label_store.assign_label(run_id, label_map["low-0"], "Focused Work", assigned_at=_ts())
    label_store.assign_label(run_id, label_map["high-0"], "Browsing", assigned_at=_ts())

    captured = capture_labeled_examples(vectors, result, label_store, examples_store, captured_at=_ts())

    assert len(captured) == 12
    assert len(examples_store.list_examples()) == 12
    assert {e.label for e in examples_store.list_examples()} == {"Focused Work", "Browsing"}


def test_capture_labeled_examples_stores_correct_provenance(tmp_path: Path) -> None:
    vectors = _two_well_separated_groups(n_per_group=6)
    result = _fit_two_clusters(vectors)
    label_store = ContextLabelStore(storage_path=tmp_path / "labels.json")
    examples_store = TrainingExampleStore(storage_path=tmp_path / "examples.json")

    label_map = result.to_label_map()
    run_id = compute_run_id(result)
    label_store.assign_label(run_id, label_map["low-0"], "Focused Work", assigned_at=_ts())
    label_store.assign_label(run_id, label_map["high-0"], "Browsing", assigned_at=_ts())

    capture_labeled_examples(vectors, result, label_store, examples_store, captured_at=_ts())

    example = examples_store.get_example("low-0")
    assert example.source_run_id == run_id
    assert example.source_cluster_label == label_map["low-0"]


def test_capture_labeled_examples_returns_only_newly_written(tmp_path: Path) -> None:
    vectors = _two_well_separated_groups(n_per_group=6)
    result = _fit_two_clusters(vectors)
    label_store = ContextLabelStore(storage_path=tmp_path / "labels.json")
    examples_store = TrainingExampleStore(storage_path=tmp_path / "examples.json")

    label_map = result.to_label_map()
    run_id = compute_run_id(result)
    label_store.assign_label(run_id, label_map["low-0"], "Focused Work", assigned_at=_ts())
    label_store.assign_label(run_id, label_map["high-0"], "Browsing", assigned_at=_ts())

    captured = capture_labeled_examples(vectors, result, label_store, examples_store, captured_at=_ts())

    assert len(captured) == 12
    assert all(isinstance(e, LabeledExample) for e in captured)


# ============================================================================
# Unlabeled-session skipping
# ============================================================================


def test_capture_labeled_examples_excludes_unlabeled_clusters(tmp_path: Path) -> None:
    vectors = _two_well_separated_groups(n_per_group=6)
    result = _fit_two_clusters(vectors)
    label_store = ContextLabelStore(storage_path=tmp_path / "labels.json")
    examples_store = TrainingExampleStore(storage_path=tmp_path / "examples.json")

    label_map = result.to_label_map()
    run_id = compute_run_id(result)
    # Only label ONE of the two clusters.
    label_store.assign_label(run_id, label_map["low-0"], "Focused Work", assigned_at=_ts())

    captured = capture_labeled_examples(vectors, result, label_store, examples_store, captured_at=_ts())

    assert len(captured) == 6
    assert {e.label for e in examples_store.list_examples()} == {"Focused Work"}


def test_capture_labeled_examples_with_no_labels_captures_nothing(tmp_path: Path) -> None:
    vectors = _two_well_separated_groups(n_per_group=6)
    result = _fit_two_clusters(vectors)
    label_store = ContextLabelStore(storage_path=tmp_path / "labels.json")  # nothing labeled
    examples_store = TrainingExampleStore(storage_path=tmp_path / "examples.json")

    captured = capture_labeled_examples(vectors, result, label_store, examples_store, captured_at=_ts())

    assert captured == ()
    assert examples_store.list_examples() == ()


# ============================================================================
# Missing-session skipping
# ============================================================================


def test_capture_labeled_examples_skips_vectors_not_in_result(tmp_path: Path) -> None:
    vectors = _two_well_separated_groups(n_per_group=6)
    result = _fit_two_clusters(vectors)
    label_store = ContextLabelStore(storage_path=tmp_path / "labels.json")
    examples_store = TrainingExampleStore(storage_path=tmp_path / "examples.json")

    label_map = result.to_label_map()
    run_id = compute_run_id(result)
    label_store.assign_label(run_id, label_map["low-0"], "Focused Work", assigned_at=_ts())
    label_store.assign_label(run_id, label_map["high-0"], "Browsing", assigned_at=_ts())

    # Extra vector whose session_id was never part of the clustering result.
    extra_vector = _vector("totally-unrelated-session", (999.0, 999.0))
    captured = capture_labeled_examples(
        list(vectors) + [extra_vector], result, label_store, examples_store, captured_at=_ts()
    )

    assert len(captured) == 12  # extra vector excluded
    assert examples_store.get_example("totally-unrelated-session") is None


# ============================================================================
# Accumulation across independent clustering runs
# ============================================================================


def test_capture_accumulates_across_two_independent_runs(tmp_path: Path) -> None:
    """The flagship proof the Phase 2C persistence gap is closed."""
    label_store = ContextLabelStore(storage_path=tmp_path / "labels.json")
    examples_store = TrainingExampleStore(storage_path=tmp_path / "examples.json")

    # Run A: two contexts.
    vectors_a = _two_well_separated_groups(n_per_group=5)
    result_a = _fit_two_clusters(vectors_a)
    run_id_a = compute_run_id(result_a)
    label_map_a = result_a.to_label_map()
    label_store.assign_label(run_id_a, label_map_a["low-0"], "Focused Work", assigned_at=_ts(day=1))
    label_store.assign_label(run_id_a, label_map_a["high-0"], "Browsing", assigned_at=_ts(day=1))
    capture_labeled_examples(vectors_a, result_a, label_store, examples_store, captured_at=_ts(day=1))

    # Run B ("later" / a different, unrelated clustering run): a brand-new
    # third context that didn't exist in run A at all.
    vectors_b = [_vector(f"gaming-{i}", (200.0 + i * 0.01, 200.0)) for i in range(5)]
    result_b = ContextClusterer(n_clusters=1, random_state=7).fit(vectors_b)
    run_id_b = compute_run_id(result_b)
    label_store.assign_label(run_id_b, 0, "Gaming", assigned_at=_ts(day=2))
    capture_labeled_examples(vectors_b, result_b, label_store, examples_store, captured_at=_ts(day=2))

    assert run_id_a != run_id_b
    all_examples = examples_store.list_examples()
    assert len(all_examples) == 15  # 10 from run A + 5 from run B
    assert {e.label for e in all_examples} == {"Focused Work", "Browsing", "Gaming"}


# ============================================================================
# Anti-leakage
# ============================================================================


def test_captured_feature_values_match_source_vector_exactly(tmp_path: Path) -> None:
    vectors = _two_well_separated_groups(n_per_group=6)
    result = _fit_two_clusters(vectors)
    label_store = ContextLabelStore(storage_path=tmp_path / "labels.json")
    examples_store = TrainingExampleStore(storage_path=tmp_path / "examples.json")

    label_map = result.to_label_map()
    run_id = compute_run_id(result)
    label_store.assign_label(run_id, label_map["low-0"], "Focused Work", assigned_at=_ts())
    label_store.assign_label(run_id, label_map["high-0"], "Browsing", assigned_at=_ts())

    capture_labeled_examples(vectors, result, label_store, examples_store, captured_at=_ts())

    source_vector = next(v for v in vectors if v.session_id == "low-0")
    stored = examples_store.get_example("low-0")

    assert stored.feature_values == source_vector.feature_values
    assert stored.feature_names == source_vector.feature_names
    # Provenance is stored, but never merged into the numeric payload.
    assert "run" not in stored.feature_names
    assert len(stored.feature_values) == len(source_vector.feature_names)


def test_metadata_never_enters_feature_values(tmp_path: Path) -> None:
    """A FeatureVector's metadata dict must have zero influence on what gets persisted as features."""
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    vector_with_metadata = FeatureVector(
        session_id="s1",
        feature_names=("f1", "f2"),
        feature_values=(1.0, 2.0),
        metadata={"cluster_label": 999, "run_id": "fake-run", "label": "LeakedLabel"},
    )

    entry = store.record_example(
        vector_with_metadata.session_id,
        vector_with_metadata.feature_names,
        vector_with_metadata.feature_values,
        "RealLabel",
        source_run_id="real-run",
        source_cluster_label=0,
        captured_at=_ts(),
    )

    assert entry.feature_values == (1.0, 2.0)
    assert entry.label == "RealLabel"  # not "LeakedLabel" from metadata


# ============================================================================
# Schema handling
# ============================================================================


def test_default_feature_schema_version_is_current(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")

    entry = store.record_example("s1", ("f1",), (1.0,), "Coding", source_run_id="r", source_cluster_label=0, captured_at=_ts())

    assert entry.feature_schema_version == CURRENT_FEATURE_SCHEMA_VERSION


def test_explicit_feature_schema_version_is_respected(tmp_path: Path) -> None:
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")

    entry = store.record_example(
        "s1", ("f1",), (1.0,), "Coding",
        source_run_id="r", source_cluster_label=0, captured_at=_ts(),
        feature_schema_version=2,
    )

    assert entry.feature_schema_version == 2


def test_feature_schema_version_persists_across_restart(tmp_path: Path) -> None:
    path = tmp_path / "examples.json"
    store_a = TrainingExampleStore(storage_path=path)
    store_a.record_example(
        "s1", ("f1",), (1.0,), "Coding",
        source_run_id="r", source_cluster_label=0, captured_at=_ts(), feature_schema_version=3,
    )

    store_b = TrainingExampleStore(storage_path=path)

    assert store_b.get_example("s1").feature_schema_version == 3


# ============================================================================
# Real Phase 2A -> 2B -> 2C integration
# ============================================================================


class _MockActivity:
    def __init__(self, started_at, ended_at, duration_seconds, application, process_name):
        self.started_at = started_at
        self.ended_at = ended_at
        self.duration_seconds = duration_seconds
        self.application = application
        self.process_name = process_name


def test_end_to_end_with_real_feature_extractor(tmp_path: Path) -> None:
    """Real Phase 2A FeatureExtractor -> real Phase 2B -> real Phase 2C -> capture."""
    from datetime import timedelta

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

    vectors = coding_vectors + browsing_vectors
    result = ContextClusterer(n_clusters=2, random_state=42).fit(vectors)

    label_store = ContextLabelStore(storage_path=tmp_path / "labels.json")
    examples_store = TrainingExampleStore(storage_path=tmp_path / "examples.json")

    label_map = result.to_label_map()
    run_id = compute_run_id(result)
    coding_cluster = label_map["coding-0"]
    browsing_cluster = label_map["browsing-0"]
    label_store.assign_label(run_id, coding_cluster, "Focused Coding", assigned_at=_ts())
    if browsing_cluster != coding_cluster:
        label_store.assign_label(run_id, browsing_cluster, "Evening Browsing", assigned_at=_ts())

    captured = capture_labeled_examples(vectors, result, label_store, examples_store, captured_at=_ts())

    assert len(captured) == len(vectors)
    assert all(e.feature_names == FeatureExtractor.FEATURE_NAMES for e in captured)


# ============================================================================
# REGRESSION GUARDS
#
# These tests do not exercise "normal" behavior -- they exist specifically
# to FAIL if a future change reintroduces a contract violation, even one
# that would otherwise look like it "still passes the tests" behaviorally.
# Several inspect the actual module source rather than just its behavior,
# because a structurally-wrong implementation (e.g. NDJSON, or an
# arbitrary-index delete key) can still superficially satisfy black-box
# CRUD tests while violating the frozen contract.
# ============================================================================

import dataclasses
import inspect
import json


def _training_examples_source() -> str:
    import app.ml.training_examples as module

    return inspect.getsource(module)


def test_guard_labeled_example_has_exact_contract_fields() -> None:
    """Fails if any field is renamed, removed, or reordered relative to the frozen contract."""
    field_names = tuple(f.name for f in dataclasses.fields(LabeledExample))

    assert field_names == (
        "session_id",
        "feature_names",
        "feature_values",
        "label",
        "feature_schema_version",
        "source_run_id",
        "source_cluster_label",
        "created_at",
        "updated_at",
    )


def test_guard_created_at_and_updated_at_are_datetime_not_float() -> None:
    """Fails if timestamps are ever swapped for float/epoch representations."""
    store = TrainingExampleStore(storage_path=__import__("tempfile").mkdtemp() + "/examples.json")
    entry = store.record_example(
        "s1", ("f1",), (1.0,), "Coding", source_run_id="r", source_cluster_label=0, captured_at=_ts()
    )

    assert isinstance(entry.created_at, datetime)
    assert isinstance(entry.updated_at, datetime)
    assert entry.created_at.tzinfo is not None
    assert entry.updated_at.tzinfo is not None
    assert entry.created_at.tzinfo == timezone.utc or entry.created_at.utcoffset().total_seconds() == 0


def test_guard_required_public_methods_exist_with_exact_names() -> None:
    """Fails if record_example (or any other contract method) is ever renamed."""
    for method_name in ("record_example", "get_example", "remove_example", "list_examples", "list_examples_by_label"):
        assert hasattr(TrainingExampleStore, method_name), f"missing required method: {method_name}"
        assert callable(getattr(TrainingExampleStore, method_name))


def test_guard_forbidden_alternative_apis_do_not_exist() -> None:
    """
    Fails if the store is ever refactored to use the non-contract API
    names explicitly forbidden by the frozen spec (from the conflicting
    PDF design or otherwise), e.g. add_example/query/to_arrays or
    made-up convenience names like add_or_update_example/find_by_session.
    """
    forbidden = (
        "add_example", "query", "to_arrays", "clear", "clear_all",
        "add_or_update_example", "find_by_session", "find_by_run",
    )
    for method_name in forbidden:
        assert not hasattr(TrainingExampleStore, method_name), (
            f"forbidden non-contract method exists: {method_name}"
        )


def test_guard_persisted_file_is_single_json_document_not_ndjson(tmp_path: Path) -> None:
    """
    Fails if persistence is ever switched to NDJSON/JSONL. A single valid
    JSON document parses whole-file with json.loads(); NDJSON with more
    than one record does not (multiple top-level values is invalid JSON).
    """
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    store.record_example("s1", ("f1",), (1.0,), "A", source_run_id="r", source_cluster_label=0, captured_at=_ts())
    store.record_example("s2", ("f1",), (2.0,), "B", source_run_id="r", source_cluster_label=1, captured_at=_ts())
    store.record_example("s3", ("f1",), (3.0,), "C", source_run_id="r", source_cluster_label=2, captured_at=_ts())

    raw_text = (tmp_path / "examples.json").read_text(encoding="utf-8")

    # Must parse as ONE JSON value for the whole file (would raise
    # json.JSONDecodeError on "extra data" if this were NDJSON with >1 line).
    payload = json.loads(raw_text)
    assert isinstance(payload, dict)
    assert "examples" in payload
    assert len(payload["examples"]) == 3

    # An NDJSON file would have one JSON object per physical line and
    # NOT be parseable as a single top-level object the way we just did;
    # as a second, independent signal, confirm the file is not simply a
    # sequence of standalone '{...}' lines with no enclosing structure.
    non_blank_lines = [line for line in raw_text.splitlines() if line.strip()]
    single_line_json_objects = sum(
        1 for line in non_blank_lines if line.strip().startswith("{") and line.strip().endswith("}")
    )
    assert single_line_json_objects == 0, "file looks like NDJSON (one bare JSON object per line)"


def test_guard_no_ndjson_indicators_in_source() -> None:
    """Fails if NDJSON/JSONL writing logic (line-oriented JSON) is ever introduced."""
    source = _training_examples_source()
    lowered = source.lower()
    assert "ndjson" not in lowered
    assert "jsonl" not in lowered
    assert ".jsonl" not in lowered


def test_guard_file_extension_is_json_not_jsonl(tmp_path: Path) -> None:
    store = TrainingExampleStore()  # default path, not overridden

    assert store.storage_path.name == "training_examples.json"
    assert store.storage_path.suffix == ".json"


def test_guard_duplicate_session_id_cannot_exist_in_persisted_file(tmp_path: Path) -> None:
    """
    Fails if record_example is ever changed to append rather than upsert,
    which would let the same session_id appear twice in the persisted file.
    """
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    store.record_example("dup", ("f1",), (1.0,), "A", source_run_id="r", source_cluster_label=0, captured_at=_ts(hour=1))
    store.record_example("s2", ("f1",), (2.0,), "B", source_run_id="r", source_cluster_label=1, captured_at=_ts(hour=2))
    store.record_example("dup", ("f1",), (99.0,), "Z", source_run_id="r", source_cluster_label=9, captured_at=_ts(hour=3))

    payload = json.loads((tmp_path / "examples.json").read_text(encoding="utf-8"))
    session_ids_in_file = [record["session_id"] for record in payload["examples"].values()]

    assert session_ids_in_file.count("dup") == 1
    assert sorted(session_ids_in_file) == ["dup", "s2"]
    assert len(store.list_examples()) == 2


def test_guard_feature_names_and_feature_values_survive_full_round_trip(tmp_path: Path) -> None:
    """
    Fails if feature_names/feature_values are ever dropped from the
    dataclass, the persistence layer, or silently truncated/reordered.
    """
    path = tmp_path / "examples.json"
    store_a = TrainingExampleStore(storage_path=path)
    original_names = ("session_duration_minutes", "app_diversity_entropy", "is_business_hours")
    original_values = (42.5, 0.73, 1.0)
    store_a.record_example(
        "s1", original_names, original_values, "Coding",
        source_run_id="r", source_cluster_label=0, captured_at=_ts(),
    )

    # Force a full reload from disk (new object -- proves it survived serialization).
    store_b = TrainingExampleStore(storage_path=path)
    reloaded = store_b.get_example("s1")

    assert reloaded.feature_names == original_names
    assert reloaded.feature_values == original_values
    assert len(reloaded.feature_names) == len(reloaded.feature_values) == 3


def test_guard_provenance_fields_never_appear_inside_feature_values(tmp_path: Path) -> None:
    """
    Fails if source_run_id/source_cluster_label/label/session_id/timestamps
    are ever smuggled into feature_values as extra numeric columns.
    """
    store = TrainingExampleStore(storage_path=tmp_path / "examples.json")
    entry = store.record_example(
        "s1", ("f1", "f2"), (1.0, 2.0), "Coding",
        source_run_id="run-xyz", source_cluster_label=7, captured_at=_ts(),
    )

    # Exactly as many values as declared feature names -- no hidden extras.
    assert len(entry.feature_values) == len(entry.feature_names) == 2
    assert entry.feature_values == (1.0, 2.0)
    # The numeric payload contains none of the provenance/label content.
    assert 7.0 not in entry.feature_values
    assert entry.source_cluster_label not in entry.feature_values


def test_guard_capture_labeled_examples_exact_signature_shape() -> None:
    """Fails if capture_labeled_examples's parameter shape ever drifts from the frozen contract."""
    signature = inspect.signature(capture_labeled_examples)
    parameter_names = list(signature.parameters.keys())

    assert parameter_names == ["feature_vectors", "result", "label_store", "examples_store", "captured_at", "run_id"]
    assert signature.parameters["captured_at"].kind == inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["run_id"].kind == inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["run_id"].default is None


def test_guard_run_id_is_computed_via_actual_compute_run_id_when_omitted(tmp_path: Path) -> None:
    """
    Fails if run_id resolution is ever reimplemented instead of delegating
    to the real Phase 2C compute_run_id -- captured examples'
    source_run_id must equal what compute_run_id actually returns for
    that exact ClusteringResult.
    """
    vectors = _two_well_separated_groups(n_per_group=6)
    result = _fit_two_clusters(vectors)
    label_store = ContextLabelStore(storage_path=tmp_path / "labels.json")
    examples_store = TrainingExampleStore(storage_path=tmp_path / "examples.json")

    label_map = result.to_label_map()
    expected_run_id = compute_run_id(result)
    label_store.assign_label(expected_run_id, label_map["low-0"], "Focused Work", assigned_at=_ts())
    label_store.assign_label(expected_run_id, label_map["high-0"], "Browsing", assigned_at=_ts())

    captured = capture_labeled_examples(vectors, result, label_store, examples_store, captured_at=_ts())  # run_id omitted

    assert all(example.source_run_id == expected_run_id for example in captured)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
