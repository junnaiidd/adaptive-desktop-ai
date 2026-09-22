"""
Comprehensive tests for Phase 2C: Human Interpretation and Persistent
Context Labeling.

Tests cover:
- Creating a cluster interpretation (summarize_cluster / summarize_all_clusters)
- Labeling a cluster
- Retrieving labels
- Updating labels
- Removing labels
- Multiple cluster labels
- Unlabeled clusters
- Arbitrary user-defined labels (no hardcoded vocabulary)
- Persistence across object/module reload
- Invalid / empty labels
- Unknown cluster IDs
- Deterministic serialization / persistence
- Cluster identity / stability behavior (run_id fingerprinting)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.ml.context_clustering import ContextClusterer
from app.ml.context_labeling import (
    ClusterSummary,
    ContextLabelStore,
    LabeledCluster,
    compute_run_id,
    session_id_to_label,
    summarize_all_clusters,
    summarize_cluster,
)
from app.ml.feature_engineering import FeatureVector


# ============================================================================
# Test helpers
# ============================================================================

FEATURE_NAMES_2D = ("f1", "f2")


def _vector(session_id: str, values: tuple[float, ...], names: tuple[str, ...] = FEATURE_NAMES_2D) -> FeatureVector:
    return FeatureVector(session_id=session_id, feature_names=names, feature_values=values, metadata={})


def _two_well_separated_groups(n_per_group: int = 6) -> list[FeatureVector]:
    vectors = []
    for i in range(n_per_group):
        vectors.append(_vector(f"low-{i}", (0.0 + i * 0.01, 0.0 + i * 0.01)))
    for i in range(n_per_group):
        vectors.append(_vector(f"high-{i}", (50.0 + i * 0.01, 50.0 + i * 0.01)))
    return vectors


def _three_well_separated_groups(n_per_group: int = 5) -> list[FeatureVector]:
    vectors = []
    centers = [(0.0, 0.0), (30.0, 0.0), (0.0, 30.0)]
    for group_index, (cx, cy) in enumerate(centers):
        for i in range(n_per_group):
            vectors.append(_vector(f"g{group_index}-{i}", (cx + i * 0.01, cy + i * 0.01)))
    return vectors


def _fit_two_clusters():
    vectors = _two_well_separated_groups()
    return ContextClusterer(n_clusters=2, random_state=42).fit(vectors)


def _fit_three_clusters():
    vectors = _three_well_separated_groups()
    return ContextClusterer(n_clusters=3, random_state=42).fit(vectors)


def _ts(hour: int = 12, day: int = 1) -> datetime:
    return datetime(2026, 1, day, hour=hour, tzinfo=timezone.utc)


# ============================================================================
# Creating a cluster interpretation
# ============================================================================


def test_summarize_cluster_returns_expected_shape() -> None:
    result = _fit_two_clusters()

    summary = summarize_cluster(result, cluster_label=0)

    assert isinstance(summary, ClusterSummary)
    assert summary.cluster_label == 0
    assert summary.n_sessions == result.cluster_sizes[0]
    assert set(summary.feature_center.keys()) == set(FEATURE_NAMES_2D)
    assert 0.0 <= summary.cluster_share <= 1.0
    assert summary.silhouette == result.silhouette
    assert summary.silhouette_note == result.silhouette_note
    assert summary.run_id  # non-empty


def test_summarize_cluster_share_is_fraction_of_total_sessions() -> None:
    result = _fit_two_clusters()  # 12 sessions, 2 clusters of 6 each

    summary = summarize_cluster(result, cluster_label=0)

    assert summary.n_sessions == 6
    assert summary.cluster_share == pytest.approx(6 / 12)


def test_summarize_cluster_reuses_result_feature_center() -> None:
    """The summary's feature_center must match Phase 2B's own center_as_dict output exactly."""
    result = _fit_two_clusters()

    summary = summarize_cluster(result, cluster_label=1)

    assert summary.feature_center == result.center_as_dict(1)


def test_summarize_cluster_out_of_range_raises() -> None:
    result = _fit_two_clusters()

    with pytest.raises(ValueError, match="out of range"):
        summarize_cluster(result, cluster_label=99)


def test_summarize_all_clusters_returns_one_per_cluster() -> None:
    result = _fit_three_clusters()

    summaries = summarize_all_clusters(result)

    assert len(summaries) == 3
    assert [s.cluster_label for s in summaries] == [0, 1, 2]
    assert len({s.run_id for s in summaries}) == 1  # all share one run_id


def test_summarize_all_clusters_sessions_sum_to_total() -> None:
    result = _fit_three_clusters()  # 15 sessions total

    summaries = summarize_all_clusters(result)

    assert sum(s.n_sessions for s in summaries) == result.n_samples


# ============================================================================
# Labeling a cluster
# ============================================================================


def test_assign_label_stores_and_returns_entry(tmp_path: Path) -> None:
    store = ContextLabelStore(storage_path=tmp_path / "labels.json")

    entry = store.assign_label("run-abc", 0, "Coding", assigned_at=_ts())

    assert isinstance(entry, LabeledCluster)
    assert entry.run_id == "run-abc"
    assert entry.cluster_label == 0
    assert entry.label == "Coding"
    assert entry.created_at == _ts()
    assert entry.updated_at == _ts()


def test_assign_label_strips_surrounding_whitespace(tmp_path: Path) -> None:
    store = ContextLabelStore(storage_path=tmp_path / "labels.json")

    entry = store.assign_label("run-abc", 0, "  Deep Work  ", assigned_at=_ts())

    assert entry.label == "Deep Work"


# ============================================================================
# Retrieving labels
# ============================================================================


def test_get_label_returns_stored_entry(tmp_path: Path) -> None:
    store = ContextLabelStore(storage_path=tmp_path / "labels.json")
    store.assign_label("run-abc", 2, "Research", assigned_at=_ts())

    entry = store.get_label("run-abc", 2)

    assert entry is not None
    assert entry.label == "Research"


def test_get_label_for_unknown_cluster_returns_none(tmp_path: Path) -> None:
    store = ContextLabelStore(storage_path=tmp_path / "labels.json")

    assert store.get_label("run-abc", 0) is None


def test_get_label_distinguishes_between_runs(tmp_path: Path) -> None:
    """Same cluster_label integer under two different run_ids must be independent."""
    store = ContextLabelStore(storage_path=tmp_path / "labels.json")
    store.assign_label("run-A", 0, "Coding", assigned_at=_ts())

    assert store.get_label("run-B", 0) is None
    assert store.get_label("run-A", 0).label == "Coding"


# ============================================================================
# Updating labels
# ============================================================================


def test_update_label_overwrites_text(tmp_path: Path) -> None:
    store = ContextLabelStore(storage_path=tmp_path / "labels.json")
    store.assign_label("run-abc", 0, "Coding", assigned_at=_ts(hour=9))

    updated = store.assign_label("run-abc", 0, "Focused Coding", assigned_at=_ts(hour=10))

    assert updated.label == "Focused Coding"
    assert store.get_label("run-abc", 0).label == "Focused Coding"


def test_update_label_preserves_original_created_at(tmp_path: Path) -> None:
    store = ContextLabelStore(storage_path=tmp_path / "labels.json")
    first = store.assign_label("run-abc", 0, "Coding", assigned_at=_ts(hour=9))

    second = store.assign_label("run-abc", 0, "Focused Coding", assigned_at=_ts(hour=15))

    assert second.created_at == first.created_at == _ts(hour=9)
    assert second.updated_at == _ts(hour=15)
    assert second.updated_at != first.updated_at


# ============================================================================
# Removing labels
# ============================================================================


def test_remove_label_deletes_existing_entry(tmp_path: Path) -> None:
    store = ContextLabelStore(storage_path=tmp_path / "labels.json")
    store.assign_label("run-abc", 0, "Coding", assigned_at=_ts())

    removed = store.remove_label("run-abc", 0)

    assert removed is True
    assert store.get_label("run-abc", 0) is None


def test_remove_label_on_unlabeled_cluster_returns_false(tmp_path: Path) -> None:
    store = ContextLabelStore(storage_path=tmp_path / "labels.json")

    removed = store.remove_label("run-abc", 0)

    assert removed is False


def test_remove_label_does_not_affect_other_clusters(tmp_path: Path) -> None:
    store = ContextLabelStore(storage_path=tmp_path / "labels.json")
    store.assign_label("run-abc", 0, "Coding", assigned_at=_ts())
    store.assign_label("run-abc", 1, "Browsing", assigned_at=_ts())

    store.remove_label("run-abc", 0)

    assert store.get_label("run-abc", 0) is None
    assert store.get_label("run-abc", 1).label == "Browsing"


# ============================================================================
# Multiple cluster labels
# ============================================================================


def test_list_labels_returns_all_entries_sorted(tmp_path: Path) -> None:
    store = ContextLabelStore(storage_path=tmp_path / "labels.json")
    store.assign_label("run-B", 1, "Gaming", assigned_at=_ts())
    store.assign_label("run-A", 0, "Coding", assigned_at=_ts())
    store.assign_label("run-A", 2, "Reading", assigned_at=_ts())

    labels = store.list_labels()

    assert [(entry.run_id, entry.cluster_label) for entry in labels] == [
        ("run-A", 0),
        ("run-A", 2),
        ("run-B", 1),
    ]


def test_list_labels_for_run_filters_to_one_run(tmp_path: Path) -> None:
    store = ContextLabelStore(storage_path=tmp_path / "labels.json")
    store.assign_label("run-A", 0, "Coding", assigned_at=_ts())
    store.assign_label("run-A", 1, "Browsing", assigned_at=_ts())
    store.assign_label("run-B", 0, "Gaming", assigned_at=_ts())

    run_a_labels = store.list_labels_for_run("run-A")

    assert {entry.cluster_label for entry in run_a_labels} == {0, 1}
    assert all(entry.run_id == "run-A" for entry in run_a_labels)


def test_list_labels_empty_store_returns_empty_tuple(tmp_path: Path) -> None:
    store = ContextLabelStore(storage_path=tmp_path / "labels.json")

    assert store.list_labels() == ()
    assert store.list_labels_for_run("anything") == ()


# ============================================================================
# Unlabeled clusters (bridging helper)
# ============================================================================


def test_session_id_to_label_all_unlabeled_returns_none_for_every_session(tmp_path: Path) -> None:
    result = _fit_two_clusters()
    store = ContextLabelStore(storage_path=tmp_path / "labels.json")

    mapping = session_id_to_label(result, store)

    assert set(mapping.keys()) == {a.session_id for a in result.assignments}
    assert all(value is None for value in mapping.values())


def test_session_id_to_label_reflects_partial_labeling(tmp_path: Path) -> None:
    result = _fit_two_clusters()
    store = ContextLabelStore(storage_path=tmp_path / "labels.json")
    run_id = compute_run_id(result)
    store.assign_label(run_id, 0, "Focused Work", assigned_at=_ts())
    # Cluster 1 intentionally left unlabeled.

    mapping = session_id_to_label(result, store, run_id=run_id)

    labels_seen = result.to_label_map()
    for session_id, cluster_label in labels_seen.items():
        if cluster_label == 0:
            assert mapping[session_id] == "Focused Work"
        else:
            assert mapping[session_id] is None


# ============================================================================
# Arbitrary user-defined labels (no hardcoded vocabulary)
# ============================================================================


@pytest.mark.parametrize(
    "label_text",
    [
        "Coding",
        "Deep Work / Flow State",
        "misc-cluster-47",
        "Client project: Acme Corp",
        "🎮 Gaming night",
        "unlabeled-but-interesting",
        "a" * 200,  # long but non-empty label is accepted, no hardcoded max
    ],
)
def test_assign_label_accepts_arbitrary_text(tmp_path: Path, label_text: str) -> None:
    store = ContextLabelStore(storage_path=tmp_path / "labels.json")

    entry = store.assign_label("run-abc", 0, label_text, assigned_at=_ts())

    assert entry.label == label_text.strip()


# ============================================================================
# Persistence across object/module reload
# ============================================================================


def test_labels_persist_across_new_store_instance_same_path(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    store_a = ContextLabelStore(storage_path=path)
    store_a.assign_label("run-abc", 0, "Coding", assigned_at=_ts())

    # Simulate a fresh process/module reload: brand-new object, same file.
    store_b = ContextLabelStore(storage_path=path)

    entry = store_b.get_label("run-abc", 0)
    assert entry is not None
    assert entry.label == "Coding"


def test_multiple_labels_persist_across_reload(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    store_a = ContextLabelStore(storage_path=path)
    store_a.assign_label("run-abc", 0, "Coding", assigned_at=_ts())
    store_a.assign_label("run-abc", 1, "Browsing", assigned_at=_ts())
    store_a.assign_label("run-xyz", 0, "Gaming", assigned_at=_ts())

    store_b = ContextLabelStore(storage_path=path)

    assert len(store_b.list_labels()) == 3
    assert store_b.get_label("run-xyz", 0).label == "Gaming"


def test_removal_persists_across_reload(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    store_a = ContextLabelStore(storage_path=path)
    store_a.assign_label("run-abc", 0, "Coding", assigned_at=_ts())
    store_a.remove_label("run-abc", 0)

    store_b = ContextLabelStore(storage_path=path)

    assert store_b.get_label("run-abc", 0) is None
    assert store_b.list_labels() == ()


def test_store_creates_file_on_first_use(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "labels.json"
    assert not path.exists()

    ContextLabelStore(storage_path=path)

    assert path.exists()


# ============================================================================
# Invalid / empty labels
# ============================================================================


def test_assign_label_rejects_empty_string(tmp_path: Path) -> None:
    store = ContextLabelStore(storage_path=tmp_path / "labels.json")

    with pytest.raises(ValueError, match="empty"):
        store.assign_label("run-abc", 0, "", assigned_at=_ts())


def test_assign_label_rejects_whitespace_only_string(tmp_path: Path) -> None:
    store = ContextLabelStore(storage_path=tmp_path / "labels.json")

    with pytest.raises(ValueError, match="empty"):
        store.assign_label("run-abc", 0, "   \t  ", assigned_at=_ts())


def test_assign_label_rejects_non_string_label(tmp_path: Path) -> None:
    store = ContextLabelStore(storage_path=tmp_path / "labels.json")

    with pytest.raises(ValueError, match="string"):
        store.assign_label("run-abc", 0, 12345, assigned_at=_ts())  # type: ignore[arg-type]


def test_assign_label_rejects_empty_run_id(tmp_path: Path) -> None:
    store = ContextLabelStore(storage_path=tmp_path / "labels.json")

    with pytest.raises(ValueError, match="run_id"):
        store.assign_label("", 0, "Coding", assigned_at=_ts())


def test_assign_label_rejects_negative_cluster_label(tmp_path: Path) -> None:
    store = ContextLabelStore(storage_path=tmp_path / "labels.json")

    with pytest.raises(ValueError, match="cluster_label"):
        store.assign_label("run-abc", -1, "Coding", assigned_at=_ts())


def test_assign_label_rejects_timezone_naive_timestamp(tmp_path: Path) -> None:
    store = ContextLabelStore(storage_path=tmp_path / "labels.json")
    naive = datetime(2026, 1, 1, 12, 0, 0)  # no tzinfo

    with pytest.raises(ValueError, match="timezone-aware"):
        store.assign_label("run-abc", 0, "Coding", assigned_at=naive)


def test_nothing_persisted_after_rejected_assignment(tmp_path: Path) -> None:
    """A failed assign_label call must not corrupt or partially write the store."""
    store = ContextLabelStore(storage_path=tmp_path / "labels.json")

    with pytest.raises(ValueError):
        store.assign_label("run-abc", 0, "", assigned_at=_ts())

    assert store.list_labels() == ()


# ============================================================================
# Unknown cluster IDs
# ============================================================================


def test_get_label_unknown_run_and_cluster_returns_none(tmp_path: Path) -> None:
    store = ContextLabelStore(storage_path=tmp_path / "labels.json")
    store.assign_label("run-abc", 0, "Coding", assigned_at=_ts())

    assert store.get_label("run-nonexistent", 0) is None
    assert store.get_label("run-abc", 999) is None


def test_remove_label_unknown_run_returns_false(tmp_path: Path) -> None:
    store = ContextLabelStore(storage_path=tmp_path / "labels.json")
    store.assign_label("run-abc", 0, "Coding", assigned_at=_ts())

    assert store.remove_label("run-does-not-exist", 0) is False


# ============================================================================
# Deterministic serialization / persistence
# ============================================================================


def test_repeated_writes_of_same_content_produce_identical_bytes(tmp_path: Path) -> None:
    path_a = tmp_path / "labels_a.json"
    path_b = tmp_path / "labels_b.json"
    store_a = ContextLabelStore(storage_path=path_a)
    store_b = ContextLabelStore(storage_path=path_b)

    store_a.assign_label("run-abc", 0, "Coding", assigned_at=_ts())
    store_a.assign_label("run-abc", 1, "Browsing", assigned_at=_ts(hour=13))
    store_b.assign_label("run-abc", 0, "Coding", assigned_at=_ts())
    store_b.assign_label("run-abc", 1, "Browsing", assigned_at=_ts(hour=13))

    assert path_a.read_text(encoding="utf-8") == path_b.read_text(encoding="utf-8")


def test_run_id_is_deterministic_for_identical_clustering_content() -> None:
    """Re-fitting identical data with an identical configuration must yield the same run_id."""
    vectors = _two_well_separated_groups()
    result_1 = ContextClusterer(n_clusters=2, random_state=42).fit(vectors)
    result_2 = ContextClusterer(n_clusters=2, random_state=42).fit(vectors)

    assert compute_run_id(result_1) == compute_run_id(result_2)


def test_compute_run_id_is_pure_and_repeatable() -> None:
    result = _fit_two_clusters()

    assert compute_run_id(result) == compute_run_id(result)


# ============================================================================
# Cluster identity / stability behavior
# ============================================================================


def test_run_id_differs_for_different_k() -> None:
    vectors = _three_well_separated_groups()
    result_k2 = ContextClusterer(n_clusters=2, random_state=42).fit(vectors)
    result_k3 = ContextClusterer(n_clusters=3, random_state=42).fit(vectors)

    assert compute_run_id(result_k2) != compute_run_id(result_k3)


def test_run_id_differs_for_different_random_state() -> None:
    vectors = _three_well_separated_groups()
    result_seed_1 = ContextClusterer(n_clusters=3, random_state=1).fit(vectors)
    result_seed_99 = ContextClusterer(n_clusters=3, random_state=99).fit(vectors)

    # Different seeds are not guaranteed to differ in every case, but with
    # this well-separated synthetic dataset the fingerprint components
    # (random_state itself, and typically cluster ordering) do differ.
    assert result_seed_1.random_state != result_seed_99.random_state
    assert compute_run_id(result_seed_1) != compute_run_id(result_seed_99)


def test_run_id_differs_when_underlying_data_changes() -> None:
    vectors_a = _two_well_separated_groups(n_per_group=6)
    vectors_b = _two_well_separated_groups(n_per_group=8)  # more sessions -> different centers/n_samples
    result_a = ContextClusterer(n_clusters=2, random_state=42).fit(vectors_a)
    result_b = ContextClusterer(n_clusters=2, random_state=42).fit(vectors_b)

    assert compute_run_id(result_a) != compute_run_id(result_b)


def test_label_does_not_silently_carry_over_to_a_new_run(tmp_path: Path) -> None:
    """
    Simulates retraining: a label assigned under one run's cluster 0 must
    NOT be visible under a differently-fingerprinted run's cluster 0, even
    though the raw integer cluster label (0) is identical in both cases.
    """
    store = ContextLabelStore(storage_path=tmp_path / "labels.json")

    vectors_before = _two_well_separated_groups(n_per_group=6)
    result_before = ContextClusterer(n_clusters=2, random_state=42).fit(vectors_before)
    run_id_before = compute_run_id(result_before)
    store.assign_label(run_id_before, 0, "Coding", assigned_at=_ts())

    # "Retrain" with slightly different data -- same K, same random_state,
    # but the discovered cluster centers (and therefore the run_id) shift.
    vectors_after = _two_well_separated_groups(n_per_group=9)
    result_after = ContextClusterer(n_clusters=2, random_state=42).fit(vectors_after)
    run_id_after = compute_run_id(result_after)

    assert run_id_after != run_id_before
    assert store.get_label(run_id_after, 0) is None
    # The original label is still safely there under its own run_id.
    assert store.get_label(run_id_before, 0).label == "Coding"


def test_labeled_cluster_key_pair_is_the_true_identity_not_bare_integer(tmp_path: Path) -> None:
    """Two different runs may reuse cluster_label=0 independently without collision."""
    store = ContextLabelStore(storage_path=tmp_path / "labels.json")
    store.assign_label("run-A", 0, "Coding", assigned_at=_ts())
    store.assign_label("run-B", 0, "Gaming", assigned_at=_ts())

    assert store.get_label("run-A", 0).label == "Coding"
    assert store.get_label("run-B", 0).label == "Gaming"
    assert len(store.list_labels()) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
