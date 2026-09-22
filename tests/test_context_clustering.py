"""
Comprehensive tests for Phase 2B: Unsupervised Behavioral Context Discovery

Tests cover:
- Valid clustering with well-separated synthetic groups
- Deterministic output with fixed random_state
- Configurable cluster count (K)
- Session-ID preservation through the clustering pipeline
- Cluster-center shape and interpretability
- Silhouette score computation on valid datasets
- Invalid / empty input handling
- Datasets too small for the requested K
- Degenerate / identical feature vectors
- Multiple-K evaluation (evaluate_k_range)
- Feature dimension mismatch across vectors and raw matrices
- Real Phase 2A integration (FeatureExtractor -> ContextClusterer)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.ml.context_clustering import (
    ClusteringResult,
    ContextClusterer,
    KRangeEvaluationResult,
    MIN_SAMPLES_FOR_CLUSTERING,
    evaluate_k_range,
)
from app.ml.feature_engineering import FeatureExtractor, FeatureVector


# ============================================================================
# Test helpers
# ============================================================================

FEATURE_NAMES_2D = ("f1", "f2")


def _vector(session_id: str, values: tuple[float, ...], names: tuple[str, ...] = FEATURE_NAMES_2D) -> FeatureVector:
    return FeatureVector(session_id=session_id, feature_names=names, feature_values=values, metadata={})


def _two_well_separated_groups(n_per_group: int = 6) -> list[FeatureVector]:
    """Build a synthetic dataset with two obviously distinct clusters."""
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


def _identical_vectors(n: int = 5, values: tuple[float, ...] = (1.0, 1.0)) -> list[FeatureVector]:
    return [_vector(f"same-{i}", values) for i in range(n)]


def _timestamp(seconds: int = 0, hour: int = 12) -> datetime:
    return datetime(2026, 1, 1, hour=hour, tzinfo=timezone.utc) + timedelta(seconds=seconds)


class _MockActivity:
    """Minimal stand-in matching the attributes FeatureExtractor expects."""

    def __init__(self, started_at, ended_at, duration_seconds, application, process_name):
        self.started_at = started_at
        self.ended_at = ended_at
        self.duration_seconds = duration_seconds
        self.application = application
        self.process_name = process_name


# ============================================================================
# Valid clustering
# ============================================================================


def test_valid_clustering_separates_two_obvious_groups() -> None:
    """K-Means should cleanly separate two well-separated synthetic groups."""
    vectors = _two_well_separated_groups()
    clusterer = ContextClusterer(n_clusters=2, random_state=42)

    result = clusterer.fit(vectors)

    labels = result.to_label_map()
    low_labels = {labels[f"low-{i}"] for i in range(6)}
    high_labels = {labels[f"high-{i}"] for i in range(6)}

    assert len(low_labels) == 1  # all "low" sessions share one cluster
    assert len(high_labels) == 1  # all "high" sessions share one cluster
    assert low_labels != high_labels  # the two groups are in different clusters


def test_valid_clustering_returns_full_result_shape() -> None:
    vectors = _two_well_separated_groups()
    clusterer = ContextClusterer(n_clusters=2, random_state=42)

    result = clusterer.fit(vectors)

    assert isinstance(result, ClusteringResult)
    assert result.n_clusters == 2
    assert result.n_samples == len(vectors)
    assert len(result.assignments) == len(vectors)
    assert result.feature_names == FEATURE_NAMES_2D


# ============================================================================
# Deterministic output
# ============================================================================


def test_deterministic_output_with_fixed_random_state() -> None:
    """Running the same clustering twice with the same random_state must match exactly."""
    vectors = _three_well_separated_groups()
    clusterer_a = ContextClusterer(n_clusters=3, random_state=7)
    clusterer_b = ContextClusterer(n_clusters=3, random_state=7)

    result_a = clusterer_a.fit(vectors)
    result_b = clusterer_b.fit(vectors)

    assert result_a.to_label_map() == result_b.to_label_map()
    assert result_a.cluster_centers == result_b.cluster_centers
    assert result_a.inertia == result_b.inertia
    assert result_a.silhouette == result_b.silhouette


def test_different_random_state_still_produces_valid_result() -> None:
    """Different seeds are allowed to produce different (but still valid) partitions."""
    vectors = _three_well_separated_groups()
    result_seed_1 = ContextClusterer(n_clusters=3, random_state=1).fit(vectors)
    result_seed_2 = ContextClusterer(n_clusters=3, random_state=99).fit(vectors)

    # Both must still be legitimate 3-cluster results, even if labels differ.
    assert result_seed_1.n_clusters == 3
    assert result_seed_2.n_clusters == 3
    assert len(set(result_seed_1.to_label_map().values())) <= 3
    assert len(set(result_seed_2.to_label_map().values())) <= 3


# ============================================================================
# Configurable K
# ============================================================================


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_configurable_k_is_honored(k: int) -> None:
    vectors = _three_well_separated_groups(n_per_group=5)  # 15 sessions total
    clusterer = ContextClusterer(n_clusters=k, random_state=42)

    result = clusterer.fit(vectors)

    assert result.n_clusters == k
    assert len(result.cluster_centers) == k
    assert set(result.cluster_sizes.keys()) == set(range(k))


def test_k_equal_to_n_samples_is_allowed() -> None:
    """Each session may become its own cluster when K == n_samples."""
    vectors = _two_well_separated_groups(n_per_group=3)  # 6 sessions
    clusterer = ContextClusterer(n_clusters=6, random_state=42)

    result = clusterer.fit(vectors)

    assert result.n_clusters == 6
    assert result.n_samples == 6


# ============================================================================
# Session-ID preservation
# ============================================================================


def test_session_ids_are_preserved_and_complete() -> None:
    vectors = _two_well_separated_groups()
    expected_ids = {v.session_id for v in vectors}
    clusterer = ContextClusterer(n_clusters=2, random_state=42)

    result = clusterer.fit(vectors)

    assigned_ids = {assignment.session_id for assignment in result.assignments}
    assert assigned_ids == expected_ids


def test_session_ids_map_one_to_one_no_duplicates_no_loss() -> None:
    vectors = _three_well_separated_groups()
    clusterer = ContextClusterer(n_clusters=3, random_state=42)

    result = clusterer.fit(vectors)

    session_ids_in_result = [a.session_id for a in result.assignments]
    assert len(session_ids_in_result) == len(set(session_ids_in_result))
    assert len(session_ids_in_result) == len(vectors)


# ============================================================================
# Cluster-center shape
# ============================================================================


def test_cluster_center_shape_matches_feature_dimensionality() -> None:
    vectors = _two_well_separated_groups()
    clusterer = ContextClusterer(n_clusters=2, random_state=42)

    result = clusterer.fit(vectors)

    assert len(result.cluster_centers) == 2
    for center in result.cluster_centers:
        assert len(center) == len(FEATURE_NAMES_2D)


def test_cluster_centers_are_in_original_feature_units() -> None:
    """Centers should be inverse-transformed back to original (unscaled) units."""
    vectors = _two_well_separated_groups(n_per_group=6)
    clusterer = ContextClusterer(n_clusters=2, random_state=42)

    result = clusterer.fit(vectors)

    # One center should be near (0, 0), the other near (50, 50) -- NOT
    # near 0 in standardized space.
    centers_x = sorted(center[0] for center in result.cluster_centers)
    assert centers_x[0] == pytest.approx(0.0, abs=1.0)
    assert centers_x[1] == pytest.approx(50.0, abs=1.0)


def test_center_as_dict_returns_named_features() -> None:
    vectors = _two_well_separated_groups()
    clusterer = ContextClusterer(n_clusters=2, random_state=42)
    result = clusterer.fit(vectors)

    center_dict = result.center_as_dict(0)

    assert set(center_dict.keys()) == set(FEATURE_NAMES_2D)


def test_center_as_dict_rejects_out_of_range_label() -> None:
    vectors = _two_well_separated_groups()
    result = ContextClusterer(n_clusters=2, random_state=42).fit(vectors)

    with pytest.raises(ValueError, match="out of range"):
        result.center_as_dict(99)


# ============================================================================
# Silhouette score on valid datasets
# ============================================================================


def test_silhouette_score_is_high_for_well_separated_clusters() -> None:
    vectors = _two_well_separated_groups()
    clusterer = ContextClusterer(n_clusters=2, random_state=42)

    result = clusterer.fit(vectors)

    assert result.silhouette is not None
    assert -1.0 <= result.silhouette <= 1.0
    assert result.silhouette > 0.8  # near-perfect separation expected
    assert "successfully" in result.silhouette_note.lower()


def test_silhouette_score_is_none_for_single_cluster() -> None:
    """Silhouette is mathematically undefined for n_clusters=1."""
    vectors = _two_well_separated_groups()
    clusterer = ContextClusterer(n_clusters=1, random_state=42)

    result = clusterer.fit(vectors)

    assert result.silhouette is None
    assert "undefined" in result.silhouette_note.lower()


def test_silhouette_score_is_none_when_k_equals_n_samples() -> None:
    """Silhouette requires n_clusters <= n_samples - 1."""
    vectors = _two_well_separated_groups(n_per_group=3)  # 6 sessions
    clusterer = ContextClusterer(n_clusters=6, random_state=42)

    result = clusterer.fit(vectors)

    assert result.silhouette is None
    assert "undefined" in result.silhouette_note.lower()


# ============================================================================
# Invalid / empty input
# ============================================================================


def test_empty_feature_vector_list_raises() -> None:
    clusterer = ContextClusterer(n_clusters=2, random_state=42)

    with pytest.raises(ValueError, match="Insufficient data"):
        clusterer.fit([])


def test_empty_feature_matrix_raises() -> None:
    clusterer = ContextClusterer(n_clusters=2, random_state=42)

    with pytest.raises(ValueError, match="Insufficient data"):
        clusterer.fit_matrix([], [])


def test_zero_n_clusters_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="n_clusters"):
        ContextClusterer(n_clusters=0)


def test_negative_n_clusters_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="n_clusters"):
        ContextClusterer(n_clusters=-1)


def test_non_finite_values_in_matrix_raise() -> None:
    clusterer = ContextClusterer(n_clusters=1, random_state=42)

    with pytest.raises(ValueError, match="NaN|infinite"):
        clusterer.fit_matrix([[1.0, float("nan")], [2.0, 3.0]], ["a", "b"])


def test_mismatched_session_ids_length_raises() -> None:
    clusterer = ContextClusterer(n_clusters=1, random_state=42)

    with pytest.raises(ValueError, match="session_ids"):
        clusterer.fit_matrix([[1.0, 2.0], [3.0, 4.0]], ["only-one-id"])


def test_duplicate_session_ids_raise() -> None:
    clusterer = ContextClusterer(n_clusters=1, random_state=42)

    with pytest.raises(ValueError, match="unique"):
        clusterer.fit_matrix([[1.0, 2.0], [3.0, 4.0]], ["dup", "dup"])


# ============================================================================
# Dataset too small for requested K
# ============================================================================


def test_dataset_smaller_than_min_samples_raises() -> None:
    """A single session can never be clustered."""
    vectors = [_vector("only-one", (1.0, 2.0))]
    clusterer = ContextClusterer(n_clusters=1, random_state=42)

    with pytest.raises(ValueError, match="Insufficient data"):
        clusterer.fit(vectors)

    assert MIN_SAMPLES_FOR_CLUSTERING == 2


def test_k_greater_than_available_sessions_raises() -> None:
    vectors = _two_well_separated_groups(n_per_group=1)  # 2 sessions total
    clusterer = ContextClusterer(n_clusters=5, random_state=42)

    with pytest.raises(ValueError, match="Insufficient data"):
        clusterer.fit(vectors)


def test_error_message_for_small_dataset_is_actionable() -> None:
    vectors = _two_well_separated_groups(n_per_group=1)  # 2 sessions
    clusterer = ContextClusterer(n_clusters=10, random_state=42)

    with pytest.raises(ValueError) as exc_info:
        clusterer.fit(vectors)

    message = str(exc_info.value)
    assert "n_clusters=10" in message
    assert "2 session" in message


# ============================================================================
# Degenerate / identical feature vectors
# ============================================================================


def test_identical_feature_vectors_do_not_crash() -> None:
    """K-Means must handle all-identical input gracefully rather than erroring."""
    vectors = _identical_vectors(n=5, values=(3.0, 3.0))
    clusterer = ContextClusterer(n_clusters=3, random_state=42)

    result = clusterer.fit(vectors)

    assert result.n_samples == 5
    assert len(result.assignments) == 5


def test_identical_feature_vectors_have_undefined_silhouette() -> None:
    """Identical points collapse to a single effective cluster; silhouette is undefined."""
    vectors = _identical_vectors(n=6, values=(2.0, 2.0))
    clusterer = ContextClusterer(n_clusters=3, random_state=42)

    result = clusterer.fit(vectors)

    assert result.silhouette is None
    assert "could not be computed" in result.silhouette_note.lower()


def test_identical_feature_vectors_all_assigned_same_cluster() -> None:
    vectors = _identical_vectors(n=4, values=(7.0, 7.0))
    clusterer = ContextClusterer(n_clusters=4, random_state=42)

    result = clusterer.fit(vectors)

    labels = set(result.to_label_map().values())
    assert len(labels) == 1  # K-Means cannot separate identical points


def test_near_identical_vectors_with_tiny_noise() -> None:
    """Near-duplicate (not perfectly identical) points should still cluster without error."""
    vectors = [_vector(f"s{i}", (1.0 + i * 1e-9, 1.0 + i * 1e-9)) for i in range(5)]
    clusterer = ContextClusterer(n_clusters=2, random_state=42)

    result = clusterer.fit(vectors)

    assert result.n_samples == 5


# ============================================================================
# Multiple K evaluation
# ============================================================================


def test_evaluate_k_range_returns_one_entry_per_k() -> None:
    vectors = _three_well_separated_groups(n_per_group=5)  # 15 sessions

    evaluation = evaluate_k_range(vectors, k_min=2, k_max=5, random_state=42)

    assert isinstance(evaluation, KRangeEvaluationResult)
    assert [e.k for e in evaluation.evaluations] == [2, 3, 4, 5]


def test_evaluate_k_range_best_by_silhouette_finds_true_cluster_count() -> None:
    """For 3 obviously separated groups, K=3 should score at least as well as other K."""
    vectors = _three_well_separated_groups(n_per_group=6)

    evaluation = evaluate_k_range(vectors, k_min=2, k_max=6, random_state=42)
    best = evaluation.best_by_silhouette()

    assert best is not None
    assert best.k == 3


def test_evaluate_k_range_caps_k_max_to_available_sessions() -> None:
    """k_max larger than n_samples should be capped, not raise, as long as k_min still fits."""
    vectors = _two_well_separated_groups(n_per_group=2)  # 4 sessions

    evaluation = evaluate_k_range(vectors, k_min=2, k_max=100, random_state=42)

    assert max(e.k for e in evaluation.evaluations) == 4


def test_evaluate_k_range_raises_when_k_min_unreachable() -> None:
    vectors = _two_well_separated_groups(n_per_group=1)  # 2 sessions

    with pytest.raises(ValueError, match="Insufficient data"):
        evaluate_k_range(vectors, k_min=5, k_max=10, random_state=42)


def test_evaluate_k_range_rejects_invalid_bounds() -> None:
    vectors = _two_well_separated_groups()

    with pytest.raises(ValueError, match="k_min"):
        evaluate_k_range(vectors, k_min=0, k_max=3)

    with pytest.raises(ValueError, match="k_max"):
        evaluate_k_range(vectors, k_min=5, k_max=2)


def test_evaluate_k_range_with_no_valid_silhouette_returns_none_best() -> None:
    """All-identical data across the whole K range should yield no usable silhouette."""
    vectors = _identical_vectors(n=5, values=(1.0, 1.0))

    evaluation = evaluate_k_range(vectors, k_min=2, k_max=4, random_state=42)

    assert evaluation.best_by_silhouette() is None


def test_as_silhouette_table_shape() -> None:
    vectors = _three_well_separated_groups(n_per_group=4)

    evaluation = evaluate_k_range(vectors, k_min=2, k_max=4, random_state=42)
    table = evaluation.as_silhouette_table()

    assert len(table) == 3
    assert all(isinstance(k, int) for k, _ in table)


# ============================================================================
# Feature dimension mismatch
# ============================================================================


def test_feature_vectors_with_different_names_raise() -> None:
    vectors = [
        _vector("a", (1.0, 2.0), names=("f1", "f2")),
        _vector("b", (3.0, 4.0, 5.0), names=("f1", "f2", "f3")),
    ]
    clusterer = ContextClusterer(n_clusters=1, random_state=42)

    with pytest.raises(ValueError, match="dimension mismatch"):
        clusterer.fit(vectors)


def test_feature_vectors_with_reordered_names_raise() -> None:
    """Same feature names but different order must be rejected, not silently accepted."""
    vectors = [
        _vector("a", (1.0, 2.0), names=("f1", "f2")),
        _vector("b", (3.0, 4.0), names=("f2", "f1")),
    ]
    clusterer = ContextClusterer(n_clusters=1, random_state=42)

    with pytest.raises(ValueError, match="dimension mismatch"):
        clusterer.fit(vectors)


def test_raw_matrix_with_inconsistent_row_lengths_raises() -> None:
    clusterer = ContextClusterer(n_clusters=1, random_state=42)

    with pytest.raises(ValueError, match="dimension mismatch"):
        clusterer.fit_matrix([[1.0, 2.0], [3.0, 4.0, 5.0]], ["a", "b"])


def test_raw_matrix_with_zero_features_raises() -> None:
    clusterer = ContextClusterer(n_clusters=1, random_state=42)

    with pytest.raises(ValueError, match="at least one feature"):
        clusterer.fit_matrix([[], []], ["a", "b"])


# ============================================================================
# fit_matrix without feature_names (placeholder names)
# ============================================================================


def test_fit_matrix_without_feature_names_uses_placeholders() -> None:
    clusterer = ContextClusterer(n_clusters=2, random_state=42)

    result = clusterer.fit_matrix(
        [[0.0, 0.0], [0.1, 0.1], [50.0, 50.0], [50.1, 50.1]],
        ["a", "b", "c", "d"],
    )

    assert result.feature_names == ("feature_0", "feature_1")


# ============================================================================
# Real Phase 2A integration (FeatureExtractor -> ContextClusterer)
# ============================================================================


def test_integration_with_real_feature_extractor_output() -> None:
    """
    End-to-end sanity check: features produced by the actual Phase 2A
    FeatureExtractor (not synthetic FeatureVectors) can be clustered.
    """
    extractor = FeatureExtractor()

    # Session A: focused single-app coding session, morning.
    session_a_start = _timestamp(0, hour=9)
    session_a_activities = [
        _MockActivity(session_a_start, session_a_start + timedelta(seconds=1800), 1800.0, "VSCode", "Code.exe"),
    ]
    features_a = extractor.extract_features(
        session_id="session-a",
        session_started_at=session_a_start,
        session_ended_at=session_a_start + timedelta(seconds=1800),
        activities=session_a_activities,
    )

    # Session B: scattered multi-app browsing session, evening.
    session_b_start = _timestamp(0, hour=20)
    session_b_activities = []
    for i in range(6):
        session_b_activities.append(
            _MockActivity(
                session_b_start + timedelta(seconds=i * 100),
                session_b_start + timedelta(seconds=(i + 1) * 100),
                100.0,
                f"App{i % 3}",
                f"app{i % 3}.exe",
            )
        )
    features_b = extractor.extract_features(
        session_id="session-b",
        session_started_at=session_b_start,
        session_ended_at=session_b_start + timedelta(seconds=600),
        activities=session_b_activities,
    )

    # A third, near-duplicate of session A to give K-Means something to group.
    session_c_start = _timestamp(0, hour=9) + timedelta(days=1)
    session_c_activities = [
        _MockActivity(session_c_start, session_c_start + timedelta(seconds=1700), 1700.0, "VSCode", "Code.exe"),
    ]
    features_c = extractor.extract_features(
        session_id="session-c",
        session_started_at=session_c_start,
        session_ended_at=session_c_start + timedelta(seconds=1700),
        activities=session_c_activities,
    )

    clusterer = ContextClusterer(n_clusters=2, random_state=42)
    result = clusterer.fit([features_a, features_b, features_c])

    assert result.n_samples == 3
    assert result.feature_names == FeatureExtractor.FEATURE_NAMES
    labels = result.to_label_map()
    # The two VSCode-only sessions should land in the same cluster, distinct from the browsing one.
    assert labels["session-a"] == labels["session-c"]
    assert labels["session-a"] != labels["session-b"]


def test_integration_batch_extractor_matrix_feeds_clusterer_directly() -> None:
    """BatchFeatureExtractor's matrix output should be directly usable via fit_matrix."""
    from app.ml.feature_engineering import BatchFeatureExtractor
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class MockSession:
        id: str
        started_at: datetime
        ended_at: datetime | None

    batch_extractor = BatchFeatureExtractor()

    sessions_with_activities = []
    for group in range(2):
        for i in range(3):
            start = _timestamp(0, hour=9 + group * 10) + timedelta(days=i)
            end = start + timedelta(seconds=600)
            session = MockSession(f"g{group}-{i}", start, end)
            activity = _MockActivity(start, end, 600.0, f"App{group}", f"app{group}.exe")
            sessions_with_activities.append((session, [activity]))

    feature_matrix, session_ids = batch_extractor.extract_as_matrix(sessions_with_activities)

    clusterer = ContextClusterer(n_clusters=2, random_state=42)
    result = clusterer.fit_matrix(feature_matrix, session_ids, feature_names=FeatureExtractor.FEATURE_NAMES)

    assert result.n_samples == 6
    assert len(result.assignments) == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
