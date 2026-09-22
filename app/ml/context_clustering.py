"""
Phase 2B: Unsupervised Behavioral Context Discovery

Groups Phase 2A behavioral feature vectors into recurring patterns using
K-Means clustering, without any predefined or hardcoded context labels.

Design principles:
- Operates only on Phase 2A `FeatureVector` objects or equivalent raw matrices.
  This module has NO knowledge of sessions, activities, or the database.
- Introduces no new behavioral features. Whatever Phase 2A extracted is
  what gets clustered.
- Produces numeric cluster labels only (e.g. `0`, `1`, `2`). It never
  assigns semantic names such as "Work" or "Gaming" -- that is a Phase 2C
  (human labeling) concern.
- Silhouette score is reported as a geometric/statistical diagnostic of
  how well-separated the discovered clusters are under Euclidean distance
  on the scaled feature space. A high silhouette score does NOT prove the
  clusters are semantically meaningful; it only indicates the clusters are
  compact and well separated in this particular feature representation.
  Only a human (Phase 2C) can judge semantic meaning.
- Deterministic: every clustering call requires an explicit `random_state`,
  so repeated runs on the same data produce identical cluster assignments.

Preprocessing / scaling
------------------------
K-Means measures similarity using Euclidean distance, so features with
larger natural ranges (e.g. `session_duration_minutes`, 0-1440) would
otherwise dominate features with small ranges (e.g. `dominant_app_percentage`,
0-1, or the binary `is_business_hours`). To prevent this, every feature is
standardized (zero mean, unit variance) with `sklearn.preprocessing.StandardScaler`
before clustering. Cluster centers are transformed back into the original
feature units (via `inverse_transform`) before being reported, so they stay
human-readable in the same units Phase 2A defined.

Known limitation (documented, not silently patched): `session_start_hour`,
`session_end_hour`, and `session_start_day_of_week` are cyclical quantities
(hour 23 is adjacent to hour 0; day 6 is adjacent to day 0), but standard
scaling treats them as ordinary linear values. This module does not invent
a cyclical (sin/cos) encoding for them, because doing so would introduce a
new feature representation beyond what Phase 2A defined, which is out of
scope for Phase 2B. Consumers of `ClusteringResult` should keep this
limitation in mind when interpreting clusters that split along hour/day
boundaries.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import silhouette_score as _sklearn_silhouette_score
from sklearn.preprocessing import StandardScaler

from app.ml.feature_engineering import FeatureVector

# K-Means needs at least this many samples to fit anything meaningful at all.
MIN_SAMPLES_FOR_CLUSTERING = 2


# ============================================================================
# Result data structures
# ============================================================================


@dataclass(frozen=True, slots=True)
class ClusterAssignment:
    """The discovered cluster label for one session."""

    session_id: str
    cluster_label: int


@dataclass(frozen=True, slots=True)
class ClusteringResult:
    """
    The full output of one K-Means clustering run.

    Cluster labels are arbitrary integers (0..n_clusters-1) with no inherent
    meaning. They must be interpreted by a human in a later phase.
    """

    feature_names: tuple[str, ...]
    n_clusters: int
    n_samples: int
    random_state: int
    assignments: tuple[ClusterAssignment, ...]
    cluster_centers: tuple[tuple[float, ...], ...]  # in original feature units
    cluster_sizes: dict[int, int]
    inertia: float
    silhouette: float | None
    silhouette_note: str

    def to_label_map(self) -> dict[str, int]:
        """Return a simple {session_id: cluster_label} mapping."""
        return {assignment.session_id: assignment.cluster_label for assignment in self.assignments}

    def center_as_dict(self, cluster_label: int) -> dict[str, float]:
        """Return one cluster's center as a {feature_name: value} mapping."""
        if cluster_label < 0 or cluster_label >= len(self.cluster_centers):
            raise ValueError(
                f"cluster_label {cluster_label} is out of range for {len(self.cluster_centers)} clusters."
            )
        return dict(zip(self.feature_names, self.cluster_centers[cluster_label]))


@dataclass(frozen=True, slots=True)
class KEvaluation:
    """The clustering result obtained for one candidate value of K."""

    k: int
    result: ClusteringResult


@dataclass(frozen=True, slots=True)
class KRangeEvaluationResult:
    """The results of fitting K-Means across a range of candidate K values."""

    evaluations: tuple[KEvaluation, ...]

    def best_by_silhouette(self) -> KEvaluation | None:
        """
        Return the evaluation with the highest silhouette score among the
        evaluations where a silhouette score could actually be computed.

        Returns None if no evaluation produced a valid silhouette score
        (e.g. every dataset was degenerate, or the range was empty).

        IMPORTANT: This is a statistical suggestion, not a semantic
        judgement. The silhouette score measures geometric cluster
        separation only. It does not know whether the discovered clusters
        correspond to anything a human would recognize as a distinct
        behavioral context. Treat this as one input to a decision, not the
        decision itself.
        """
        candidates = [evaluation for evaluation in self.evaluations if evaluation.result.silhouette is not None]
        if not candidates:
            return None
        return max(candidates, key=lambda evaluation: evaluation.result.silhouette)

    def as_silhouette_table(self) -> tuple[tuple[int, float | None], ...]:
        """Return (k, silhouette_score) pairs for quick inspection/plotting."""
        return tuple((evaluation.k, evaluation.result.silhouette) for evaluation in self.evaluations)


# ============================================================================
# Clustering API
# ============================================================================


class ContextClusterer:
    """
    Discovers recurring behavioral patterns in Phase 2A feature vectors
    using K-Means clustering.

    This class has no knowledge of sessions, activities, or the database.
    It only operates on `FeatureVector` objects (or equivalent raw
    matrices) that the caller supplies.
    """

    def __init__(self, n_clusters: int = 3, random_state: int = 42, n_init: int = 10) -> None:
        """
        Args:
            n_clusters: Number of clusters K-Means should discover. Must be >= 1.
            random_state: Seed controlling K-Means' centroid initialization,
                           so repeated calls on the same data are deterministic.
            n_init: Number of independent K-Means initializations to run;
                     the best (lowest-inertia) result is kept. Passed straight
                     through to scikit-learn.
        """
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1.")
        if n_init < 1:
            raise ValueError("n_init must be at least 1.")
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init

    def fit(self, feature_vectors: Sequence[FeatureVector]) -> ClusteringResult:
        """
        Cluster a collection of Phase 2A `FeatureVector` objects.

        Args:
            feature_vectors: Feature vectors produced by
                `FeatureExtractor`/`BatchFeatureExtractor`. All vectors must
                share the same `feature_names` (same features, same order).

        Returns:
            A ClusteringResult describing the discovered clusters.

        Raises:
            ValueError: if the input is empty, has fewer sessions than
                `n_clusters`, or feature vectors do not share the same
                feature dimensions/names.
        """
        matrix, session_ids, feature_names = _feature_vectors_to_matrix(feature_vectors)
        return self.fit_matrix(matrix, session_ids, feature_names)

    def fit_matrix(
        self,
        feature_matrix: Sequence[Sequence[float]],
        session_ids: Sequence[str],
        feature_names: Sequence[str] | None = None,
    ) -> ClusteringResult:
        """
        Cluster a raw feature matrix directly (without going through
        `FeatureVector` objects).

        Args:
            feature_matrix: A 2D array-like of shape (n_samples, n_features).
                Every row must have the same length.
            session_ids: One session identifier per row, in the same order
                as `feature_matrix`. Must be unique.
            feature_names: Optional names for each feature column, purely
                for interpretability of the reported cluster centers. If
                omitted, generic placeholder names are used.

        Returns:
            A ClusteringResult describing the discovered clusters.

        Raises:
            ValueError: on empty input, mismatched lengths, non-finite
                values, duplicate session IDs, or when `n_clusters` exceeds
                the number of available sessions.
        """
        matrix = _validate_matrix(feature_matrix, session_ids)
        n_samples = matrix.shape[0]

        _require_min_samples(n_samples)
        if self.n_clusters > n_samples:
            raise ValueError(
                f"Insufficient data: requested n_clusters={self.n_clusters} but only "
                f"{n_samples} session(s) are available. Reduce n_clusters or collect "
                f"more monitoring history before clustering."
            )

        scaler = StandardScaler()
        scaled_matrix = scaler.fit_transform(matrix)

        with warnings.catch_warnings():
            # KMeans warns when fewer distinct clusters are found than requested
            # (e.g. degenerate/identical input). We surface this as part of the
            # silhouette note instead of letting it print to stderr.
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=self.n_init,
            )
            labels = model.fit_predict(scaled_matrix)

        centers_original_units = scaler.inverse_transform(model.cluster_centers_)

        silhouette, silhouette_note = _safe_silhouette_score(
            scaled_matrix, labels, self.n_clusters, n_samples
        )

        resolved_feature_names = (
            tuple(feature_names) if feature_names is not None else _placeholder_feature_names(matrix.shape[1])
        )

        return ClusteringResult(
            feature_names=resolved_feature_names,
            n_clusters=self.n_clusters,
            n_samples=n_samples,
            random_state=self.random_state,
            assignments=tuple(
                ClusterAssignment(session_id=session_id, cluster_label=int(label))
                for session_id, label in zip(session_ids, labels)
            ),
            cluster_centers=tuple(tuple(float(value) for value in row) for row in centers_original_units),
            cluster_sizes=_cluster_sizes(labels, self.n_clusters),
            inertia=float(model.inertia_),
            silhouette=silhouette,
            silhouette_note=silhouette_note,
        )


def evaluate_k_range(
    feature_vectors: Sequence[FeatureVector],
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 42,
    n_init: int = 10,
) -> KRangeEvaluationResult:
    """
    Fit K-Means once for every K in [k_min, k_max] (inclusive) and report
    the silhouette score for each, so a human can compare candidate values
    of K rather than assuming a single fixed K is correct.

    If `k_max` exceeds the number of available sessions, it is silently
    capped at `n_samples` (K-Means cannot produce more clusters than there
    are points to cluster). If even `k_min` cannot be satisfied, a
    ValueError is raised describing the insufficient data.

    Args:
        feature_vectors: Feature vectors produced by Phase 2A.
        k_min: Smallest K to evaluate (must be >= 1).
        k_max: Largest K to evaluate (must be >= k_min).
        random_state: Seed used for every K's K-Means fit, for determinism.
        n_init: Number of K-Means initializations per K.

    Returns:
        A KRangeEvaluationResult holding one ClusteringResult per evaluated K.

    Raises:
        ValueError: if k_min/k_max are invalid, input is empty, or there
            is not enough data to satisfy even k_min.
    """
    if k_min < 1:
        raise ValueError("k_min must be at least 1.")
    if k_max < k_min:
        raise ValueError("k_max must be greater than or equal to k_min.")

    matrix, session_ids, feature_names = _feature_vectors_to_matrix(feature_vectors)
    n_samples = matrix.shape[0]
    _require_min_samples(n_samples)

    effective_k_max = min(k_max, n_samples)
    if effective_k_max < k_min:
        raise ValueError(
            f"Insufficient data: cannot evaluate K range [{k_min}, {k_max}] with only "
            f"{n_samples} session(s) available. The largest usable K here is {n_samples}."
        )

    evaluations = []
    for k in range(k_min, effective_k_max + 1):
        clusterer = ContextClusterer(n_clusters=k, random_state=random_state, n_init=n_init)
        result = clusterer.fit_matrix(matrix, session_ids, feature_names)
        evaluations.append(KEvaluation(k=k, result=result))

    return KRangeEvaluationResult(evaluations=tuple(evaluations))


# ============================================================================
# Internal helpers
# ============================================================================


def _require_min_samples(n_samples: int) -> None:
    """Raise a clear error when there is not enough data to cluster at all."""
    if n_samples < MIN_SAMPLES_FOR_CLUSTERING:
        raise ValueError(
            f"Insufficient data: at least {MIN_SAMPLES_FOR_CLUSTERING} sessions are "
            f"required for clustering, got {n_samples}. Monitoring likely has not "
            f"collected enough history yet; clustering on this little data would "
            f"not be meaningful."
        )


def _feature_vectors_to_matrix(
    feature_vectors: Sequence[FeatureVector],
) -> tuple[np.ndarray, list[str], tuple[str, ...]]:
    """Convert a sequence of Phase 2A FeatureVectors into a plain matrix."""
    if not feature_vectors:
        raise ValueError("Insufficient data: no feature vectors were provided.")

    reference_names = feature_vectors[0].feature_names
    for vector in feature_vectors:
        if vector.feature_names != reference_names:
            raise ValueError(
                "Feature dimension mismatch: all feature vectors must share the "
                "same feature names, in the same order. Got "
                f"{vector.feature_names} which differs from {reference_names}."
            )

    matrix = np.array([vector.feature_values for vector in feature_vectors], dtype=float)
    session_ids = [vector.session_id for vector in feature_vectors]
    return matrix, session_ids, reference_names


def _validate_matrix(feature_matrix: Sequence[Sequence[float]], session_ids: Sequence[str]) -> np.ndarray:
    """Validate a raw feature matrix and its accompanying session IDs."""
    if feature_matrix is None or len(feature_matrix) == 0:
        raise ValueError("Insufficient data: feature_matrix is empty.")

    if len(feature_matrix) != len(session_ids):
        raise ValueError(
            f"session_ids length ({len(session_ids)}) does not match feature_matrix "
            f"length ({len(feature_matrix)})."
        )

    if len(set(session_ids)) != len(session_ids):
        raise ValueError("session_ids must be unique; duplicate session ID found.")

    row_lengths = {len(row) for row in feature_matrix}
    if len(row_lengths) > 1:
        raise ValueError(
            f"Feature dimension mismatch: rows have inconsistent lengths {sorted(row_lengths)}. "
            f"Every session must contribute the same number of features."
        )

    matrix = np.asarray(feature_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("feature_matrix must be a 2D array-like of shape (n_samples, n_features).")
    if matrix.shape[1] == 0:
        raise ValueError("feature_matrix rows must contain at least one feature.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("feature_matrix contains NaN or infinite values.")

    return matrix


def _placeholder_feature_names(n_features: int) -> tuple[str, ...]:
    return tuple(f"feature_{index}" for index in range(n_features))


def _cluster_sizes(labels: np.ndarray, n_clusters: int) -> dict[int, int]:
    """Count how many sessions fell into each cluster label, including empty ones."""
    sizes = {label: 0 for label in range(n_clusters)}
    for label in labels:
        sizes[int(label)] = sizes.get(int(label), 0) + 1
    return sizes


def _safe_silhouette_score(
    scaled_matrix: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    n_samples: int,
) -> tuple[float | None, str]:
    """
    Compute the silhouette score only when it is mathematically valid.

    Silhouette score requires:
      - at least 2 clusters requested (undefined for a single cluster)
      - at most n_samples - 1 clusters
      - at least 2 distinct labels actually produced by K-Means (a
        degenerate/identical dataset can collapse to a single label even
        when more clusters were requested)

    Returns:
        (score, note) where score is None when the score could not be
        computed, and note explains why (or confirms success).
    """
    if n_clusters < 2:
        return None, (
            "Silhouette score is undefined for n_clusters=1; there is no "
            "separation to measure."
        )
    if n_clusters > n_samples - 1:
        return None, (
            f"Silhouette score is undefined when n_clusters ({n_clusters}) exceeds "
            f"n_samples - 1 ({n_samples - 1})."
        )

    unique_labels = set(int(label) for label in labels)
    if len(unique_labels) < 2:
        return None, (
            "Silhouette score could not be computed: K-Means produced only "
            f"{len(unique_labels)} distinct cluster(s) even though {n_clusters} were "
            "requested. This usually means the input sessions are identical or "
            "nearly identical in feature space."
        )

    try:
        score = float(_sklearn_silhouette_score(scaled_matrix, labels))
    except ValueError as error:
        return None, f"Silhouette score could not be computed: {error}"

    return score, "Silhouette score computed successfully on the standardized feature space."
