"""
Phase 2C: Human Interpretation and Persistent Context Labeling

Bridges Phase 2B's anonymous K-Means cluster IDs to human-assigned
semantic meaning, and persists that mapping locally so it survives
process restarts.

    Phase 2A: activity/session data -> behavioral FeatureVector
    Phase 2B: FeatureVector -> K-Means -> anonymous cluster IDs
    Phase 2C: cluster ID -> cluster summary/interpretation -> USER LABEL
              -> persistent cluster -> context mapping

Design principles
------------------
- This module has NO knowledge of sessions, activities, or the database.
  It only operates on `ClusteringResult` objects produced by Phase 2B
  (`app.ml.context_clustering`), the same way Phase 2B only operates on
  `FeatureVector` objects produced by Phase 2A. Each phase depends only on
  the phase directly below it.
- It never invents, infers, or guesses a label. There is no keyword
  mapping ("Chrome means Browsing"), no predefined category list ("Work",
  "Gaming", ...), and no LLM call. Every label stored here was typed by a
  human through `ContextLabelStore.assign_label`.
- Cluster *summaries* are built entirely from information Phase 2B already
  computed (cluster size, cluster center, silhouette score/note). This
  module does not re-run K-Means or recompute anything Phase 2B produced.

Cluster identity is NOT globally permanent
-------------------------------------------
A raw K-Means cluster label (0, 1, 2, ...) is only meaningful *within the
one `ClusteringResult` that produced it*. K-Means does not guarantee any
particular ordering or numbering between separate fits -- retraining
tomorrow with one more session, a different `n_clusters`, or a different
`random_state` can (and typically will) shuffle which integer represents
which behavioral pattern, or dissolve/merge/split patterns entirely.

To avoid silently treating "cluster 0" from two different training runs
as if they were the same context, every label in this module is keyed by
a `run_id`: a deterministic fingerprint computed from the *content* of the
`ClusteringResult` itself (feature names, K, sample count, random_state,
and the resulting cluster centers/inertia -- see `compute_run_id`).

Consequences of this design, stated explicitly:
  - The same `ClusteringResult` content always produces the same `run_id`,
    so labels persist correctly across repeated reads of one clustering
    outcome and across process restarts.
  - A *different* clustering run (new data, different K, different
    random_state, or any change that shifts the cluster centers) produces
    a *different* `run_id`. Labels assigned under the old run_id will NOT
    automatically apply to the new run, even if a human would recognize
    "cluster 0" in both as "the same" context.
  - Carrying labels forward across retraining runs (e.g. by matching new
    cluster centers to previously-labeled ones) is a reconciliation
    problem explicitly OUT OF SCOPE for Phase 2C. Re-labeling after a
    retrain is a deliberate, visible step -- not something this module
    does silently.

Persistence
-----------
Labels are stored in a small dedicated local JSON file
(`data/cluster_labels.json` by default), completely separate from the
SQLite activity database. This keeps Phase 2C decoupled from the Phase
1 storage schema: no schema changes, no migrations, no shared tables.
The file is rewritten atomically (write-to-temp then replace) on every
mutation, so a `ContextLabelStore` pointed at the same path from a new
process (or a new object) always sees the latest state -- there is no
separate "save" step to remember.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.ml.context_clustering import ClusteringResult

# ============================================================================
# Cluster interpretation (read-only summaries of a Phase 2B result)
# ============================================================================


@dataclass(frozen=True, slots=True)
class ClusterSummary:
    """
    A human-readable interpretation of one cluster from a Phase 2B
    `ClusteringResult`. Contains no semantic judgement -- just the
    numeric facts a person needs in order to decide what, if anything,
    the cluster represents.
    """

    run_id: str
    cluster_label: int
    n_sessions: int
    cluster_share: float  # n_sessions / total sessions in the run, 0.0-1.0
    feature_center: dict[str, float]  # feature_name -> cluster-center value
    silhouette: float | None
    silhouette_note: str


def compute_run_id(result: ClusteringResult) -> str:
    """
    Derive a deterministic identity for one clustering *result*.

    The fingerprint is built from the result's feature names, K, sample
    count, random_state, cluster centers, and inertia -- i.e. everything
    that characterizes what was actually discovered, not just the
    parameters that were requested. Two calls on `ClusteringResult`
    objects with identical content always produce the identical run_id;
    any change to the underlying data or clustering configuration that
    shifts the cluster centers produces a different run_id.

    This is a pure function of the result's content: it does not read the
    system clock, generate a random UUID, or depend on call order.
    """
    payload = {
        "feature_names": list(result.feature_names),
        "n_clusters": result.n_clusters,
        "n_samples": result.n_samples,
        "random_state": result.random_state,
        # Rounded to guard against negligible floating-point jitter while
        # still uniquely fingerprinting materially different results.
        "cluster_centers": [[round(value, 8) for value in center] for center in result.cluster_centers],
        "inertia": round(result.inertia, 6),
    }
    canonical_text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
    return f"run-{digest[:16]}"


def summarize_cluster(
    result: ClusteringResult,
    cluster_label: int,
    run_id: str | None = None,
) -> ClusterSummary:
    """
    Build a human-readable summary of one cluster within `result`.

    Args:
        result: A Phase 2B `ClusteringResult`.
        cluster_label: Which cluster (0..result.n_clusters-1) to summarize.
        run_id: Optional precomputed run_id (see `compute_run_id`). Pass
            this when summarizing several clusters from the same result
            to avoid recomputing the fingerprint each time. If omitted,
            it is computed from `result`.

    Returns:
        A ClusterSummary describing size, share, center, and clustering
        quality for the requested cluster.

    Raises:
        ValueError: if `cluster_label` is out of range for `result`
            (delegated to `ClusteringResult.center_as_dict`).
    """
    # Reuses Phase 2B's own range validation and unit-converted center
    # rather than re-deriving it, so there is exactly one source of truth
    # for "what does this cluster's center look like in original units".
    feature_center = result.center_as_dict(cluster_label)

    resolved_run_id = run_id if run_id is not None else compute_run_id(result)
    n_sessions = result.cluster_sizes.get(cluster_label, 0)
    cluster_share = (n_sessions / result.n_samples) if result.n_samples > 0 else 0.0

    return ClusterSummary(
        run_id=resolved_run_id,
        cluster_label=cluster_label,
        n_sessions=n_sessions,
        cluster_share=cluster_share,
        feature_center=feature_center,
        silhouette=result.silhouette,
        silhouette_note=result.silhouette_note,
    )


def summarize_all_clusters(
    result: ClusteringResult,
    run_id: str | None = None,
) -> tuple[ClusterSummary, ...]:
    """
    Build a `ClusterSummary` for every cluster in `result`, in cluster
    label order (0, 1, 2, ...).
    """
    resolved_run_id = run_id if run_id is not None else compute_run_id(result)
    return tuple(
        summarize_cluster(result, cluster_label, run_id=resolved_run_id)
        for cluster_label in range(result.n_clusters)
    )


# ============================================================================
# Persistent human labels
# ============================================================================


@dataclass(frozen=True, slots=True)
class LabeledCluster:
    """One user-assigned semantic label for one (run_id, cluster_label) pair."""

    run_id: str
    cluster_label: int
    label: str
    created_at: datetime  # when this cluster was first labeled (UTC)
    updated_at: datetime  # when the label was last set/changed (UTC)


class ContextLabelStore:
    """
    Local, JSON-backed persistence for user-assigned cluster labels.

    Deliberately independent of `ActivityRepository`'s SQLite schema:
    cluster labels are a lightweight concern layered on top of ML
    clustering results, not part of the core activity-monitoring data
    model. No schema changes and no migration system were introduced for
    this phase -- a dedicated JSON file plays that role instead.

    Every method reads the current file state from disk and, for
    mutations, writes it back immediately. There is no separate "save"
    call and no held-open file handle: a second `ContextLabelStore`
    instance pointed at the same path (in this process or a new one)
    always observes the latest persisted state.
    """

    SCHEMA_VERSION = 1

    def __init__(self, storage_path: str | Path | None = None) -> None:
        default_path = Path(__file__).resolve().parents[2] / "data" / "cluster_labels.json"
        self.storage_path = Path(storage_path) if storage_path else default_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            self._write_all({})

    def assign_label(
        self,
        run_id: str,
        cluster_label: int,
        label: str,
        *,
        assigned_at: datetime,
    ) -> LabeledCluster:
        """
        Assign (or reassign/update) a semantic label to one cluster.

        Any non-empty string is accepted as `label` -- there is no fixed
        vocabulary. Calling this again for the same (run_id, cluster_label)
        overwrites the previous label and updates `updated_at`, while
        preserving the original `created_at`.

        Args:
            run_id: The clustering run this label belongs to (see
                `compute_run_id`).
            cluster_label: Which cluster within that run (>= 0).
            label: The human-provided semantic label. Leading/trailing
                whitespace is stripped; empty or whitespace-only labels
                are rejected.
            assigned_at: Timezone-aware timestamp for this assignment.
                Required explicitly (rather than read from the system
                clock) so callers -- including tests -- get fully
                deterministic, reproducible results.

        Returns:
            The stored LabeledCluster.

        Raises:
            ValueError: on an invalid run_id, cluster_label, label, or a
                timezone-naive `assigned_at`.
        """
        _validate_run_id(run_id)
        _validate_cluster_label(cluster_label)
        cleaned_label = _validate_and_clean_label(label)
        _require_timezone_aware(assigned_at, "assigned_at")

        entries = self._read_all()
        key = _composite_key(run_id, cluster_label)
        existing = entries.get(key)
        created_at = existing.created_at if existing is not None else assigned_at

        entry = LabeledCluster(
            run_id=run_id,
            cluster_label=cluster_label,
            label=cleaned_label,
            created_at=created_at,
            updated_at=assigned_at,
        )
        entries[key] = entry
        self._write_all(entries)
        return entry

    def get_label(self, run_id: str, cluster_label: int) -> LabeledCluster | None:
        """Return the label for (run_id, cluster_label), or None if unlabeled."""
        return self._read_all().get(_composite_key(run_id, cluster_label))

    def remove_label(self, run_id: str, cluster_label: int) -> bool:
        """
        Remove a cluster's label, if one exists.

        Returns:
            True if a label was removed, False if the cluster had no
            label to begin with (this is not an error -- removing an
            already-unlabeled cluster is a safe no-op).
        """
        entries = self._read_all()
        key = _composite_key(run_id, cluster_label)
        if key not in entries:
            return False
        del entries[key]
        self._write_all(entries)
        return True

    def list_labels(self) -> tuple[LabeledCluster, ...]:
        """Return every stored label, across all runs, sorted for stable output."""
        entries = self._read_all()
        return tuple(sorted(entries.values(), key=lambda entry: (entry.run_id, entry.cluster_label)))

    def list_labels_for_run(self, run_id: str) -> tuple[LabeledCluster, ...]:
        """Return only the labels belonging to one clustering run, sorted by cluster_label."""
        return tuple(entry for entry in self.list_labels() if entry.run_id == run_id)

    # ------------------------------------------------------------------
    # Persistence internals
    # ------------------------------------------------------------------

    def _read_all(self) -> dict[str, LabeledCluster]:
        if not self.storage_path.exists():
            return {}
        raw_text = self.storage_path.read_text(encoding="utf-8")
        if not raw_text.strip():
            return {}
        payload = json.loads(raw_text)
        entries: dict[str, LabeledCluster] = {}
        for key, record in payload.get("labels", {}).items():
            entries[key] = LabeledCluster(
                run_id=record["run_id"],
                cluster_label=record["cluster_label"],
                label=record["label"],
                created_at=_parse_utc(record["created_at_utc"]),
                updated_at=_parse_utc(record["updated_at_utc"]),
            )
        return entries

    def _write_all(self, entries: dict[str, LabeledCluster]) -> None:
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "labels": {
                key: {
                    "run_id": entry.run_id,
                    "cluster_label": entry.cluster_label,
                    "label": entry.label,
                    "created_at_utc": _format_utc(entry.created_at),
                    "updated_at_utc": _format_utc(entry.updated_at),
                }
                for key, entry in entries.items()
            },
        }
        # Deterministic formatting (sorted keys, fixed indent) so that
        # writing the same logical content always produces byte-identical
        # output -- useful both for debugging and for tests asserting
        # deterministic persistence.
        canonical_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"

        # Atomic write: write to a temp file in the same directory, then
        # replace the real file in one step, so a crash mid-write cannot
        # leave a truncated/corrupt label file behind.
        tmp_path = self.storage_path.with_suffix(self.storage_path.suffix + ".tmp")
        tmp_path.write_text(canonical_text, encoding="utf-8")
        tmp_path.replace(self.storage_path)


# ============================================================================
# Bridging helper (read-only; not a model, not a prediction)
# ============================================================================


def session_id_to_label(
    result: ClusteringResult,
    store: ContextLabelStore,
    run_id: str | None = None,
) -> dict[str, str | None]:
    """
    Map each session in `result` to its human-assigned label text, or
    None if that session's cluster has not been labeled yet.

    This is a pure read-side lookup that bridges Phase 2B's numeric
    cluster assignments to Phase 2C's persisted labels. It performs no
    learning, inference, or classification of its own -- it exists so
    that a future Phase 2D supervised classifier has a ready-made
    (session_id -> label) view to build a training set from (treating
    sessions with a non-None label as labeled examples, and sessions
    with None as still unlabeled).

    Args:
        result: The Phase 2B clustering result whose sessions to map.
        store: Where persisted labels are read from.
        run_id: Optional precomputed run_id for `result`. If omitted, it
            is computed via `compute_run_id`.

    Returns:
        {session_id: label_text_or_None} for every session in `result`.
    """
    resolved_run_id = run_id if run_id is not None else compute_run_id(result)

    label_by_cluster: dict[int, str | None] = {}
    for cluster_label in range(result.n_clusters):
        entry = store.get_label(resolved_run_id, cluster_label)
        label_by_cluster[cluster_label] = entry.label if entry is not None else None

    return {
        assignment.session_id: label_by_cluster.get(assignment.cluster_label)
        for assignment in result.assignments
    }


# ============================================================================
# Validation & small internal helpers
# ============================================================================


def _validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string.")


def _validate_cluster_label(cluster_label: int) -> None:
    if isinstance(cluster_label, bool) or not isinstance(cluster_label, int) or cluster_label < 0:
        raise ValueError("cluster_label must be a non-negative integer.")


def _validate_and_clean_label(label: str) -> str:
    if not isinstance(label, str):
        raise ValueError("label must be a string.")
    cleaned = label.strip()
    if not cleaned:
        raise ValueError("label must not be empty or whitespace-only.")
    return cleaned


def _require_timezone_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware.")


def _composite_key(run_id: str, cluster_label: int) -> str:
    return f"{run_id}::{cluster_label}"


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)
