"""
Phase 2E: Durable Training-Example Corpus

Fixes the flagged Phase 2C persistence gap: `ContextLabelStore` (2C)
persists only `{(run_id, cluster_label): label}`, never the
`ClusteringResult` or `FeatureVector`s that produced it. Once the process
holding that in-memory `ClusteringResult` exits, Phase 2D's
`build_training_set()` bridge becomes unusable for that data -- not
because of a bug, but because the thing it needs to join against no
longer exists anywhere. This module durably captures the *joined*
(feature_vector, label) pairs at the moment they are produced, so labeled
examples accumulate across independent, unrelated clustering runs and
across process restarts -- which is a prerequisite for any real
"personalization over time".

    Phase 2A: activity/session data -> behavioral FeatureVector
    Phase 2B: FeatureVector -> K-Means -> anonymous cluster IDs
    Phase 2C: cluster ID -> human label -> persisted (run_id, cluster_label) -> label
    Phase 2E: FeatureVector + resolved label -> durable LabeledExample corpus

This module has NO knowledge of Phase 2D's `ContextClassifier` at all --
it is corpus-only. Training/retraining lives in `training_pipeline.py`.

Anti-leakage
------------
`LabeledExample.feature_values` is copied byte-for-byte from the
originating `FeatureVector.feature_values`. Provenance fields
(`source_run_id`, `source_cluster_label`, `session_id`,
`feature_schema_version`, `created_at`, `updated_at`) are stored
*alongside* the numeric payload, never merged into it -- exactly the
pattern `LabeledCluster` (2C) and `PredictionResult` (2D) already use for
carrying identifiers next to, not inside, model input. `label` is stored
as the training TARGET, never as a feature.

Labels come ONLY from Phase 2C
-------------------------------
`capture_labeled_examples` never invents, infers, or derives a label. It
is a pure bridge built on 2C's unmodified `session_id_to_label`: a
session whose cluster has no Phase 2C label is silently skipped, exactly
matching the semantics Phase 2D's `build_training_set` already
established.

Persistence
-----------
`TrainingExampleStore` follows the exact conventions `ContextLabelStore`
(2C) already established: a small dedicated local JSON file
(`data/training_examples.json` by default), atomic writes (temp file +
`replace()`), deterministic serialization (`sort_keys=True`, fixed
indent), UTC ISO-8601 timestamps, and read-modify-write on every mutating
call so a second `TrainingExampleStore` instance on the same path --
including one in a freshly restarted process -- always observes the
latest state. No SQLite involvement, no schema changes, no migration
system.

Deduplication
-------------
Exactly one `LabeledExample` per `session_id`. Re-capturing the same
session (whether from the same clustering run again, or a different one
entirely) overwrites the existing record's `feature_values`/`label`/
provenance/`updated_at`, while preserving the original `created_at`. This
upsert-by-`session_id` *is* the deduplication mechanism; there is no
separate dedup pass.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from app.ml.feature_engineering import FeatureVector

if TYPE_CHECKING:
    # Only imported for type hints (never at runtime) so this module stays
    # a pure corpus/persistence layer at import time. The one function
    # that genuinely needs these at runtime (`capture_labeled_examples`)
    # imports them locally instead.
    from app.ml.context_clustering import ClusteringResult
    from app.ml.context_labeling import ContextLabelStore

# Bumped only if Phase 2A's FEATURE_NAMES shape/order ever changes in a
# way that makes old examples numerically incomparable to new ones.
CURRENT_FEATURE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class LabeledExample:
    """
    One durably-persisted (behavioral feature vector, human label) pair.

    Only `feature_names`/`feature_values` are ever used as ML input.
    Every other field is identity/provenance/bookkeeping -- see the
    module docstring's "Anti-leakage" section.
    """

    session_id: str
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    label: str
    feature_schema_version: int
    source_run_id: str
    source_cluster_label: int
    created_at: datetime  # UTC, tz-aware
    updated_at: datetime  # UTC, tz-aware


class TrainingExampleStore:
    """
    Local, JSON-backed persistence for the durable labeled-example corpus.

    Deliberately independent of `ActivityRepository`'s SQLite schema and
    of `ContextLabelStore`'s own file, for the same reason 2C's store is
    independent of the activity database: this is a lightweight,
    dedicated concern layered on top of the ML pipeline, not part of the
    core monitoring data model.
    """

    SCHEMA_VERSION = 1  # JSON *file format* version (distinct from feature_schema_version)

    def __init__(self, storage_path: str | Path | None = None) -> None:
        default_path = Path(__file__).resolve().parents[2] / "data" / "training_examples.json"
        self.storage_path = Path(storage_path) if storage_path else default_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            self._write_all({})

    def record_example(
        self,
        session_id: str,
        feature_names: tuple[str, ...],
        feature_values: tuple[float, ...],
        label: str,
        *,
        source_run_id: str,
        source_cluster_label: int,
        captured_at: datetime,
        feature_schema_version: int = CURRENT_FEATURE_SCHEMA_VERSION,
    ) -> LabeledExample:
        """
        Record (or update, upserting by `session_id`) one labeled example.

        Calling this again for a `session_id` that already has a record
        overwrites its `feature_values`/`label`/provenance and sets a new
        `updated_at`, while preserving the original `created_at`.

        Raises:
            ValueError: on an invalid session_id, mismatched/empty/
                non-finite feature_names/feature_values, an invalid
                label, an invalid source_run_id/source_cluster_label, or
                a timezone-naive `captured_at`.
        """
        _validate_session_id(session_id)
        _validate_feature_names_values(feature_names, feature_values)
        cleaned_label = _validate_and_clean_label(label)
        _validate_run_id(source_run_id)
        _validate_cluster_label(source_cluster_label)
        _require_timezone_aware(captured_at, "captured_at")

        entries = self._read_all()
        existing = entries.get(session_id)
        created_at = existing.created_at if existing is not None else captured_at

        entry = LabeledExample(
            session_id=session_id,
            feature_names=tuple(feature_names),
            feature_values=tuple(float(value) for value in feature_values),
            label=cleaned_label,
            feature_schema_version=feature_schema_version,
            source_run_id=source_run_id,
            source_cluster_label=source_cluster_label,
            created_at=created_at,
            updated_at=captured_at,
        )
        entries[session_id] = entry
        self._write_all(entries)
        return entry

    def get_example(self, session_id: str) -> LabeledExample | None:
        """Return the example for `session_id`, or None if none is stored."""
        return self._read_all().get(session_id)

    def remove_example(self, session_id: str) -> bool:
        """
        Remove one example, if it exists.

        Returns:
            True if an example was removed, False if there was nothing to
            remove (not an error -- removing an absent example is a safe
            no-op).
        """
        entries = self._read_all()
        if session_id not in entries:
            return False
        del entries[session_id]
        self._write_all(entries)
        return True

    def list_examples(self) -> tuple[LabeledExample, ...]:
        """Return every stored example, sorted deterministically by session_id."""
        entries = self._read_all()
        return tuple(sorted(entries.values(), key=lambda entry: entry.session_id))

    def list_examples_by_label(self, label: str) -> tuple[LabeledExample, ...]:
        """Return only the examples currently assigned `label`, sorted by session_id."""
        return tuple(entry for entry in self.list_examples() if entry.label == label)

    # ------------------------------------------------------------------
    # Persistence internals
    # ------------------------------------------------------------------

    def _read_all(self) -> dict[str, LabeledExample]:
        if not self.storage_path.exists():
            return {}
        raw_text = self.storage_path.read_text(encoding="utf-8")
        if not raw_text.strip():
            return {}
        payload = json.loads(raw_text)
        entries: dict[str, LabeledExample] = {}
        for session_id, record in payload.get("examples", {}).items():
            entries[session_id] = LabeledExample(
                session_id=record["session_id"],
                feature_names=tuple(record["feature_names"]),
                feature_values=tuple(float(value) for value in record["feature_values"]),
                label=record["label"],
                feature_schema_version=record["feature_schema_version"],
                source_run_id=record["source_run_id"],
                source_cluster_label=record["source_cluster_label"],
                created_at=_parse_utc(record["created_at_utc"]),
                updated_at=_parse_utc(record["updated_at_utc"]),
            )
        return entries

    def _write_all(self, entries: dict[str, LabeledExample]) -> None:
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "examples": {
                session_id: {
                    "session_id": entry.session_id,
                    "feature_names": list(entry.feature_names),
                    "feature_values": list(entry.feature_values),
                    "label": entry.label,
                    "feature_schema_version": entry.feature_schema_version,
                    "source_run_id": entry.source_run_id,
                    "source_cluster_label": entry.source_cluster_label,
                    "created_at_utc": _format_utc(entry.created_at),
                    "updated_at_utc": _format_utc(entry.updated_at),
                }
                for session_id, entry in entries.items()
            },
        }
        # Deterministic formatting so identical logical content always
        # produces byte-identical output.
        canonical_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"

        # Atomic write: temp file in the same directory, then replace in
        # one step, so a crash mid-write cannot leave a corrupt file.
        tmp_path = self.storage_path.with_suffix(self.storage_path.suffix + ".tmp")
        tmp_path.write_text(canonical_text, encoding="utf-8")
        tmp_path.replace(self.storage_path)


def capture_labeled_examples(
    feature_vectors: Sequence[FeatureVector],
    result: "ClusteringResult",
    label_store: "ContextLabelStore",
    examples_store: TrainingExampleStore,
    *,
    captured_at: datetime,
    run_id: str | None = None,
) -> tuple[LabeledExample, ...]:
    """
    Join Phase 2A feature vectors with Phase 2C human labels (through the
    Phase 2B clustering result that connects a `session_id` to a
    `cluster_label`), and durably persist the result into `examples_store`.

    This is the ONLY writer path expected to be used at the product
    level; `TrainingExampleStore.record_example` exists for the store's
    own internals/tests. A session whose cluster has no Phase 2C label,
    or that is not part of `result` at all, is silently skipped -- it is
    never guessed or defaulted.

    Args:
        feature_vectors: Phase 2A feature vectors. Any superset is fine;
            only sessions that both appear in `result` and have a
            resolvable label are captured.
        result: The Phase 2B `ClusteringResult` whose sessions to look up
            labels for.
        label_store: Where Phase 2C labels are persisted.
        examples_store: Where captured examples are durably persisted.
        captured_at: Timezone-aware timestamp for this capture (see
            `TrainingExampleStore.record_example`).
        run_id: Optional precomputed run_id for `result` (see
            `app.ml.context_labeling.compute_run_id`). Computed from
            `result` if omitted.

    Returns:
        The `LabeledExample` records written (new or updated) this call.
    """
    from app.ml.context_labeling import compute_run_id, session_id_to_label  # local: bridge-only dependency

    resolved_run_id = run_id if run_id is not None else compute_run_id(result)
    label_by_session = session_id_to_label(result, label_store, run_id=resolved_run_id)
    cluster_label_by_session = {
        assignment.session_id: assignment.cluster_label for assignment in result.assignments
    }

    captured: list[LabeledExample] = []
    for vector in feature_vectors:
        label = label_by_session.get(vector.session_id)
        if label is None:
            # Either the session isn't part of this clustering result at
            # all, or its cluster has no Phase 2C label yet -- both cases
            # are "not resolvable" and are skipped, never guessed.
            continue
        cluster_label = cluster_label_by_session.get(vector.session_id)
        if cluster_label is None:
            continue  # defensive; unreachable if label was resolved, kept for clarity/robustness

        example = examples_store.record_example(
            vector.session_id,
            vector.feature_names,
            vector.feature_values,
            label,
            source_run_id=resolved_run_id,
            source_cluster_label=cluster_label,
            captured_at=captured_at,
        )
        captured.append(example)

    return tuple(captured)


# ============================================================================
# Validation & small internal helpers
# ============================================================================


def _validate_session_id(session_id: str) -> None:
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string.")


def _validate_feature_names_values(
    feature_names: Sequence[str],
    feature_values: Sequence[float],
) -> None:
    if len(feature_names) != len(feature_values):
        raise ValueError(
            f"feature_names and feature_values must have the same length: got "
            f"{len(feature_names)} name(s) and {len(feature_values)} value(s)."
        )
    if not feature_names:
        raise ValueError("feature_names must not be empty.")
    if not all(isinstance(name, str) and name for name in feature_names):
        raise ValueError("feature_names must all be non-empty strings.")
    if not all(math.isfinite(float(value)) for value in feature_values):
        raise ValueError("feature_values must all be finite numbers.")


def _validate_and_clean_label(label: str) -> str:
    if not isinstance(label, str):
        raise ValueError(f"label must be a string, got {type(label).__name__}: {label!r}.")
    cleaned = label.strip()
    if not cleaned:
        raise ValueError("label must not be empty or whitespace-only.")
    return cleaned


def _validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("source_run_id must be a non-empty string.")


def _validate_cluster_label(cluster_label: int) -> None:
    if isinstance(cluster_label, bool) or not isinstance(cluster_label, int) or cluster_label < 0:
        raise ValueError("source_cluster_label must be a non-negative integer.")


def _require_timezone_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware.")


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)
