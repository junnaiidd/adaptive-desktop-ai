"""
Phase 2D: Personalized Supervised Context Classification

Learns to classify a session's behavioral context from Phase 2A feature
vectors paired with human-provided labels (typically the labels a person
assigned in Phase 2C), using a scikit-learn `RandomForestClassifier`.

    Phase 2A: activity/session data -> behavioral FeatureVector
    Phase 2B: FeatureVector -> K-Means -> anonymous cluster IDs
    Phase 2C: cluster ID -> cluster summary -> USER LABEL -> persisted mapping
    Phase 2D: FeatureVector + USER LABEL -> RandomForestClassifier -> predicted context

Module layout
-------------
This file has two clearly separated parts:

1. `ContextClassifier` (the model) and its supporting result types. This
   part is deliberately ISOLATED: it imports nothing from Qt/UI,
   `app.core` (monitoring), `app.database` (SQLite), or Phase 2B/2C's
   clustering/labeling persistence. It only knows about Phase 2A's
   `FeatureVector` and scikit-learn. It can be trained, evaluated,
   saved, and loaded with no knowledge of clusters, runs, or the label
   store at all.

2. `build_training_set`, a small Phase 2C BRIDGE function. This is the
   "clean bridge" requested for obtaining labeled training examples from
   Phase 2C where practical. It is the only place in this file that
   imports `app.ml.context_clustering` / `app.ml.context_labeling`, and
   it does not change how either of those modules persists data.

Anti-leakage: what is (and is not) a feature
---------------------------------------------
The classifier is trained and predicts using ONLY `FeatureVector.feature_values`
(the 14 behavioral numbers Phase 2A computed: durations, app-switch rates,
entropy, time-of-day, etc.), matched against `FeatureVector.feature_names`
for validation. The following are NEVER used as model input, anywhere in
this file:

  - cluster ID / cluster_label (Phase 2B)
  - run_id (Phase 2B/2C fingerprint)
  - the context label itself (used only as the training TARGET, never
    concatenated into the feature matrix)
  - session_id (carried through into `PredictionResult` purely for
    attribution/bookkeeping, exactly as Phase 2B carries `session_id` in
    `ClusterAssignment` without treating it as a feature)
  - `FeatureVector.metadata` (segment_count, unique_apps, total_duration --
    Phase 2A's bookkeeping dict; not read by this module at all)
  - raw window titles (never present in Phase 2A features to begin with)

A dedicated test (`test_session_id_and_metadata_are_not_used_as_features`)
verifies this directly: two feature vectors with identical
`feature_values` but different `session_id`/`metadata` produce identical
predictions and identical class probabilities.

Why RandomForestClassifier, and why these hyperparameters
------------------------------------------------------------
`RandomForestClassifier` handles the mixed-scale, non-linear behavioral
features Phase 2A produces without requiring the feature scaling that
K-Means (Phase 2B) needs, supports multiclass and binary classification
natively, exposes `predict_proba` for confidence scores, and gives
interpretable feature importances. Hyperparameters are kept modest and
explicit (`n_estimators`, `max_depth`, `random_state`) -- no grid search,
no hyperparameter optimization, no ensembling of multiple model types,
and no deep learning, per Phase 2D's scope.

On evaluation metrics
----------------------
`evaluate()` reports accuracy, macro-averaged precision/recall/F1 (which
weight every user-defined context equally, regardless of how many
labeled examples it has -- appropriate since a "personalized" model
should not silently ignore a context just because the user has labeled
fewer sessions for it), per-class precision/recall/F1, and a confusion
matrix. `zero_division=0` is used throughout so a class with no
predicted (or no true) examples in a small evaluation set reports 0.0
rather than raising or warning. No specific accuracy number (e.g. "85%")
is asserted as a requirement anywhere in this module; evaluation results
are reported, not graded.

`evaluate()` must be called with a dataset the caller supplies --
typically a held-out split produced by `train_test_split_feature_vectors`
(explicit, deterministic, guards against too-small datasets) or any other
data the caller assembles. `fit()` never evaluates its own training data
and calls it "performance"; training-set accuracy is not exposed as an
evaluation metric anywhere in this module.

On the Phase 2C persistence limitation (read before using `build_training_set`)
----------------------------------------------------------------------------------
Phase 2C's `ContextLabelStore` persists only `{(run_id, cluster_label):
label_text}`. It does NOT persist the `ClusteringResult` (cluster
assignments, centers, etc.) or the `FeatureVector`s that produced it --
those are computed on demand in memory by Phase 2A/2B and never written
to disk as such. Consequently, **a labeled training set cannot be
reconstructed from the label store alone.** To call `build_training_set`,
the caller must still have (or exactly re-derive, with the same data and
the same `random_state`/`n_clusters`) the specific `ClusteringResult`
whose `run_id` the desired labels were assigned under. This is a real,
current limitation of the persistence architecture, not a bug in this
module -- and it is intentionally NOT being solved here by adding a new
database table, a model-artifact cache of `ClusteringResult`s, or any
other persistence change, since Phase 2D's scope is the classifier
itself. Building durable, run-independent training-set persistence (or
solving cross-run cluster reconciliation, which is a different, harder
problem noted in Phase 2C's own docstring) is left to a future phase.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix as _sklearn_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as _sklearn_prf_support
from sklearn.model_selection import train_test_split as _sklearn_train_test_split

from app.ml.feature_engineering import FeatureVector

if TYPE_CHECKING:
    # Only imported for type hints (never at runtime) so the model
    # implementation above stays isolated from Phase 2B/2C at import time.
    # The one function that genuinely needs these at runtime
    # (`build_training_set`) imports them locally instead.
    from app.ml.context_clustering import ClusteringResult
    from app.ml.context_labeling import ContextLabelStore

# A classifier cannot be trained on fewer examples than this.
MIN_SAMPLES_FOR_TRAINING = 2

# A classifier that has only ever seen one label cannot discriminate
# between contexts; it is not meaningfully "a classifier".
MIN_DISTINCT_LABELS_FOR_TRAINING = 2

# train_test_split_feature_vectors needs enough data that both the train
# and test splits can be non-empty.
MIN_SAMPLES_FOR_SPLIT = 4


# ============================================================================
# Result data structures
# ============================================================================


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    """What happened during one `ContextClassifier.fit()` call."""

    n_samples: int
    n_features: int
    feature_names: tuple[str, ...]
    classes: tuple[str, ...]
    n_classes: int
    random_state: int


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """
    One prediction for one session.

    `session_id` is carried through purely for attribution/bookkeeping --
    it is never part of the model's input features (see module docstring).
    """

    session_id: str
    predicted_label: str
    class_probabilities: dict[str, float]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """
    Classification metrics computed on a caller-supplied labeled dataset.

    `labels` is the sorted union of every class the model was trained on
    and every class actually present in the evaluated data (so a label
    the model has never seen still shows up with 0 precision/recall/F1
    rather than being silently dropped). `confusion_matrix` rows are true
    labels and columns are predicted labels, both ordered per `labels`.
    """

    n_samples: int
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]
    confusion_matrix: tuple[tuple[int, ...], ...]
    labels: tuple[str, ...]


# ============================================================================
# The model (isolated from Qt/UI, monitoring, SQLite, clustering/labeling)
# ============================================================================


class ContextClassifier:
    """
    A personalized supervised classifier that predicts a session's
    behavioral context label from its Phase 2A `FeatureVector`.

    This class has no knowledge of sessions beyond the feature vectors it
    is given, no knowledge of clusters or clustering runs, no knowledge
    of the label store, and no knowledge of the database or UI. It only
    consumes `FeatureVector.feature_values`/`feature_names` (features)
    and plain strings (labels).
    """

    SCHEMA_VERSION = 1
    DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "context_classifier.joblib"

    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int | None = None,
    ) -> None:
        """
        Args:
            random_state: Seed controlling RandomForestClassifier's
                bootstrap sampling and feature selection, so repeated
                training on the same data is deterministic.
            n_estimators: Number of trees in the forest. Kept at
                scikit-learn's own modest default; not tuned.
            max_depth: Optional maximum tree depth. None (scikit-learn's
                default) lets trees grow until leaves are pure or hit
                `min_samples_split`; pass a small integer to keep trees
                shallower and more explainable on small datasets.
        """
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be at least 1 when provided.")

        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth

        self._model: RandomForestClassifier | None = None
        self._feature_names: tuple[str, ...] | None = None
        self._classes: tuple[str, ...] | None = None

    # ------------------------------------------------------------------
    # Model state
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """Whether `fit()` has been called successfully (or a model was loaded)."""
        return self._model is not None

    @property
    def classes(self) -> tuple[str, ...]:
        """The distinct labels this classifier learned, sorted. Requires training."""
        self._require_trained()
        return self._classes

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The feature names/order this classifier was trained on. Requires training."""
        self._require_trained()
        return self._feature_names

    def _require_trained(self) -> None:
        if self._model is None:
            raise RuntimeError(
                "This ContextClassifier has not been trained yet. Call fit() "
                "(or load() an existing model) before predict(), evaluate(), "
                "classes, feature_names, or save()."
            )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, feature_vectors: Sequence[FeatureVector], labels: Sequence[str]) -> TrainingSummary:
        """
        Train (or retrain, overwriting any previous model) on labeled
        feature vectors.

        Args:
            feature_vectors: Phase 2A `FeatureVector` objects. All must
                share the same `feature_names`, in the same order.
            labels: One label string per feature vector, same order and
                length. Labels are arbitrary user-defined strings -- there
                is no fixed vocabulary. Leading/trailing whitespace is
                stripped; empty/whitespace-only labels are rejected.

        Returns:
            A TrainingSummary describing what was learned.

        Raises:
            ValueError: on mismatched lengths, too few examples, fewer
                than 2 distinct labels, inconsistent/mismatched feature
                names across vectors, invalid labels, or non-finite
                feature values.
        """
        matrix, feature_names, cleaned_labels = _validate_training_input(feature_vectors, labels)

        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        model.fit(matrix, cleaned_labels)

        self._model = model
        self._feature_names = feature_names
        self._classes = tuple(str(label) for label in model.classes_)

        return TrainingSummary(
            n_samples=matrix.shape[0],
            n_features=matrix.shape[1],
            feature_names=feature_names,
            classes=self._classes,
            n_classes=len(self._classes),
            random_state=self.random_state,
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, feature_vector: FeatureVector) -> PredictionResult:
        """
        Predict the context label for one session's feature vector.

        Raises:
            RuntimeError: if the classifier has not been trained.
            ValueError: if `feature_vector.feature_names` does not match
                the feature names (and order) this classifier was
                trained on.
        """
        self._require_trained()
        self._validate_feature_vector_matches_model(feature_vector)

        matrix = np.array([feature_vector.feature_values], dtype=float)
        predicted_label = str(self._model.predict(matrix)[0])
        probabilities = self._model.predict_proba(matrix)[0]
        class_probabilities = {
            str(cls): float(prob) for cls, prob in zip(self._model.classes_, probabilities)
        }

        return PredictionResult(
            session_id=feature_vector.session_id,
            predicted_label=predicted_label,
            class_probabilities=class_probabilities,
        )

    def predict_many(self, feature_vectors: Sequence[FeatureVector]) -> tuple[PredictionResult, ...]:
        """Predict for several feature vectors at once. See `predict`."""
        self._require_trained()
        if not feature_vectors:
            return ()
        for vector in feature_vectors:
            self._validate_feature_vector_matches_model(vector)

        matrix = np.array([vector.feature_values for vector in feature_vectors], dtype=float)
        predicted_labels = self._model.predict(matrix)
        probabilities = self._model.predict_proba(matrix)

        results = []
        for vector, predicted_label, probs in zip(feature_vectors, predicted_labels, probabilities):
            class_probabilities = {
                str(cls): float(prob) for cls, prob in zip(self._model.classes_, probs)
            }
            results.append(
                PredictionResult(
                    session_id=vector.session_id,
                    predicted_label=str(predicted_label),
                    class_probabilities=class_probabilities,
                )
            )
        return tuple(results)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, feature_vectors: Sequence[FeatureVector], labels: Sequence[str]) -> EvaluationResult:
        """
        Evaluate the trained model on a caller-supplied labeled dataset.

        This does NOT re-evaluate on training data implicitly -- whatever
        `feature_vectors`/`labels` are passed here is exactly what gets
        scored. Pass a held-out split (see `train_test_split_feature_vectors`)
        to get a genuine generalization estimate rather than a training-set
        score.

        Raises:
            RuntimeError: if the classifier has not been trained.
            ValueError: on mismatched lengths, empty input, feature-name
                mismatch against the trained model, invalid labels, or
                non-finite feature values.
        """
        self._require_trained()

        if len(feature_vectors) != len(labels):
            raise ValueError(
                f"feature_vectors and labels must have the same length: got "
                f"{len(feature_vectors)} feature vector(s) and {len(labels)} label(s)."
            )
        if not feature_vectors:
            raise ValueError("Insufficient data: no feature vectors were provided for evaluation.")

        for vector in feature_vectors:
            self._validate_feature_vector_matches_model(vector)

        matrix = np.array([vector.feature_values for vector in feature_vectors], dtype=float)
        if not np.all(np.isfinite(matrix)):
            raise ValueError("feature_matrix contains NaN or infinite values.")

        true_labels = _clean_labels(labels)
        predicted_labels = [str(label) for label in self._model.predict(matrix)]

        # Union of trained classes and classes actually seen in this
        # evaluation set, so an unseen true label still appears (with 0
        # precision/recall/F1) instead of being silently dropped, and a
        # trained class that never appears here still appears too.
        label_universe = tuple(sorted(set(self._classes) | set(true_labels)))

        accuracy = float(np.mean([t == p for t, p in zip(true_labels, predicted_labels)]))

        precision_macro, recall_macro, f1_macro, _ = _sklearn_prf_support(
            true_labels, predicted_labels, labels=label_universe, average="macro", zero_division=0
        )
        precision_per, recall_per, f1_per, _ = _sklearn_prf_support(
            true_labels, predicted_labels, labels=label_universe, average=None, zero_division=0
        )
        cm = _sklearn_confusion_matrix(true_labels, predicted_labels, labels=label_universe)

        return EvaluationResult(
            n_samples=len(true_labels),
            accuracy=accuracy,
            precision_macro=float(precision_macro),
            recall_macro=float(recall_macro),
            f1_macro=float(f1_macro),
            per_class_precision={cls: float(v) for cls, v in zip(label_universe, precision_per)},
            per_class_recall={cls: float(v) for cls, v in zip(label_universe, recall_per)},
            per_class_f1={cls: float(v) for cls, v in zip(label_universe, f1_per)},
            confusion_matrix=tuple(tuple(int(v) for v in row) for row in cm),
            labels=label_universe,
        )

    def _validate_feature_vector_matches_model(self, feature_vector: FeatureVector) -> None:
        if feature_vector.feature_names != self._feature_names:
            raise ValueError(
                "Feature mismatch: this classifier was trained on "
                f"{len(self._feature_names)} feature(s) named {self._feature_names}, but "
                f"received {len(feature_vector.feature_names)} feature(s) named "
                f"{feature_vector.feature_names}."
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> Path:
        """
        Save this trained classifier to disk with `joblib` (the standard
        scikit-learn-recommended serialization approach for fitted
        estimators).

        Args:
            path: Where to write the model artifact. Defaults to
                `DEFAULT_MODEL_PATH` (`models/context_classifier.joblib`
                under the project root -- the existing `models/`
                directory). Generated artifacts are not committed to Git
                (see `.gitignore`'s `models/*.joblib` pattern).

        Returns:
            The path actually written to.

        Raises:
            RuntimeError: if the classifier has not been trained.
        """
        self._require_trained()
        target = Path(path) if path is not None else self.DEFAULT_MODEL_PATH
        target.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "random_state": self.random_state,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "feature_names": self._feature_names,
            "classes": self._classes,
            "model": self._model,
        }
        joblib.dump(payload, target)
        return target

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ContextClassifier":
        """
        Load a classifier previously written by `save()`.

        Args:
            path: Where to read the model artifact from. Defaults to
                `DEFAULT_MODEL_PATH`.

        Raises:
            FileNotFoundError: if no artifact exists at the resolved path.
        """
        source = Path(path) if path is not None else cls.DEFAULT_MODEL_PATH
        if not source.exists():
            raise FileNotFoundError(f"No saved ContextClassifier found at {source}.")

        payload = joblib.load(source)
        instance = cls(
            random_state=payload["random_state"],
            n_estimators=payload["n_estimators"],
            max_depth=payload["max_depth"],
        )
        instance._model = payload["model"]
        instance._feature_names = tuple(payload["feature_names"])
        instance._classes = tuple(payload["classes"])
        return instance


# ============================================================================
# Explicit, deterministic train/test split (opt-in; not run inside fit())
# ============================================================================


def train_test_split_feature_vectors(
    feature_vectors: Sequence[FeatureVector],
    labels: Sequence[str],
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[tuple[tuple[FeatureVector, ...], tuple[str, ...]], tuple[tuple[FeatureVector, ...], tuple[str, ...]]]:
    """
    Split labeled feature vectors into a train set and a held-out test
    set, explicitly and deterministically.

    This is a standalone helper the CALLER opts into -- `ContextClassifier.fit`
    never performs a hidden internal split. Splitting is stratified by
    label when every class has at least 2 examples (so class proportions
    are preserved in both splits); otherwise it falls back to a plain
    random split, since scikit-learn cannot stratify a class with fewer
    than 2 members.

    Args:
        feature_vectors: Labeled feature vectors to split.
        labels: One label per feature vector, same order/length.
        test_size: Fraction (0-1 exclusive) of examples to hold out for
            testing.
        random_state: Seed for the split, for determinism.

    Returns:
        ((train_vectors, train_labels), (test_vectors, test_labels))

    Raises:
        ValueError: on mismatched lengths, too little data to produce a
            non-empty train and test split, or an invalid test_size.
    """
    if len(feature_vectors) != len(labels):
        raise ValueError(
            f"feature_vectors and labels must have the same length: got "
            f"{len(feature_vectors)} feature vector(s) and {len(labels)} label(s)."
        )
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1 (exclusive).")

    n_samples = len(feature_vectors)
    if n_samples < MIN_SAMPLES_FOR_SPLIT:
        raise ValueError(
            f"Insufficient data: at least {MIN_SAMPLES_FOR_SPLIT} labeled examples are "
            f"required for an explicit train/test split, got {n_samples}. Label more "
            f"sessions before splitting, or evaluate on a separately assembled dataset "
            f"instead of an automatic split."
        )

    cleaned_labels = _clean_labels(labels)
    label_counts = Counter(cleaned_labels)
    can_stratify = len(label_counts) >= 2 and all(count >= 2 for count in label_counts.values())

    indices = list(range(n_samples))
    try:
        train_idx, test_idx = _sklearn_train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=cleaned_labels if can_stratify else None,
        )
    except ValueError as error:
        raise ValueError(
            f"Could not split {n_samples} example(s) with test_size={test_size}: {error}"
        ) from error

    if not train_idx or not test_idx:
        raise ValueError(
            f"Insufficient data: test_size={test_size} on {n_samples} example(s) would "
            f"leave an empty train or test split. Use more data or a different test_size."
        )

    train_vectors = tuple(feature_vectors[i] for i in train_idx)
    train_labels = tuple(cleaned_labels[i] for i in train_idx)
    test_vectors = tuple(feature_vectors[i] for i in test_idx)
    test_labels = tuple(cleaned_labels[i] for i in test_idx)

    return (train_vectors, train_labels), (test_vectors, test_labels)


# ============================================================================
# Phase 2C bridge (the ONLY section that imports clustering/labeling)
# ============================================================================


def build_training_set(
    feature_vectors: Sequence[FeatureVector],
    result: "ClusteringResult",  # noqa: F821 -- resolved via TYPE_CHECKING import above
    store: "ContextLabelStore",  # noqa: F821 -- resolved via TYPE_CHECKING import above
    run_id: str | None = None,
) -> tuple[tuple[FeatureVector, ...], tuple[str, ...]]:
    """
    Assemble a labeled training set by joining Phase 2A feature vectors
    with Phase 2C human labels, through the Phase 2B clustering result
    that connects a `session_id` to a `cluster_label`.

    Read the "Phase 2C persistence limitation" section of this module's
    docstring before relying on this function: the caller must supply the
    *same* `ClusteringResult` (or one with identical content, e.g. from
    re-fitting with the same data and `random_state`) that the desired
    labels were originally assigned under. Labels are never guessed,
    inferred, or carried over from a different run.

    Args:
        feature_vectors: Phase 2A feature vectors (any superset is fine;
            only sessions that also appear in `result` and have a
            resolvable label are used).
        result: The Phase 2B `ClusteringResult` whose sessions to look up
            labels for.
        store: Where Phase 2C labels are persisted.
        run_id: Optional precomputed run_id for `result` (see
            `app.ml.context_labeling.compute_run_id`). Computed from
            `result` if omitted.

    Returns:
        (labeled_feature_vectors, labels) -- two parallel tuples,
        containing only the sessions that had a non-None label. The
        returned `FeatureVector` objects are exactly the ones the caller
        supplied, untouched; cluster_label/run_id/the label text are
        never injected into a FeatureVector's `feature_values`.
    """
    from app.ml.context_labeling import session_id_to_label  # local import: bridge-only dependency

    label_by_session = session_id_to_label(result, store, run_id=run_id)

    labeled_vectors: list[FeatureVector] = []
    labeled_texts: list[str] = []
    for vector in feature_vectors:
        label = label_by_session.get(vector.session_id)
        if label is not None:
            labeled_vectors.append(vector)
            labeled_texts.append(label)

    return tuple(labeled_vectors), tuple(labeled_texts)


# ============================================================================
# Internal validation helpers
# ============================================================================


def _feature_vectors_to_matrix(feature_vectors: Sequence[FeatureVector]) -> tuple[np.ndarray, tuple[str, ...]]:
    """Convert a sequence of Phase 2A FeatureVectors into a plain matrix, validating consistency."""
    if not feature_vectors:
        raise ValueError("Insufficient data: no feature vectors were provided.")

    reference_names = feature_vectors[0].feature_names
    for vector in feature_vectors:
        if vector.feature_names != reference_names:
            raise ValueError(
                "Feature dimension mismatch: all feature vectors must share the same "
                f"feature names, in the same order. Got {vector.feature_names} which "
                f"differs from {reference_names}."
            )

    matrix = np.array([vector.feature_values for vector in feature_vectors], dtype=float)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("feature_matrix contains NaN or infinite values.")

    return matrix, reference_names


def _clean_labels(labels: Sequence[str]) -> list[str]:
    """Strip and validate a sequence of user-provided label strings."""
    cleaned = []
    for label in labels:
        if not isinstance(label, str):
            raise ValueError(f"label must be a string, got {type(label).__name__}: {label!r}.")
        stripped = label.strip()
        if not stripped:
            raise ValueError("label must not be empty or whitespace-only.")
        cleaned.append(stripped)
    return cleaned


def _validate_training_input(
    feature_vectors: Sequence[FeatureVector],
    labels: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...], list[str]]:
    """Full validation pipeline for `ContextClassifier.fit`."""
    if len(feature_vectors) != len(labels):
        raise ValueError(
            f"feature_vectors and labels must have the same length: got "
            f"{len(feature_vectors)} feature vector(s) and {len(labels)} label(s)."
        )
    if len(feature_vectors) < MIN_SAMPLES_FOR_TRAINING:
        raise ValueError(
            f"Insufficient training data: at least {MIN_SAMPLES_FOR_TRAINING} labeled "
            f"examples are required, got {len(feature_vectors)}. Label more sessions "
            f"(Phase 2C) before training."
        )

    matrix, feature_names = _feature_vectors_to_matrix(feature_vectors)
    cleaned_labels = _clean_labels(labels)

    distinct_labels = set(cleaned_labels)
    if len(distinct_labels) < MIN_DISTINCT_LABELS_FOR_TRAINING:
        raise ValueError(
            f"Insufficient training data: at least {MIN_DISTINCT_LABELS_FOR_TRAINING} "
            f"distinct labels are required to train a classifier, got "
            f"{len(distinct_labels)} ({sorted(distinct_labels)}). A classifier cannot "
            f"discriminate between contexts from a single label."
        )

    return matrix, feature_names, cleaned_labels