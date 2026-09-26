"""
Phase 2F: Context Inference

A thin bridge from an already-built Phase 2A `FeatureVector` to a
prediction, using the existing, unmodified Phase 2D `ContextClassifier`.

    FeatureVector (2A)
        |
        v
    ContextInferenceEngine (2F)  -- delegates, does not duplicate
        |
        v
    ContextClassifier.predict()/.predict_many() (2D, unmodified)
        |
        v
    PredictionResult (2D, unmodified)

What this module IS
--------------------
A narrow construction/loading convenience plus direct delegation to
`ContextClassifier`'s own `predict`/`predict_many`. Every behavior a
caller can observe here -- trained-state checking, feature-schema
validation, the returned result type, determinism, error types -- is
Phase 2D's own behavior, unchanged. `ContextInferenceEngine` adds no
prediction logic of its own.

What this module is NOT
------------------------
- Not a second classifier: no `RandomForestClassifier` is constructed
  here, and no prediction math happens here.
- Not a second `PredictionResult` type: Phase 2D's `PredictionResult` is
  returned exactly as `ContextClassifier.predict`/`predict_many` produce
  it, never wrapped or copied into a new dataclass.
- Not a second feature validator: feature-name/order/count mismatches are
  detected by `ContextClassifier._validate_feature_vector_matches_model`
  (2D, unchanged) when `predict`/`predict_many` is called; this module
  performs no schema checking of its own.
- Not a training or retraining system: this module never calls `fit()`,
  never touches `TrainingExampleStore`, `capture_labeled_examples`,
  `ContextLabelStore`, or `retrain_from_examples`. Phase 2E (training)
  and Phase 2F (inference) are deliberately separate; nothing here feeds
  a prediction back into the training corpus.
- Not a clustering or labeling dependency: this module imports nothing
  from `app.ml.context_clustering` or `app.ml.context_labeling`. Ordinary
  supervised inference on an already-trained classifier needs neither --
  Phase 2B/2C exist to help produce labeled *training* data, not to
  participate in each inference call.
- Not a persistence layer: no prediction is written to disk, to a
  database, or to any cache. The `PredictionResult` is returned to the
  caller and this module keeps no record of it.
- Not connected to live monitoring: no timers, background workers, or
  polling. A caller (a future, later phase) decides when to invoke
  inference.

Anti-leakage
------------
Only `FeatureVector.feature_values` (validated against `feature_names`)
ever reaches the classifier, exactly as Phase 2D already guarantees.
`session_id` is passed through into the returned `PredictionResult`
purely for attribution, never as a feature. `metadata`, cluster IDs, run
IDs, and training provenance are never read by this module at all -- it
has no code path that could access them, since it never receives
anything but a `FeatureVector` and (at construction time) a
`ContextClassifier`.

Determinism
-----------
Inherited entirely from Phase 2D: a trained `RandomForestClassifier`'s
`predict`/`predict_proba` are deterministic given fixed model state and
input. This module introduces no randomness, no seeds, and no
new sources of non-determinism.

Error handling
--------------
No error is caught, translated, or suppressed here. Constructing the
engine with a `classifier` that is not a `ContextClassifier` raises
`TypeError` immediately (a plain Python type guard, not ML validation).
Beyond that:

  - An untrained classifier: `ContextClassifier.predict`/`predict_many`
    raise `RuntimeError` (2D's own `_require_trained`); this module does
    not duplicate that check at construction time, so the exact same
    error a caller would get from `ContextClassifier` directly is what
    they get through this module too.
  - A feature-schema mismatch: `ContextClassifier.predict`/`predict_many`
    raise `ValueError` (2D's own `_validate_feature_vector_matches_model`).
  - A missing/unreadable model artifact: `ContextInferenceEngine.load`
    delegates directly to `ContextClassifier.load`, which raises
    `FileNotFoundError`. No silent retraining, no fallback prediction,
    no fabricated "Unknown" label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from app.ml.context_classifier import ContextClassifier, PredictionResult
from app.ml.feature_engineering import FeatureVector


class ContextInferenceEngine:
    """
    A thin holder around an already-trained (or loadable) `ContextClassifier`,
    exposing prediction only. See the module docstring for what this
    deliberately does not do.
    """

    def __init__(self, classifier: ContextClassifier) -> None:
        """
        Args:
            classifier: An existing `ContextClassifier`. It does not need
                to be trained yet at construction time -- `predict`/
                `predict_many` will raise Phase 2D's own `RuntimeError`
                if it isn't, exactly as calling the classifier directly
                would.

        Raises:
            TypeError: if `classifier` is not a `ContextClassifier`
                instance (a plain type guard; not ML validation).
        """
        if not isinstance(classifier, ContextClassifier):
            raise TypeError(
                f"ContextInferenceEngine requires an app.ml.context_classifier."
                f"ContextClassifier instance, got {type(classifier).__name__}."
            )
        self.classifier = classifier

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ContextInferenceEngine":
        """
        Load a previously-saved classifier and wrap it for inference.

        This is a direct pass-through to `ContextClassifier.load` -- no
        alternate joblib loading, no new model file format, no
        hash-based or timestamped filenames. `path` defaults to Phase
        2D's own `ContextClassifier.DEFAULT_MODEL_PATH`
        (`models/context_classifier.joblib`) exactly as
        `ContextClassifier.load(None)` already does.

        Raises:
            FileNotFoundError: if no artifact exists at the resolved
                path (propagated unchanged from `ContextClassifier.load`).
        """
        return cls(ContextClassifier.load(path))

    @property
    def is_trained(self) -> bool:
        """Pass-through to the wrapped classifier's own `is_trained`."""
        return self.classifier.is_trained

    def predict(self, feature_vector: FeatureVector) -> PredictionResult:
        """
        Predict the context for one session's feature vector.

        A direct, unmodified call to `ContextClassifier.predict`. See
        that method's docstring (app/ml/context_classifier.py) for the
        exact trained-state and feature-schema validation performed and
        the exact errors raised; none of it is duplicated here.
        """
        return self.classifier.predict(feature_vector)

    def predict_many(self, feature_vectors: Sequence[FeatureVector]) -> tuple[PredictionResult, ...]:
        """
        Predict for several feature vectors at once.

        A direct, unmodified call to `ContextClassifier.predict_many`.
        See `predict`.
        """
        return self.classifier.predict_many(feature_vectors)
