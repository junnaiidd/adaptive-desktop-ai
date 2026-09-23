"""
Phase 2E: Deterministic Retraining Pipeline

Orchestrates retraining Phase 2D's `ContextClassifier` from the durable
corpus `training_examples.py` builds, without introducing any new ML
logic. Every model-touching line in this file is a direct, unmodified
call into Phase 2D -- this module contains NO second RandomForest
implementation and NO duplicated training/evaluation validation logic.

    Phase 2E (training_examples.py): durable LabeledExample corpus
            │
            ▼
    Phase 2E (this file): reconstruct FeatureVectors, decide split/no-split,
            call Phase 2D's fit()/evaluate()/save() exactly as they exist
            │
            ▼
    RetrainResult: Phase 2D's own TrainingSummary + EvaluationResult|None

This file has NO dependency on Phase 2B or Phase 2C -- it never touches
`ClusteringResult`, `ContextLabelStore`, or run/cluster concepts. It only
ever sees the already-captured, already-labeled corpus.

Minimum data / insufficient-data behavior
-------------------------------------------
No new thresholds are invented here. `MIN_SAMPLES_FOR_SPLIT` is Phase
2D's own constant, imported unmodified, and used only to decide whether a
held-out split is attempted:

  - 0 examples: this module raises `ValueError` itself, before touching
    `ContextClassifier` at all.
  - >=1 but < MIN_SAMPLES_FOR_SPLIT (4): no split is attempted. The full
    corpus is passed to `ContextClassifier.fit()`. `evaluation` is `None`
    and `evaluation_skipped_reason` explains why -- training-set accuracy
    is never substituted for a held-out evaluation. If the corpus still
    doesn't meet Phase 2D's OWN `fit()` minimums (`MIN_SAMPLES_FOR_TRAINING`,
    `MIN_DISTINCT_LABELS_FOR_TRAINING`), that `ValueError` propagates
    unmodified from `fit()` -- this module does not re-implement or
    duplicate that validation.
  - >= MIN_SAMPLES_FOR_SPLIT (4): `train_test_split_feature_vectors`
    (Phase 2D, unmodified) is used; `fit()` on train, `evaluate()` on
    test.

Determinism
-----------
Identical corpus + identical `random_state` -> identical `TrainingSummary`
and `EvaluationResult`, inherited entirely from Phase 2D's own
deterministic `fit`/`evaluate`/split. No randomness is introduced here.

Model replacement
------------------
Every retrain is a full, from-scratch replacement over the ENTIRE current
corpus -- never an incremental/online update. If a `classifier` instance
is passed in, it is retrained in place (Phase 2D's `fit()` itself
overwrites any previous model state); if omitted, a new `ContextClassifier`
is constructed.

Failure safety
--------------
`ContextClassifier.save()` is only ever called AFTER a successful `fit()`
(and `evaluate()`, when a split occurred). This module never catches or
suppresses exceptions from Phase 2D -- `ValueError`/`RuntimeError`
propagate unmodified. Consequently, a failed retrain cannot overwrite or
corrupt a previously saved, working model artifact: the code path that
would overwrite it is never reached.

Model artifact location
------------------------
`save_path` defaults to Phase 2D's own `ContextClassifier.DEFAULT_MODEL_PATH`
(`models/context_classifier.joblib`) when omitted, via `ContextClassifier.save(None)`.
This module introduces no alternate naming scheme (no content hashing, no
versioned filenames) -- the default artifact location is unchanged from
Phase 2D.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.ml.context_classifier import (
    MIN_SAMPLES_FOR_SPLIT,
    ContextClassifier,
    EvaluationResult,
    TrainingSummary,
    train_test_split_feature_vectors,
)
from app.ml.feature_engineering import FeatureVector
from app.ml.training_examples import LabeledExample, TrainingExampleStore


@dataclass(frozen=True, slots=True)
class RetrainResult:
    """The outcome of one `retrain_from_examples` call."""

    training_summary: TrainingSummary
    evaluation: EvaluationResult | None
    evaluation_skipped_reason: str | None
    n_examples_used: int
    saved_to: Path | None
    random_state: int


def retrain_from_examples(
    examples_store: TrainingExampleStore,
    classifier: ContextClassifier | None = None,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
    save: bool = True,
    save_path: str | Path | None = None,
) -> RetrainResult:
    """
    Retrain a `ContextClassifier` from the complete durable corpus in
    `examples_store`.

    Args:
        examples_store: The corpus to train from. `list_examples()` is
            read once, in full -- every currently-stored example is used.
        classifier: An existing `ContextClassifier` to retrain in place
            (its previous model state is fully overwritten by `fit()`).
            If omitted, a new `ContextClassifier(random_state=random_state)`
            is constructed.
        test_size: Passed through to `train_test_split_feature_vectors`
            when a split is attempted (corpus size >= `MIN_SAMPLES_FOR_SPLIT`).
        random_state: Seed for both the split and the classifier, for
            full determinism. Ignored for the split/classifier's own
            seeding if an already-configured `classifier` is passed in
            (its own `random_state` governs `fit()`); still used for
            `train_test_split_feature_vectors`.
        save: Whether to call `ContextClassifier.save()` after a
            successful fit (and evaluate, if applicable).
        save_path: Passed through to `ContextClassifier.save()`. Omit to
            use Phase 2D's own default (`models/context_classifier.joblib`).

    Returns:
        A RetrainResult wrapping Phase 2D's own `TrainingSummary` and
        `EvaluationResult` (or `None`, with an explanation) unmodified.

    Raises:
        ValueError: if the corpus is empty, if examples in the corpus
            have inconsistent `feature_names` (mixed feature schemas), or
            (propagated unmodified from Phase 2D) if the corpus is too
            small or has too few distinct labels to train at all.
    """
    examples = examples_store.list_examples()

    if not examples:
        raise ValueError(
            "No labeled training examples are available yet. Capture some "
            "with app.ml.training_examples.capture_labeled_examples before "
            "retraining."
        )

    _require_consistent_feature_schema(examples)

    feature_vectors = [
        FeatureVector(
            session_id=example.session_id,
            feature_names=example.feature_names,
            feature_values=example.feature_values,
            metadata={},
        )
        for example in examples
    ]
    labels = [example.label for example in examples]

    resolved_classifier = classifier if classifier is not None else ContextClassifier(random_state=random_state)

    if len(examples) < MIN_SAMPLES_FOR_SPLIT:
        # Too small for a meaningful held-out split. Train on the full
        # corpus and report evaluation as unavailable -- never fabricate
        # a "held-out" score from training data itself.
        training_summary = resolved_classifier.fit(feature_vectors, labels)
        evaluation = None
        evaluation_skipped_reason = (
            f"Insufficient data for a held-out split (need >= {MIN_SAMPLES_FOR_SPLIT} "
            f"examples, have {len(examples)}); trained on the full corpus without evaluation."
        )
    else:
        (train_vectors, train_labels), (test_vectors, test_labels) = train_test_split_feature_vectors(
            feature_vectors, labels, test_size=test_size, random_state=random_state
        )
        training_summary = resolved_classifier.fit(train_vectors, train_labels)
        evaluation = resolved_classifier.evaluate(test_vectors, test_labels)
        evaluation_skipped_reason = None

    saved_to: Path | None = None
    if save:
        # Only reached after fit() (and evaluate(), if applicable) have
        # already succeeded -- see "Failure safety" in the module docstring.
        saved_to = resolved_classifier.save(save_path)

    return RetrainResult(
        training_summary=training_summary,
        evaluation=evaluation,
        evaluation_skipped_reason=evaluation_skipped_reason,
        n_examples_used=len(examples),
        saved_to=saved_to,
        random_state=random_state,
    )


def _require_consistent_feature_schema(examples: tuple[LabeledExample, ...]) -> None:
    """
    Ensure every example in the corpus shares an identical `feature_names`
    tuple before anything is handed to `ContextClassifier`.

    Detects internal corpus inconsistency (e.g. Phase 2A's FEATURE_NAMES
    changed between two capture sessions, leaving old and new examples
    with different feature schemas mixed in the same store). This is
    deliberately NOT a check against the currently-live
    `FeatureExtractor.FEATURE_NAMES` -- that would require re-deriving
    features from raw activity data, which examples don't retain, and is
    out of scope for this phase.
    """
    reference_names = examples[0].feature_names
    if all(example.feature_names == reference_names for example in examples):
        return

    groups: dict[tuple[str, ...], int] = {}
    for example in examples:
        groups[example.feature_names] = groups.get(example.feature_names, 0) + 1

    group_descriptions = ", ".join(f"{count} example(s) with {names}" for names, count in groups.items())
    raise ValueError(
        f"Feature schema mismatch in the training-example corpus: found "
        f"{len(groups)} distinct feature_names group(s) -- {group_descriptions}. "
        f"All examples used in one retrain must share an identical feature "
        f"schema. This usually means Phase 2A's feature set changed between "
        f"captures; old examples are not automatically migrated."
    )
