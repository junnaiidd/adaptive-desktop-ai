"""
Phase 2G: Session Context Integration

The first bridge connecting the persisted Phase 1 desktop-session system
to the existing Phase 2A/2F feature-extraction and inference pipeline.

    ActivityRepository (Phase 1, unmodified except for one new read method)
        |
        v
    StoredSession / StoredActivity (Phase 1, unmodified)
        |
        v
    SessionContextIntegrator (Phase 2G -- THIS FILE)
        |
        v
    FeatureExtractor.extract_features() (Phase 2A, unmodified)
        |
        v
    FeatureVector (Phase 2A, unmodified)
        |
        v
    ContextInferenceEngine.predict() (Phase 2F, unmodified)
        |
        v
    PredictionResult (Phase 2D, unmodified)

What this module IS
--------------------
A pure orchestration layer: look up one completed session, validate it,
fetch its activities, hand them to the existing Phase 2A extractor, hand
the resulting `FeatureVector` to the existing Phase 2F engine, and return
whatever `PredictionResult` it produces. No step here recomputes anything
Phase 1/2A/2F already compute.

What this module is NOT
------------------------
- Not a new ML capability: no model, no feature logic, no clustering.
- Not a persistence layer: the returned `PredictionResult` is an
  in-memory inference result only. Nothing here writes a prediction,
  a "context" column, a cache, or a history record anywhere. This
  preserves the distinction between OBSERVED data (activity, session,
  duration -- Phase 1's durable record of what actually happened) and
  INFERRED data (a predicted label and probabilities -- Phase 2G's
  transient output of what the model currently *thinks* that means).
- Not a training or retraining system: this module never calls `fit()`,
  never touches `TrainingExampleStore`/`capture_labeled_examples`/
  `ContextLabelStore`/`retrain_from_examples`.
- Not live/windowed inference: exactly one inference per call, for one
  already-completed session. No background workers, timers, polling,
  partial-session handling, sliding windows, debouncing, smoothing, or
  confidence thresholds. A future phase may build a live/windowed layer
  *on top of* this one-shot integrator; this module intentionally leaves
  that room by staying narrow rather than trying to anticipate it.
- Not a "context" abstraction: this module does not introduce `Context`,
  `WorkThread`, `Task`, `Deadline`, or `Workspace` concepts. A session is
  an observed activity boundary; a `PredictionResult` is an inferred
  label for that boundary. The two are deliberately kept distinct --
  nothing here conflates "session" with "context."

Completed sessions only
------------------------
A session whose `StoredSession.ended_at` is `None` is still active/open
and is never passed to `FeatureExtractor`. This is checked explicitly by
this module (see `SessionNotCompleteError` below) rather than relying on
`FeatureExtractor.extract_features` to reject it -- that method's own
`ValueError` for a missing `session_ended_at` exists for a different
reason (constructing a `FeatureVector` at all requires two timestamps)
and its message ("Session times must be timezone-aware") would be a
confusing, indirect way to report "this session hasn't ended yet."

No-activity sessions
---------------------
Phase 2A's `FeatureExtractor` can, by itself, gracefully handle a session
with zero activities: it returns a defined (mostly-zero) `FeatureVector`
rather than raising. Phase 2G deliberately does NOT rely on that path.
A prediction computed from an all-zero feature vector reflects no actual
observed behavior -- returning it as if it were a meaningful inference
would look like manufactured output even though no single step
technically "faked" anything. This module therefore rejects a completed,
zero-activity session before ever calling `FeatureExtractor`, with a
distinct, explicit error (see `SessionHasNoActivitiesError`).

Error contract
--------------
This project has no existing custom-exception hierarchy anywhere in
Phase 1-2F -- every phase raises plain `ValueError`/`RuntimeError`/
`TypeError`/`FileNotFoundError` for precondition and validation failures.
Phase 2G follows that established convention rather than introducing a
new exception base class. Three small, `ValueError` subclasses are
defined below purely so callers can distinguish the three
Phase-2G-specific precondition failures (session missing / incomplete /
empty) with `except SessionLookupError` or a specific subtype, while
still being caught by ordinary `except ValueError` code that doesn't
care about the distinction -- consistent with how the rest of the
project already uses `ValueError` for "this input isn't valid for this
operation." No exception is introduced for feature-extraction or
inference failures: those propagate completely unmodified from Phase 2A
(`ValueError`) and Phase 2F/2D (`RuntimeError` for an untrained
classifier, `ValueError` for a feature-schema mismatch) -- this module
does not wrap, translate, or catch them.
"""

from __future__ import annotations

from app.database.activity_repository import ActivityRepository, StoredSession
from app.ml.context_classifier import PredictionResult
from app.ml.context_inference import ContextInferenceEngine
from app.ml.feature_engineering import FeatureExtractor


class SessionLookupError(ValueError):
    """Base class for the Phase-2G-specific session-precondition failures below."""


class SessionNotFoundError(SessionLookupError):
    """Raised when the requested session_id does not exist in the repository."""


class SessionNotCompleteError(SessionLookupError):
    """Raised when the requested session exists but has not ended yet (ended_at is None)."""


class SessionHasNoActivitiesError(SessionLookupError):
    """Raised when the requested session is complete but has no recorded activity segments."""


class SessionContextIntegrator:
    """
    Orchestrates: completed-session lookup -> feature extraction -> context inference.

    Owns references to the three existing components it bridges -- it
    implements none of their behavior itself.
    """

    def __init__(
        self,
        repository: ActivityRepository,
        feature_extractor: FeatureExtractor,
        inference_engine: ContextInferenceEngine,
    ) -> None:
        """
        Args:
            repository: Where completed sessions/activities are read from.
            feature_extractor: Phase 2A's extractor (unmodified) used to
                turn a session's activities into a `FeatureVector`.
            inference_engine: Phase 2F's engine (unmodified) used to turn
                that `FeatureVector` into a `PredictionResult`.
        """
        self.repository = repository
        self.feature_extractor = feature_extractor
        self.inference_engine = inference_engine

    def predict_for_session(self, session_id: str) -> PredictionResult:
        """
        Run one inference for one already-completed session.

        Args:
            session_id: The id of a session that has already ended.

        Returns:
            The `PredictionResult` Phase 2F's `ContextInferenceEngine`
            produces -- returned exactly as-is, never wrapped.

        Raises:
            SessionNotFoundError: no session with this id exists.
            SessionNotCompleteError: the session exists but `ended_at` is
                still `None` (it hasn't ended yet).
            SessionHasNoActivitiesError: the session is complete but has
                zero recorded activity segments.
            ValueError: propagated unmodified from
                `FeatureExtractor.extract_features` (e.g. a malformed
                timestamp) or from `ContextInferenceEngine.predict` (a
                feature-schema mismatch against the trained classifier).
            RuntimeError: propagated unmodified from
                `ContextInferenceEngine.predict` if the wrapped
                classifier has not been trained.
        """
        session = self._get_completed_session(session_id)
        activities = self.repository.list_activities_for_session(session_id)
        if not activities:
            raise SessionHasNoActivitiesError(
                f"Session '{session_id}' is complete but has no recorded activity "
                f"segments; there is nothing to extract features from."
            )

        feature_vector = self.feature_extractor.extract_features(
            session_id=session.id,
            session_started_at=session.started_at,
            session_ended_at=session.ended_at,
            activities=activities,
        )

        return self.inference_engine.predict(feature_vector)

    def _get_completed_session(self, session_id: str) -> StoredSession:
        matching = [session for session in self.repository.list_sessions() if session.id == session_id]
        if not matching:
            raise SessionNotFoundError(f"No session found with id '{session_id}'.")

        session = matching[0]
        if session.ended_at is None:
            raise SessionNotCompleteError(
                f"Session '{session_id}' has not ended yet (ended_at is None); "
                f"only completed sessions can be used for inference."
            )
        return session
