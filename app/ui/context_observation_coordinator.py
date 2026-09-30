"""
Milestone A ("See Your Context"): Context Observation Coordinator.

The single piece of new orchestration this milestone introduces. It
closes the wiring gap identified in the post-Phase-2H reconnaissance:
`SessionContextIntegrator` and `ContextObservationStore` existed and were
fully tested, but nothing in the running application ever called them.

    ActivityRepository.list_sessions()
            |
            v
    ContextObservationCoordinator.run_once()
            |  for each completed (ended_at is not None), not-yet-observed session:
            v
    SessionContextIntegrator.predict_for_session()   (Phase 2G, unmodified)
            v
    PredictionResult   (Phase 2D/2F, unmodified)
            v
    ContextObservationStore.record_observation()   (Phase 2H, unmodified)

This module is deliberately Qt-independent (no PySide6 import) so its
core logic is directly unit-testable with plain pytest, matching
`DashboardController`'s existing precedent in this same package. A thin
`QThread` wrapper (`context_inference_worker.py`) is what actually calls
`run_once()` periodically from the running application; this file knows
nothing about threads, timers, or the UI.

Why polling, not a push signal
--------------------------------
`SessionManager`/`ActivityMonitoringService` were inspected directly:
neither exposes any callback, signal, or return value indicating "a
session just completed" to an external caller -- `poll_once()` consumes
that information entirely internally. Per this milestone's explicit
architectural rule, `SessionManager`/`MonitoringService` are not modified
to add one. Polling `ActivityRepository.list_sessions()` (a read-only,
already-existing, Phase 1 API) and filtering on `ended_at is not None` is
therefore the only available way to detect completion without reversing
the dependency direction between Phase 1 and the ML layer.

Duplicate-session handling
----------------------------
The durable, restart-safe source of truth for "has this session already
been observed" is `ContextObservationStore.list_observations_for_session`
-- not any in-memory set. A session with at least one existing
observation row is always skipped. `ContextObservationStore` is
append-only by design (Phase 2H), so this check is what actually prevents
duplicate inference; there is no upsert to rely on instead.

Model availability
--------------------
`ContextInferenceEngine.load()` is attempted lazily, once, on the first
`run_once()` call. If no model artifact exists yet (`FileNotFoundError`,
the expected state for a fresh install with nothing trained yet), that
outcome is recorded and every subsequent `run_once()` call returns
immediately without retrying the load -- so a missing model produces one
clean, one-time outcome rather than a failure every 30 seconds forever.

Failure isolation
-------------------
Every per-session inference attempt is wrapped individually. One
session's failure (an unusual edge case such as `SessionHasNoActivitiesError`,
or a feature-schema mismatch) never stops the batch: the coordinator
records that outcome and continues to the next session. `run_once()`
itself never raises for an individual session's failure; it always
returns a `CoordinatorRunResult` summarizing what happened. Nothing here
can corrupt `sessions`/`activity_segments`, since this module only ever
reads from `ActivityRepository` and only ever writes through
`ContextObservationStore`'s own already-transactional `record_observation`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.database.activity_repository import ActivityRepository
from app.ml.context_inference import ContextInferenceEngine
from app.ml.context_observation_store import ContextObservationStore
from app.ml.feature_engineering import FeatureExtractor
from app.ml.session_context_integration import SessionContextIntegrator


@dataclass(frozen=True, slots=True)
class SessionObservationOutcome:
    """What happened when the coordinator considered one session."""

    session_id: str
    recorded: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CoordinatorRunResult:
    """A summary of one `run_once()` call. Never raised; always returned."""

    model_available: bool
    outcomes: tuple[SessionObservationOutcome, ...] = field(default_factory=tuple)

    @property
    def recorded_count(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.recorded)

    @property
    def failed_count(self) -> int:
        return sum(1 for outcome in self.outcomes if not outcome.recorded)


class ContextObservationCoordinator:
    """
    Finds completed sessions with no context observation yet, infers
    context for each via the existing, unmodified Phase 2G/2F/2D
    pipeline, and persists the result via the existing, unmodified
    Phase 2H store.

    Contains no Qt dependency and performs no threading of its own --
    `run_once()` is a plain, synchronous, fully-testable method. A
    caller (in practice, `ContextInferenceWorker`) decides when and how
    often to call it.
    """

    def __init__(
        self,
        repository: ActivityRepository,
        observation_store: ContextObservationStore,
        model_path: str | Path | None = None,
    ) -> None:
        """
        Args:
            repository: Where completed sessions/activities are read from.
            observation_store: Where inferred context is durably persisted.
            model_path: Optional override for the trained classifier's
                saved location. Defaults to
                `ContextInferenceEngine.load`'s own default
                (`models/context_classifier.joblib`) when omitted.
        """
        self.repository = repository
        self.observation_store = observation_store
        self._model_path = model_path
        self._integrator: SessionContextIntegrator | None = None
        self._model_load_attempted = False
        self._model_version: str | None = None

    @property
    def model_available(self) -> bool:
        """Whether a trained model has been (successfully) loaded so far."""
        return self._integrator is not None

    def run_once(self) -> CoordinatorRunResult:
        """
        Consider every completed session once: skip already-observed
        ones, infer and persist for the rest. Never raises for an
        individual session's failure or for a missing model -- both are
        reflected in the returned `CoordinatorRunResult` instead.
        """
        self._ensure_model_loaded()
        if self._integrator is None:
            return CoordinatorRunResult(model_available=False, outcomes=())

        outcomes: list[SessionObservationOutcome] = []
        for session in self.repository.list_sessions():
            if session.ended_at is None:
                continue  # still active; never attempt inference on it
            if self.observation_store.list_observations_for_session(session.id):
                continue  # already observed; append-only store, never re-infer

            outcomes.append(self._observe_one_session(session.id))

        return CoordinatorRunResult(model_available=True, outcomes=tuple(outcomes))

    def _observe_one_session(self, session_id: str) -> SessionObservationOutcome:
        assert self._integrator is not None  # guaranteed by run_once's guard above
        try:
            prediction = self._integrator.predict_for_session(session_id)
            self.observation_store.record_observation(
                prediction.session_id,
                prediction.predicted_label,
                prediction.class_probabilities,
                self._model_version or "unknown",
                observed_at=datetime.now(timezone.utc),
            )
            return SessionObservationOutcome(session_id=session_id, recorded=True)
        except Exception as error:  # noqa: BLE001 -- deliberate: isolate any single session's failure
            return SessionObservationOutcome(session_id=session_id, recorded=False, error=str(error))

    def _ensure_model_loaded(self) -> None:
        if self._model_load_attempted:
            return
        self._model_load_attempted = True
        try:
            engine = ContextInferenceEngine.load(self._model_path)
        except FileNotFoundError:
            return  # no model trained yet; model_available stays False, no retry
        self._integrator = SessionContextIntegrator(self.repository, FeatureExtractor(), engine)
        self._model_version = _derive_model_version(engine)


def _derive_model_version(engine: ContextInferenceEngine) -> str:
    """
    The exact caller-side formula Phase 2H's own documentation specifies:
    `schema{V}-rs{R}-n{N}-depth{D}`, built from the wrapped
    `ContextClassifier`'s already-public configuration. No modification
    to Phase 2D/2F was made or needed to produce this.
    """
    classifier = engine.classifier
    return (
        f"schema{classifier.SCHEMA_VERSION}-rs{classifier.random_state}"
        f"-n{classifier.n_estimators}-depth{classifier.max_depth}"
    )
