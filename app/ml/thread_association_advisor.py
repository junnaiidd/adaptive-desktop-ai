"""
Work Thread association advisor: one completed session's context -> an advisory assessment.

    persisted context observation (existing; produced by the unmodified inference pipeline)
        |  + the session's sanitised application names (existing activity data)
        v
    SessionAssociationContext
        |  + WorkThreadProfileBuilder (reads existing stores)
        v
    WorkThreadCandidateRanker  ->  AssociationDecision  ->  AssociationSuggestion | None

What this module IS
-------------------
A thin, single-purpose orchestrator with one public method. It only wires existing components together; every
rule it relies on (what counts as history, how candidates are scored, when to abstain) lives in the module
that owns it. It is the entry point a future notification would call after a session's context is observed.

What this module is NOT
-----------------------
- Not an association: ``assess`` is advisory and never writes. Linking an observation to a Work Thread happens
  only through ``AssociationConfirmer.confirm`` after an explicit user decision.
- Not a new inference: it consumes the existing context observation; it never runs, re-runs or retrains the
  classifier, and it does not touch ``feature_engineering``, ``context_classifier`` or ``context_inference``.
- Not a timer or poller: it runs when called. It never decides *when* to prompt, so it cannot nag.
- Not a clock reader: the decision time ``as_of`` is a required keyword argument. The caller supplies "now"
  in live use, or the original time when replaying a past decision, so identical inputs give identical output.

Leakage
-------
The session being assessed is excluded from every thread's evidence by the feature extractor, so confirming
the very association being predicted cannot change how it was (or would be) scored.

Errors
------
``ValueError`` if the observation does not exist, if ``as_of`` is timezone-naive, or if the observation's
probabilities do not contain its own predicted label (malformed data is rejected, not repaired).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from app.ml.thread_association_confirmation import AssociationSuggestion, suggestion_from_decision
from app.ml.thread_association_features import SessionAssociationContext
from app.ml.thread_association_ranking import AssociationDecision, WorkThreadCandidateRanker
from app.ml.work_thread_profile import WorkThreadProfileBuilder, application_name

if TYPE_CHECKING:
    from app.database.activity_repository import ActivityRepository
    from app.ml.context_observation_store import ContextObservationStore
    from app.ml.work_thread_store import WorkThreadStore


@dataclass(frozen=True, slots=True)
class AssociationAssessment:
    """Everything one assessment produced: the input context, the full ranking, and the suggestion if any."""

    context: SessionAssociationContext
    decision: AssociationDecision
    suggestion: AssociationSuggestion | None  # None whenever the engine abstained


class WorkThreadAssociationAdvisor:
    def __init__(
        self,
        work_thread_store: WorkThreadStore,
        context_observation_store: ContextObservationStore,
        activity_repository: ActivityRepository,
        *,
        ranker: WorkThreadCandidateRanker | None = None,
    ) -> None:
        self._observations = context_observation_store
        self._activity = activity_repository
        self._profiles = WorkThreadProfileBuilder(work_thread_store, context_observation_store, activity_repository)
        self._ranker = ranker or WorkThreadCandidateRanker()

    def assess(self, observation_id: int, *, as_of: datetime) -> AssociationAssessment:
        """Rank every Work Thread for the session behind ``observation_id``. Read-only and advisory."""
        observation = self._observations.get_observation(observation_id)
        if observation is None:
            raise ValueError(f"No context observation found with id {observation_id}.")
        if observation.predicted_label not in observation.class_probabilities:
            raise ValueError(
                f"Observation {observation_id} has no probability for its own predicted label "
                f"'{observation.predicted_label}'."
            )

        applications = tuple(
            sorted(
                {
                    name
                    for segment in self._activity.list_activities_for_session(observation.session_id)
                    if (name := application_name(segment.application, segment.process_name)) is not None
                }
            )
        )
        context = SessionAssociationContext(
            session_id=observation.session_id,
            context_label=observation.predicted_label,
            context_probability=observation.class_probabilities[observation.predicted_label],
            as_of=as_of,
            applications=applications,
            observation_id=observation.id,
        )
        decision = self._ranker.rank(context, self._profiles.build_profiles())
        return AssociationAssessment(
            context=context, decision=decision, suggestion=suggestion_from_decision(decision, context)
        )
