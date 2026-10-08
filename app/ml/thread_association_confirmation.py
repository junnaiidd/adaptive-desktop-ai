"""
Work Thread association confirmation contract: what a future "Possible workspace detected" prompt consumes
and emits.

    AssociationDecision (ranking) --suggestion_from_decision--> AssociationSuggestion      (advisory data)
    AssociationSuggestion --[user clicks Confirm]--> AssociationConfirmer.confirm --> AssociationFeedback
    AssociationSuggestion --[user clicks Not this]-> AssociationConfirmer.reject  --> AssociationFeedback

What this module IS
-------------------
The seam between the association engine and any user interface. A prompt needs to show which Work Thread is
suggested, why, and which applications are involved; it then needs exactly two explicit actions. This module
defines those data shapes and the two actions. It contains no UI and no Qt, so the corner notification can be
built later without changing anything here.

Confirmation is the ONLY place anything is written
--------------------------------------------------
Producing a suggestion never writes. ``AssociationConfirmer.confirm`` is the single write path: it records the
user's explicit decision by calling the EXISTING ``WorkThreadStore.associate_observation`` - no new table, no
direct SQL. It is idempotent: confirming an observation that is already linked to that thread returns the
existing association instead of recording a duplicate. ``reject`` writes nothing at all.

A confirmed association is exactly the high-quality, user-confirmed label that a later learned ranker can be
trained on. Because the suggestion carries the ``ThreadAssociationFeatures`` it was scored with, and both
outcomes are returned as an immutable ``AssociationFeedback``, a future milestone can turn feedback into
supervised training rows (features -> confirmed / rejected) without changing this contract.

What this module is NOT
-----------------------
- Not persistent for rejections: a rejection is returned as data and is NOT stored. Remembering "the user said
  not this" (a negative training example, or suppressing repeat prompts) needs a schema and is deliberately
  deferred; the contract is shaped so adding it later does not change callers.
- Not automatic: nothing here is ever invoked by scoring, polling or a timer; only an explicit user action
  calls ``confirm`` or ``reject``.
- Not a scorer, not a ranker, not a notification: it neither decides nor displays.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from app.ml.thread_association_features import SessionAssociationContext, ThreadAssociationFeatures
from app.ml.thread_association_ranking import AssociationDecision

if TYPE_CHECKING:
    from app.ml.work_thread_store import WorkThreadObservation, WorkThreadStore


@dataclass(frozen=True, slots=True)
class AssociationSuggestion:
    """An advisory suggestion that a session belongs to a Work Thread. It is NOT an association."""

    session_id: str
    observation_id: int | None  # the persisted observation a confirmation would link; None = cannot be confirmed
    work_thread_id: int
    work_thread_name: str
    association_score: float  # a ranking signal, not a probability
    reasons: tuple[str, ...]  # grounded in real feature contributions
    applications: tuple[str, ...]  # the session's applications (what a prompt shows)
    matched_applications: tuple[str, ...]  # those that were also used in the thread's history
    features: ThreadAssociationFeatures  # the exact inputs the score was computed from (training-ready)
    as_of: datetime  # the decision time the history was evaluated at


class AssociationFeedbackKind(Enum):
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class AssociationFeedback:
    """The user's explicit decision about one suggestion. ``association`` is set only when confirmed."""

    suggestion: AssociationSuggestion
    kind: AssociationFeedbackKind
    decided_at: datetime  # UTC
    association: "WorkThreadObservation | None" = None


def suggestion_from_decision(
    decision: AssociationDecision, context: SessionAssociationContext
) -> AssociationSuggestion | None:
    """Build a suggestion from a decision, or return ``None`` when the engine abstained."""
    top = decision.suggested
    if top is None:
        return None
    return AssociationSuggestion(
        session_id=context.session_id,
        observation_id=context.observation_id,
        work_thread_id=top.work_thread.id,
        work_thread_name=top.work_thread.name,
        association_score=top.association_score,
        reasons=top.reasons,
        applications=context.applications,
        matched_applications=tuple(name for name, _count in top.detail.matched_applications),
        features=top.features,
        as_of=context.as_of,
    )


class AssociationConfirmer:
    """Records an explicit user decision. The only component in the association engine that can write."""

    def __init__(self, work_thread_store: WorkThreadStore) -> None:
        self._threads = work_thread_store

    def confirm(self, suggestion: AssociationSuggestion, *, confirmed_at: datetime) -> AssociationFeedback:
        """
        Link the suggested observation to the suggested Work Thread, via the existing store. Idempotent.

        Raises:
            ValueError: ``confirmed_at`` is timezone-naive, or the suggestion has no persisted observation.
            sqlite3.IntegrityError: propagated unchanged from the store (e.g. the Work Thread was deleted).
        """
        _require_aware(confirmed_at, "confirmed_at")
        if suggestion.observation_id is None:
            raise ValueError("This suggestion has no persisted context observation, so it cannot be confirmed.")

        for existing in self._threads.list_observations_for_work_thread(suggestion.work_thread_id):
            if existing.observation_id == suggestion.observation_id:
                return AssociationFeedback(suggestion, AssociationFeedbackKind.CONFIRMED, confirmed_at, existing)

        association = self._threads.associate_observation(
            suggestion.work_thread_id, suggestion.observation_id, associated_at=confirmed_at
        )
        return AssociationFeedback(suggestion, AssociationFeedbackKind.CONFIRMED, confirmed_at, association)

    def reject(self, suggestion: AssociationSuggestion, *, rejected_at: datetime) -> AssociationFeedback:
        """Record "Not this" as returned data only. Writes nothing (see the module docstring)."""
        _require_aware(rejected_at, "rejected_at")
        return AssociationFeedback(suggestion, AssociationFeedbackKind.REJECTED, rejected_at, None)


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware (UTC); got a naive datetime.")
