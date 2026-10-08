"""
Work Thread association features: describing ONE new session against ONE existing Work Thread.

    current session context  +  Work Thread profile
                    |
                    v
        ThreadAssociationFeatureExtractor
                    |
                    v
        ThreadAssociationFeatures   (7 named numbers)  +  ThreadMatchDetail (facts for explanations)

What this module IS
-------------------
A small, deterministic, interpretable feature definition. It is the single input contract shared by the
deterministic baseline scorer today and by a learned ranker later: a supervised model can be trained on
``ThreadAssociationFeatures.as_vector()`` without any change here.

What this module is NOT
-----------------------
- Not a scorer or ranker: it never combines the features into a score and never compares threads.
- Not a classifier and not a second context pipeline: the session's context label and probability are INPUTS
  (they come from the existing, unmodified inference); nothing is re-inferred here.
- Not connected to storage, Qt or a clock: it receives plain objects, and the decision time ``as_of`` is
  always supplied by the caller. The same inputs always produce identical features.

The seven features
------------------
Each feature is documented once, in ``ThreadAssociationFeatures.FEATURES`` (name, definition, range).
"Evidence" below means the usable ``ThreadEvidence`` units of the thread (see "Leakage"). Decay weights are
``w = 0.5 ** (age_days / half_life_days)`` where age is measured from the evidence's ``observed_at`` to
``as_of``; the half-life is a constructor parameter (default 14 days) and is a bounded, smooth signal, never a
hard cut-off, so a very old thread is down-weighted but never excluded.

Lifetime and recent statistics are kept as SEPARATE features on purpose (context_share vs
recent_context_share, application_overlap vs recent_application_overlap): habits drift, and a model must be
able to tell a context that was common long ago from one that is common now.

Missing and sparse data
-----------------------
When there is nothing to compare against (no evidence, no known applications on either side), the affected
features are 0.0. That 0.0 means "no supporting evidence found", NOT "evidence of a mismatch". The distinction
is never lost, because ``historical_session_count`` is reported separately (0.0 = the thread has no usable
history at all) and the scorer/abstention policy reads it. A sparse thread therefore stays representable and is
scored low on evidence strength instead of being silently dropped or given an invented neutral value.

Leakage (why this matters for later supervised training)
--------------------------------------------------------
A feature must never be computed from the answer it is meant to predict. Evidence is therefore filtered HERE,
in the one place that decides what counts as history, and a caller cannot bypass it:

  - the session being scored (``context.session_id``) is never evidence, even if the user has already linked
    it to the thread;
  - evidence whose activity is after ``as_of`` is ignored (it did not exist at decision time);
  - evidence whose confirmation (``associated_at``) is after ``as_of`` is ignored (the user had not yet
    confirmed it at decision time), even if the activity itself was earlier.

So replaying a past decision with its original ``as_of`` reproduces the features that were available then.

Errors
------
Consistent with the rest of the project, malformed input is rejected with ``ValueError`` rather than
repaired: a timezone-naive ``as_of``, a probability outside [0, 1] (including NaN), an empty context label, or
a non-positive half-life.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

from app.ml.work_thread_profile import WorkThreadProfile

DEFAULT_HALF_LIFE_DAYS = 14.0
_SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True, slots=True)
class SessionAssociationContext:
    """
    The newly completed session, described only by information the project already permits:
    its inferred context, that inference's probability, and sanitised application names.

    ``observation_id`` is the persisted context observation this session's inference came from; it is only
    needed if a suggestion is later confirmed (the user's confirmation links THAT observation to a thread).
    """

    session_id: str
    context_label: str
    context_probability: float
    as_of: datetime
    applications: tuple[str, ...] = ()
    observation_id: int | None = None

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None or self.as_of.utcoffset() is None:
            raise ValueError("SessionAssociationContext.as_of must be timezone-aware (UTC); got a naive datetime.")
        if not (0.0 <= self.context_probability <= 1.0):  # also rejects NaN
            raise ValueError(f"context_probability must be within [0, 1]; got {self.context_probability!r}.")
        if not self.context_label or not self.context_label.strip():
            raise ValueError("context_label must be a non-empty string.")
        normalised = tuple(sorted({name.strip() for name in self.applications if name and name.strip()}))
        if normalised != self.applications:
            object.__setattr__(self, "applications", normalised)


@dataclass(frozen=True, slots=True)
class ThreadAssociationFeatures:
    """The seven interpretable features of one (session, Work Thread) pair. Named, never anonymous."""

    # name, definition, expected range. This tuple is the single source of truth for the contract.
    FEATURES: ClassVar[tuple[tuple[str, str, tuple[float, float]], ...]] = (
        (
            "context_share",
            "Lifetime fraction of the thread's usable sessions whose inferred context equals the current one.",
            (0.0, 1.0),
        ),
        (
            "recent_context_share",
            "Same fraction, but each session is weighted by recency decay (recent sessions count more).",
            (0.0, 1.0),
        ),
        (
            "current_context_probability",
            "The classifier's probability for the current session's context (a property of the session).",
            (0.0, 1.0),
        ),
        (
            "application_overlap",
            "Mean, over the current session's applications, of the fraction of the thread's sessions using it.",
            (0.0, 1.0),
        ),
        (
            "recent_application_overlap",
            "Same overlap, with each of the thread's sessions weighted by recency decay.",
            (0.0, 1.0),
        ),
        (
            "recency",
            "0.5 ** (days since the thread's most recent usable session / half-life); 0.0 with no history.",
            (0.0, 1.0),
        ),
        (
            "historical_session_count",
            "Number of distinct usable sessions in the thread's confirmed history (0.0 = no usable history).",
            (0.0, float("inf")),
        ),
    )
    FEATURE_NAMES: ClassVar[tuple[str, ...]] = tuple(name for name, _description, _range in FEATURES)

    context_share: float
    recent_context_share: float
    current_context_probability: float
    application_overlap: float
    recent_application_overlap: float
    recency: float
    historical_session_count: float

    def as_vector(self) -> tuple[float, ...]:
        """Values in ``FEATURE_NAMES`` order, ready to be used as supervised-learning input."""
        return tuple(getattr(self, name) for name in self.FEATURE_NAMES)

    def to_dict(self) -> dict[str, float]:
        return dict(zip(self.FEATURE_NAMES, self.as_vector()))


@dataclass(frozen=True, slots=True)
class ThreadMatchDetail:
    """Facts behind the features, used only to word explanations truthfully. Not a model input."""

    evidence_count: int
    matching_context_count: int
    matched_applications: tuple[tuple[str, int], ...]  # (application, sessions of the thread that used it)
    latest_evidence_at: datetime | None
    age_days: float | None


@dataclass(frozen=True, slots=True)
class ThreadAssociationExtraction:
    features: ThreadAssociationFeatures
    detail: ThreadMatchDetail


class ThreadAssociationFeatureExtractor:
    """Compute ``ThreadAssociationFeatures`` for one session against one Work Thread profile."""

    def __init__(self, *, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> None:
        if not (half_life_days > 0.0) or math.isinf(half_life_days):  # also rejects NaN
            raise ValueError(f"half_life_days must be a positive, finite number; got {half_life_days!r}.")
        self.half_life_days = float(half_life_days)

    def extract(self, profile: WorkThreadProfile, context: SessionAssociationContext) -> ThreadAssociationExtraction:
        as_of = context.as_of
        usable = [
            unit
            for unit in profile.evidence
            if unit.session_id != context.session_id  # never the answer being predicted
            and unit.observed_at <= as_of  # not from the future
            and unit.associated_at <= as_of  # already confirmed at decision time
        ]
        count = len(usable)
        weights = [self._decay(self._age_days(unit.observed_at, as_of)) for unit in usable]
        total_weight = sum(weights)

        matching = [(unit, weight) for unit, weight in zip(usable, weights) if unit.context_label == context.context_label]
        matching_count = len(matching)
        context_share = matching_count / count if count else 0.0
        recent_context_share = sum(w for _unit, w in matching) / total_weight if total_weight > 0.0 else 0.0

        overlap_lifetime: list[float] = []
        overlap_recent: list[float] = []
        matched: list[tuple[str, int]] = []
        for application in context.applications:
            users = [(unit, weight) for unit, weight in zip(usable, weights) if application in unit.applications]
            overlap_lifetime.append(len(users) / count if count else 0.0)
            overlap_recent.append(sum(w for _unit, w in users) / total_weight if total_weight > 0.0 else 0.0)
            if users:
                matched.append((application, len(users)))
        application_overlap = sum(overlap_lifetime) / len(overlap_lifetime) if overlap_lifetime else 0.0
        recent_application_overlap = sum(overlap_recent) / len(overlap_recent) if overlap_recent else 0.0

        latest = max((unit.observed_at for unit in usable), default=None)
        age = self._age_days(latest, as_of) if latest is not None else None
        recency = self._decay(age) if age is not None else 0.0

        features = ThreadAssociationFeatures(
            context_share=context_share,
            recent_context_share=recent_context_share,
            current_context_probability=float(context.context_probability),
            application_overlap=application_overlap,
            recent_application_overlap=recent_application_overlap,
            recency=recency,
            historical_session_count=float(len({unit.session_id for unit in usable})),
        )
        detail = ThreadMatchDetail(
            evidence_count=count,
            matching_context_count=matching_count,
            matched_applications=tuple(sorted(matched)),
            latest_evidence_at=latest,
            age_days=age,
        )
        return ThreadAssociationExtraction(features=features, detail=detail)

    @staticmethod
    def _age_days(then: datetime, as_of: datetime) -> float:
        return max(0.0, (as_of - then).total_seconds() / _SECONDS_PER_DAY)

    def _decay(self, age_days: float) -> float:
        return 0.5 ** (age_days / self.half_life_days)
