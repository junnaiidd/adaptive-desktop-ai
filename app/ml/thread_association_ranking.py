"""
Work Thread association ranking: candidate generation, a deterministic baseline scorer, ranking, abstention.

    profiles + current session context
        |
        v  CandidateGenerator            which Work Threads are compared (default: all of them)
        v  ThreadAssociationFeatureExtractor (features module)
        v  AssociationScorer             features -> association_score + per-component contributions + reasons
        v  WorkThreadCandidateRanker     deterministic ordering, then the abstention policy
        v
    AssociationDecision                  ranked candidates + SUGGEST / NO_MATCH (+ why it abstained)

What this module IS
-------------------
The deterministic *baseline*: a transparent weighted sum that can be read, tested and demonstrated. It is NOT
machine learning. The scorer is a ``Protocol``, so a learned ranker can later replace
``BaselineAssociationScorer`` without touching the ranker, the policy, the features or any caller.

What this module is NOT
-----------------------
- Not an association: a result is advisory. Nothing here writes, creates a Work Thread, links an observation,
  changes a task, or persists anything. Associating is a separate, explicit, user-initiated step
  (``thread_association_confirmation``).
- Not calibrated: ``association_score`` is a ranking signal in [0, 1]. It is NOT a probability and NOT a
  confidence, and no field here is named as if it were. The abstention thresholds are heuristic defaults, not
  statistically calibrated values; they are configurable precisely because they are judgement calls.
- Not connected to storage, Qt, a clock or randomness.

The baseline score
------------------
    score = w_context     * context_component
          + w_application * application_component
          + w_recency     * recency_component
          + w_history     * history_component

    context_component     = current_context_probability
                            * ((1 - r) * context_share + r * recent_context_share)
    application_component = (1 - r) * application_overlap + r * recent_application_overlap
    recency_component     = recency
    history_component     = n / (n + k)        n = historical_session_count, k = history_saturation

with r = ``recent_weight`` and default weights (context, application, recency, history) =
(0.40, 0.30, 0.15, 0.15), which sum to 1 so the score stays within [0, 1].

Why these terms and weights (judgement, documented rather than fitted - there are not yet enough confirmed
labels to fit anything):
  - context (0.40): the classifier's context is the strongest, most direct signal the project has; it is
    scaled by the classifier's own probability, so an unsure inference contributes proportionally less. That
    is the documented rule for how the current context probability affects scoring - and it affects only this
    component.
  - application (0.30): which applications a session used is independent evidence from the label, and it still
    works when the context label is ambiguous or the thread spans several contexts.
  - recency (0.15): a thread active yesterday is more plausible than one idle for months. Bounded decay, never
    an absolute rule, and deliberately small: it can break ties and nudge, but cannot carry a match on its own.
  - history (0.15): more confirmed sessions make the comparison more trustworthy. n / (n + k) rises quickly
    for the first few sessions and saturates, so one session is weak evidence and ten sessions are not 10x
    stronger. This term is also how sparsity is represented: a sparse thread is not penalised by an invented
    "uncertainty" number, it simply earns little here.
  - r = 0.5: lifetime and recent statistics are blended equally by default (drift-aware without discarding
    history).
Recency (0.15) + history (0.15) = 0.30 is below the default ``min_top_score`` (0.45): by construction a thread
can never be suggested on recency and history size alone - it needs real context or application evidence.

Abstention policy (checked in this order; the first that applies wins)
----------------------------------------------------------------------
  1. NO_WORK_THREADS                  there is no candidate at all.
  2. UNINFORMATIVE_CURRENT_CONTEXT    context probability < ``min_context_probability`` AND the session has no
                                      known applications: nothing reliable to compare.
  3. LOW_SCORE                        the best score < ``min_top_score``.
  4. NO_COMPATIBILITY_EVIDENCE        the best candidate has no context or application overlap at all (a
                                      structural guard that holds even if the thresholds are set to zero).
  5. AMBIGUOUS                        there are >= 2 candidates and best - second < ``min_margin``.
Otherwise SUGGEST. A single strong candidate needs no margin. Abstaining is a normal, first-class outcome: the
ranked candidates are still returned so they can be inspected, but ``suggested`` is None.

Ordering and tie-breaks
-----------------------
Candidates are ordered by (score descending, then historical_session_count descending, then recency
descending, then Work Thread id ascending). Scores are compared after rounding to 9 decimals so floating-point
noise cannot reorder otherwise-equal candidates. The result never depends on input order or on dict/set
iteration order.

Explanations
------------
Every reason is generated from the actual data behind a component whose contribution is positive, using the
real counts, application names and ages. A component that contributes nothing yields no reason. Reasons are
ordered by contribution, largest first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence

from app.ml.thread_association_features import (
    SessionAssociationContext,
    ThreadAssociationExtraction,
    ThreadAssociationFeatureExtractor,
    ThreadAssociationFeatures,
    ThreadMatchDetail,
)
from app.ml.work_thread_profile import WorkThreadProfile
from app.ml.work_thread_store import WorkThread


class AssociationOutcome(Enum):
    SUGGEST = "suggest"
    NO_MATCH = "no_match"


class AbstainReason(Enum):
    NO_WORK_THREADS = "no_work_threads"
    UNINFORMATIVE_CURRENT_CONTEXT = "uninformative_current_context"
    LOW_SCORE = "low_score"
    NO_COMPATIBILITY_EVIDENCE = "no_compatibility_evidence"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class AssociationWeights:
    """Explicit baseline weights and shape parameters. See the module docstring for the rationale."""

    context: float = 0.40
    application: float = 0.30
    recency: float = 0.15
    history: float = 0.15
    recent_weight: float = 0.5  # r: share given to recency-decayed statistics vs lifetime statistics
    history_saturation: float = 3.0  # k: sessions at which the history component reaches 0.5

    def __post_init__(self) -> None:
        parts = (self.context, self.application, self.recency, self.history)
        if any(not (part >= 0.0) or math.isinf(part) for part in parts):
            raise ValueError("Association weights must be finite and non-negative.")
        if not math.isclose(sum(parts), 1.0, abs_tol=1e-9):
            raise ValueError(f"Association weights must sum to 1.0 so the score stays in [0, 1]; got {sum(parts)}.")
        if not (0.0 <= self.recent_weight <= 1.0):
            raise ValueError("recent_weight must be within [0, 1].")
        if not (self.history_saturation > 0.0) or math.isinf(self.history_saturation):
            raise ValueError("history_saturation must be a positive, finite number.")


@dataclass(frozen=True, slots=True)
class AbstentionPolicy:
    """Heuristic, configurable thresholds. Not statistically calibrated (see the module docstring)."""

    min_top_score: float = 0.45
    min_margin: float = 0.10
    min_context_probability: float = 0.30

    def __post_init__(self) -> None:
        for name in ("min_top_score", "min_margin", "min_context_probability"):
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be within [0, 1]; got {value!r}.")


@dataclass(frozen=True, slots=True)
class ScoreContribution:
    """One weighted component of a score: ``contribution == weight * value``."""

    component: str  # "context" | "application" | "recency" | "history"
    value: float  # the component's own value in [0, 1]
    weight: float
    contribution: float


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """What a scorer returns. A learned scorer may return empty ``contributions`` and ``reasons``."""

    association_score: float
    contributions: tuple[ScoreContribution, ...]
    reasons: tuple[str, ...]


class AssociationScorer(Protocol):
    """The seam for replacing the baseline with a learned model later."""

    def score(
        self, features: ThreadAssociationFeatures, detail: ThreadMatchDetail, context: SessionAssociationContext
    ) -> ScoreBreakdown: ...


class CandidateGenerator(Protocol):
    """Chooses which Work Threads are compared. May become more selective later."""

    def generate(
        self, context: SessionAssociationContext, profiles: Sequence[WorkThreadProfile]
    ) -> tuple[WorkThreadProfile, ...]: ...


class AllWorkThreadsCandidateGenerator:
    """Every existing Work Thread is a candidate, in stable id order. New or sparse threads are never dropped."""

    def generate(
        self, context: SessionAssociationContext, profiles: Sequence[WorkThreadProfile]
    ) -> tuple[WorkThreadProfile, ...]:
        return tuple(sorted(profiles, key=lambda profile: profile.work_thread.id))


class BaselineAssociationScorer:
    """The transparent weighted-sum baseline. Not a learned model."""

    def __init__(self, weights: AssociationWeights | None = None) -> None:
        self.weights = weights or AssociationWeights()

    def score(
        self, features: ThreadAssociationFeatures, detail: ThreadMatchDetail, context: SessionAssociationContext
    ) -> ScoreBreakdown:
        w = self.weights
        r = w.recent_weight
        n = features.historical_session_count

        values = (
            (
                "context",
                w.context,
                features.current_context_probability
                * ((1.0 - r) * features.context_share + r * features.recent_context_share),
            ),
            (
                "application",
                w.application,
                (1.0 - r) * features.application_overlap + r * features.recent_application_overlap,
            ),
            ("recency", w.recency, features.recency),
            ("history", w.history, n / (n + w.history_saturation)),
        )
        contributions = tuple(
            ScoreContribution(component=name, value=value, weight=weight, contribution=weight * value)
            for name, weight, value in values
        )
        score = min(1.0, max(0.0, sum(c.contribution for c in contributions)))
        return ScoreBreakdown(
            association_score=score,
            contributions=contributions,
            reasons=_reasons(contributions, detail, context, w.history_saturation),
        )


def _reasons(
    contributions: tuple[ScoreContribution, ...],
    detail: ThreadMatchDetail,
    context: SessionAssociationContext,
    history_saturation: float,
) -> tuple[str, ...]:
    """One truthful sentence per positively contributing component, largest contribution first."""
    ordered = sorted(enumerate(contributions), key=lambda item: (-item[1].contribution, item[0]))
    sentences: list[str] = []
    for _index, contribution in ordered:
        if contribution.contribution <= 0.0:
            continue
        n = detail.evidence_count
        if contribution.component == "context":
            sentences.append(
                f"Context '{context.context_label}' matched {detail.matching_context_count} of {n} "
                f"earlier {_sessions(n)} in this thread"
            )
        elif contribution.component == "application":
            shared = ", ".join(
                f"{name} ({count} of {n} {_sessions(n)})" for name, count in detail.matched_applications
            )
            sentences.append(f"Applications also used in this thread: {shared}")
        elif contribution.component == "recency" and detail.age_days is not None:
            days = int(detail.age_days)
            when = "today" if days == 0 else ("1 day ago" if days == 1 else f"{days} days ago")
            sentences.append(f"Most recent session in this thread was {when}")
        elif contribution.component == "history":
            limited = " (limited evidence)" if n < history_saturation else ""
            sentences.append(f"Based on {n} earlier {_sessions(n)}{limited}")
    return tuple(sentences)


def _sessions(count: int) -> str:
    return "session" if count == 1 else "sessions"


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """One Work Thread compared with the current session. Advisory: scoring never creates an association."""

    rank: int  # 1-based
    work_thread: WorkThread
    association_score: float  # a ranking signal, NOT a probability
    features: ThreadAssociationFeatures
    detail: ThreadMatchDetail
    contributions: tuple[ScoreContribution, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AssociationDecision:
    """The ranked candidates plus an explicit SUGGEST / NO_MATCH outcome."""

    session_id: str
    candidates: tuple[CandidateScore, ...]
    outcome: AssociationOutcome
    abstain_reason: AbstainReason | None

    @property
    def suggested(self) -> CandidateScore | None:
        """The top candidate if (and only if) the policy did not abstain."""
        if self.outcome is AssociationOutcome.SUGGEST and self.candidates:
            return self.candidates[0]
        return None


class WorkThreadCandidateRanker:
    """Rank Work Threads for one session and decide whether the evidence justifies a suggestion at all."""

    def __init__(
        self,
        *,
        extractor: ThreadAssociationFeatureExtractor | None = None,
        scorer: AssociationScorer | None = None,
        policy: AbstentionPolicy | None = None,
        candidate_generator: CandidateGenerator | None = None,
    ) -> None:
        self._extractor = extractor or ThreadAssociationFeatureExtractor()
        self._scorer = scorer or BaselineAssociationScorer()
        self._policy = policy or AbstentionPolicy()
        self._generator = candidate_generator or AllWorkThreadsCandidateGenerator()

    def rank(self, context: SessionAssociationContext, profiles: Sequence[WorkThreadProfile]) -> AssociationDecision:
        scored = []
        for profile in self._generator.generate(context, profiles):
            extraction = self._extractor.extract(profile, context)
            breakdown = self._scorer.score(extraction.features, extraction.detail, context)
            scored.append((profile, extraction, breakdown))

        scored.sort(key=_ordering_key)
        candidates = tuple(
            CandidateScore(
                rank=position,
                work_thread=profile.work_thread,
                association_score=breakdown.association_score,
                features=extraction.features,
                detail=extraction.detail,
                contributions=breakdown.contributions,
                reasons=breakdown.reasons,
            )
            for position, (profile, extraction, breakdown) in enumerate(scored, start=1)
        )
        reason = self._abstain_reason(context, candidates)
        return AssociationDecision(
            session_id=context.session_id,
            candidates=candidates,
            outcome=AssociationOutcome.NO_MATCH if reason is not None else AssociationOutcome.SUGGEST,
            abstain_reason=reason,
        )

    def _abstain_reason(
        self, context: SessionAssociationContext, candidates: tuple[CandidateScore, ...]
    ) -> AbstainReason | None:
        policy = self._policy
        if not candidates:
            return AbstainReason.NO_WORK_THREADS
        if context.context_probability < policy.min_context_probability and not context.applications:
            return AbstainReason.UNINFORMATIVE_CURRENT_CONTEXT
        top = candidates[0]
        if top.association_score < policy.min_top_score:
            return AbstainReason.LOW_SCORE
        if not _has_compatibility_evidence(top.features):
            return AbstainReason.NO_COMPATIBILITY_EVIDENCE
        if len(candidates) >= 2 and top.association_score - candidates[1].association_score < policy.min_margin:
            return AbstainReason.AMBIGUOUS
        return None


def _has_compatibility_evidence(features: ThreadAssociationFeatures) -> bool:
    """True if the candidate shares any context or application with the session (recency/size never count)."""
    return any(
        value > 0.0
        for value in (
            features.context_share,
            features.recent_context_share,
            features.application_overlap,
            features.recent_application_overlap,
        )
    )


def _ordering_key(item: tuple[WorkThreadProfile, ThreadAssociationExtraction, ScoreBreakdown]) -> tuple:
    profile, extraction, breakdown = item
    return (
        -round(breakdown.association_score, 9),
        -extraction.features.historical_session_count,
        -extraction.features.recency,
        profile.work_thread.id,
    )
