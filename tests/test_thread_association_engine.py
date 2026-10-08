"""
Work Thread association engine: deterministic baseline (features -> scorer -> ranker -> abstention).

These tests use ONLY synthetic in-memory fixtures (no database, no Qt, no ML model). They protect:

  - ranking semantics        (context match, recency, application overlap, historical strength)
  - abstention               (no threads / weak / ambiguous / uninformative -> NO_MATCH, never a forced pick)
  - temporal behaviour       (bounded decay, old threads stay candidates)
  - deterministic ordering   (explicit tie-breaks, input-order independence)
  - no leakage               (the session being scored and not-yet-confirmed evidence never reach features)
  - explanation grounding    (every reason corresponds to a real, positive feature contribution)

The score is a ranking signal, NOT a calibrated probability, and these tests never assert it is one.
"""

from __future__ import annotations

import ast
import dataclasses
import itertools
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.ml.thread_association_features import (
    SessionAssociationContext,
    ThreadAssociationFeatureExtractor,
    ThreadAssociationFeatures,
)
from app.ml.thread_association_ranking import (
    AbstainReason,
    AbstentionPolicy,
    AllWorkThreadsCandidateGenerator,
    AssociationOutcome,
    AssociationWeights,
    BaselineAssociationScorer,
    ScoreBreakdown,
    WorkThreadCandidateRanker,
)
from app.ml.work_thread_profile import ThreadEvidence, WorkThreadProfile
from app.ml.work_thread_store import WorkThread

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
_OBSERVATION_IDS = itertools.count(1)  # deterministic fixture ids (never hash(): it is randomised per process)
DEV = "Software Development"
DOCS = "Documentation"
ML = "ML / Model Development"


def days_ago(days: float) -> datetime:
    return NOW - timedelta(days=days)


def ev(session: str, label: str, age_days: float, apps: tuple[str, ...] = (), *, obs_id: int | None = None) -> ThreadEvidence:
    observed = days_ago(age_days)
    return ThreadEvidence(
        observation_id=obs_id if obs_id is not None else next(_OBSERVATION_IDS),
        session_id=session,
        context_label=label,
        observed_at=observed,
        associated_at=min(observed + timedelta(hours=1), NOW),  # confirmed shortly after, never after the decision time
        applications=tuple(sorted(apps)),
    )


def profile(thread_id: int, name: str, *evidence: ThreadEvidence) -> WorkThreadProfile:
    return WorkThreadProfile(
        work_thread=WorkThread(id=thread_id, name=name, created_at=days_ago(400)),
        evidence=tuple(evidence),
    )


def ctx(label: str = DEV, probability: float = 0.9, apps: tuple[str, ...] = ("VS Code", "Python"), session: str = "current") -> SessionAssociationContext:
    return SessionAssociationContext(
        session_id=session,
        context_label=label,
        context_probability=probability,
        as_of=NOW,
        applications=tuple(sorted(apps)),
    )


def extract(prof: WorkThreadProfile, context: SessionAssociationContext | None = None, half_life: float = 14.0):
    return ThreadAssociationFeatureExtractor(half_life_days=half_life).extract(prof, context or ctx())


def rank(profiles, context=None, **kwargs):
    return WorkThreadCandidateRanker(**kwargs).rank(context or ctx(), tuple(profiles))


def strong_dev_evidence(prefix: str, n: int, age_days: float = 1.0, apps=("VS Code", "Python")):
    return [ev(f"{prefix}{i}", DEV, age_days + i * 0.1, apps) for i in range(n)]


# ============================================================================
# Feature definitions (documented numbers, hand-computed)
# ============================================================================


def _reference_profile() -> WorkThreadProfile:
    return profile(
        1,
        "Adaptive Desktop AI",
        ev("s1", DEV, 0, ("VS Code", "Python")),
        ev("s2", DEV, 14, ("VS Code",)),
        ev("s3", DOCS, 28, ("Chrome",)),
        ev("s4", DEV, 28, ("VS Code", "Python", "Chrome")),
    )


def test_feature_values_match_their_documented_definitions() -> None:
    # decay weights at half-life 14d: 0d -> 1, 14d -> 0.5, 28d -> 0.25 (x2); total weight = 2.0
    f = extract(_reference_profile()).features

    assert f.context_share == pytest.approx(3 / 4)  # 3 of 4 sessions were Software Development
    assert f.recent_context_share == pytest.approx((1 + 0.5 + 0.25) / 2.0)
    assert f.current_context_probability == pytest.approx(0.9)
    assert f.application_overlap == pytest.approx((3 / 4 + 2 / 4) / 2)  # VS Code in 3/4, Python in 2/4
    assert f.recent_application_overlap == pytest.approx((1.75 / 2.0 + 1.25 / 2.0) / 2)
    assert f.recency == pytest.approx(1.0)  # latest evidence is "today"
    assert f.historical_session_count == 4.0


def test_recent_and_lifetime_context_share_are_genuinely_different_signals() -> None:
    old_dev_recent_docs = profile(
        1, "T", ev("a", DEV, 200), ev("b", DEV, 190), ev("c", DOCS, 1), ev("d", DOCS, 0.5)
    )

    f = extract(old_dev_recent_docs).features

    assert f.context_share == pytest.approx(0.5)  # lifetime: half the sessions were Dev
    assert f.recent_context_share < 0.01  # but decayed evidence is dominated by recent Docs sessions


def test_recency_follows_the_half_life_exactly_and_is_bounded() -> None:
    def recency(age: float) -> float:
        return extract(profile(1, "T", ev("a", DEV, age))).features.recency

    assert recency(0) == pytest.approx(1.0)
    assert recency(14) == pytest.approx(0.5)
    assert recency(28) == pytest.approx(0.25)
    assert recency(7) == pytest.approx(0.5 ** 0.5)
    assert 0.0 < recency(3650) < 1e-6  # decays toward 0 but never reaches a hard cutoff
    assert extract(profile(1, "T", ev("a", DEV, 14)), half_life=28.0).features.recency == pytest.approx(0.5 ** 0.5)


def test_recency_uses_the_most_recent_evidence_not_the_average() -> None:
    f = extract(profile(1, "T", ev("a", DEV, 100), ev("b", DEV, 0))).features
    assert f.recency == pytest.approx(1.0)


def test_application_overlap_is_frequency_aware() -> None:
    often = profile(1, "T", *[ev(f"s{i}", DEV, i, ("VS Code",)) for i in range(4)])
    once = profile(2, "T", ev("s0", DEV, 0, ("VS Code",)), *[ev(f"s{i}", DEV, i, ("Chrome",)) for i in range(1, 4)])
    context = ctx(apps=("VS Code",))

    assert extract(often, context).features.application_overlap == pytest.approx(1.0)
    assert extract(once, context).features.application_overlap == pytest.approx(0.25)


def test_distinct_context_matches_are_counted_in_the_detail() -> None:
    extraction = extract(_reference_profile())

    assert extraction.detail.evidence_count == 4
    assert extraction.detail.matching_context_count == 3
    assert dict(extraction.detail.matched_applications) == {"VS Code": 3, "Python": 2}
    assert extraction.detail.latest_evidence_at == days_ago(0)


# ---------------------------------------------------------------- missing / sparse data


def test_a_thread_with_no_evidence_yields_all_zero_features_and_never_crashes() -> None:
    extraction = extract(profile(1, "Brand new"))

    assert extraction.features.as_vector() == (0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0)  # only the session's own probability
    assert extraction.detail.evidence_count == 0
    assert extraction.detail.latest_evidence_at is None


def test_missing_application_data_is_handled_without_error() -> None:
    no_apps_history = profile(1, "T", ev("a", DEV, 1), ev("b", DEV, 2))
    assert extract(no_apps_history).features.application_overlap == 0.0

    with_apps_history = profile(2, "T", ev("a", DEV, 1, ("VS Code",)))
    assert extract(with_apps_history, ctx(apps=())).features.application_overlap == 0.0
    assert extract(with_apps_history, ctx(apps=())).features.recent_application_overlap == 0.0


def test_feature_contract_is_named_documented_and_ordered() -> None:
    names = ThreadAssociationFeatures.FEATURE_NAMES
    documented = [name for name, _description, _range in ThreadAssociationFeatures.FEATURES]

    assert list(names) == documented
    assert len(names) == len(set(names)) == 7  # a small, explainable set
    assert all(description for _name, description, _range in ThreadAssociationFeatures.FEATURES)
    features = extract(_reference_profile()).features
    assert features.as_vector() == tuple(features.to_dict()[name] for name in names)
    assert {f.name for f in dataclasses.fields(ThreadAssociationFeatures)} == set(names)  # no hidden extras


def test_every_feature_lies_inside_its_documented_range() -> None:
    features = extract(_reference_profile()).features.to_dict()
    for name, _description, (low, high) in ThreadAssociationFeatures.FEATURES:
        assert low <= features[name] <= high or math.isinf(high), name


# ---------------------------------------------------------------- malformed input


def test_naive_datetimes_are_rejected_consistently_with_the_utc_convention() -> None:
    with pytest.raises(ValueError):
        SessionAssociationContext(
            session_id="x", context_label=DEV, context_probability=0.9, as_of=datetime(2026, 10, 1, 12, 0)
        )
    with pytest.raises(ValueError):
        ThreadEvidence(1, "s", DEV, datetime(2026, 1, 1), NOW, ())


@pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan")])
def test_out_of_range_context_probability_is_rejected(probability: float) -> None:
    with pytest.raises(ValueError):
        ctx(probability=probability)


def test_half_life_must_be_positive() -> None:
    for bad in (0, -1, float("nan")):
        with pytest.raises(ValueError):
            ThreadAssociationFeatureExtractor(half_life_days=bad)


# ============================================================================
# Leakage: history is only what was known and confirmed BEFORE this decision
# ============================================================================


def test_the_session_being_scored_never_contributes_to_its_own_features() -> None:
    clean = profile(1, "T", ev("old1", DEV, 5, ("VS Code",)), ev("old2", DOCS, 6, ("Chrome",)))
    leaking = profile(
        1, "T", *clean.evidence, ev("current", DEV, 0.001, ("VS Code", "Python"))  # the answer being predicted
    )

    assert extract(leaking).features == extract(clean).features
    assert extract(leaking).detail == extract(clean).detail


def test_evidence_observed_after_as_of_is_ignored() -> None:
    past = ev("past", DEV, 3, ("VS Code",))
    future = ThreadEvidence(99, "future", DEV, NOW + timedelta(days=2), NOW + timedelta(days=2, hours=1), ("VS Code",))

    assert extract(profile(1, "T", past, future)).features == extract(profile(1, "T", past)).features


def test_evidence_whose_activity_is_after_as_of_is_ignored_even_if_its_confirmation_date_is_earlier() -> None:
    """The store does not enforce associated_at >= observed_at, so each rule must hold on its own."""
    past = ev("past", DEV, 3, ("VS Code",))
    inconsistent = ThreadEvidence(98, "odd", DEV, NOW + timedelta(days=2), days_ago(1), ("VS Code",))  # 'confirmed' before it happened

    assert extract(profile(1, "T", past, inconsistent)).features == extract(profile(1, "T", past)).features


def test_evidence_confirmed_after_as_of_is_ignored_even_if_the_activity_was_earlier() -> None:
    past = ev("past", DEV, 3, ("VS Code",))
    confirmed_late = ThreadEvidence(99, "late", DEV, days_ago(1), NOW + timedelta(hours=5), ("VS Code",))  # user linked it LATER

    assert extract(profile(1, "T", past, confirmed_late)).features == extract(profile(1, "T", past)).features


def test_features_are_computed_from_a_single_context_and_expose_no_target_field() -> None:
    names = " ".join(ThreadAssociationFeatures.FEATURE_NAMES)
    for forbidden in ("target", "confirmed", "label", "thread_id", "session_id"):
        assert forbidden not in names


# ============================================================================
# Scorer: transparent weighted baseline
# ============================================================================


def test_default_weights_are_explicit_and_sum_to_one() -> None:
    w = AssociationWeights()

    assert (w.context, w.application, w.recency, w.history) == (0.40, 0.30, 0.15, 0.15)
    assert w.context + w.application + w.recency + w.history == pytest.approx(1.0)
    assert 0.0 <= w.recent_weight <= 1.0 and w.history_saturation > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(context=-0.1, application=0.5, recency=0.3, history=0.3),
        dict(context=0.5, application=0.5, recency=0.5, history=0.5),
        dict(recent_weight=1.5),
        dict(history_saturation=0.0),
    ],
)
def test_invalid_weights_are_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        AssociationWeights(**kwargs)


def test_score_equals_the_documented_formula_for_the_reference_profile() -> None:
    extraction = extract(_reference_profile())
    breakdown = BaselineAssociationScorer().score(extraction.features, extraction.detail, ctx())

    context_component = 0.9 * (0.5 * 0.75 + 0.5 * 0.875)
    application_component = 0.5 * 0.625 + 0.5 * 0.75
    expected = 0.40 * context_component + 0.30 * application_component + 0.15 * 1.0 + 0.15 * (4 / (4 + 3))
    assert breakdown.association_score == pytest.approx(expected)
    assert breakdown.association_score == pytest.approx(0.7344642857)


def test_contributions_are_weight_times_value_and_sum_to_the_score() -> None:
    extraction = extract(_reference_profile())
    breakdown = BaselineAssociationScorer().score(extraction.features, extraction.detail, ctx())

    assert [c.component for c in breakdown.contributions] == ["context", "application", "recency", "history"]
    for c in breakdown.contributions:
        assert c.contribution == pytest.approx(c.weight * c.value)
        assert 0.0 <= c.value <= 1.0
    assert sum(c.contribution for c in breakdown.contributions) == pytest.approx(breakdown.association_score)


def test_context_probability_scales_only_the_context_component() -> None:
    thread = _reference_profile()
    scorer = BaselineAssociationScorer()

    def contributions(probability: float):
        extraction = extract(thread, ctx(probability=probability))
        return {c.component: c.contribution for c in scorer.score(extraction.features, extraction.detail, ctx(probability=probability)).contributions}

    high, low = contributions(0.9), contributions(0.45)

    assert low["context"] == pytest.approx(high["context"] / 2)  # the documented rule: linear in the probability
    for unchanged in ("application", "recency", "history"):
        assert low[unchanged] == pytest.approx(high[unchanged])


def test_history_strength_saturates_at_n_over_n_plus_k() -> None:
    scorer = BaselineAssociationScorer()

    def history_value(n: int) -> float:
        prof = profile(1, "T", *[ev(f"s{i}", DEV, i, ("VS Code",)) for i in range(n)])
        e = extract(prof)
        return [c for c in scorer.score(e.features, e.detail, ctx()).contributions if c.component == "history"][0].value

    assert history_value(0) == 0.0
    assert history_value(3) == pytest.approx(0.5)  # n == k -> half strength
    assert history_value(9) == pytest.approx(0.75)
    assert history_value(1) < history_value(2) < history_value(10) < 1.0


def test_score_is_bounded_between_zero_and_one() -> None:
    best = profile(1, "T", *[ev(f"s{i}", DEV, 0, ("VS Code", "Python")) for i in range(500)])
    e = extract(best, ctx(probability=1.0))
    top = BaselineAssociationScorer().score(e.features, e.detail, ctx(probability=1.0)).association_score
    assert 0.9 < top <= 1.0
    e0 = extract(profile(2, "T"), ctx(probability=0.0))
    assert BaselineAssociationScorer().score(e0.features, e0.detail, ctx(probability=0.0)).association_score == 0.0


# ============================================================================
# Ranking semantics
# ============================================================================


def test_strong_context_match_ranks_above_a_mismatch() -> None:
    match = profile(1, "Match", *strong_dev_evidence("m", 5, apps=("Other",)))
    mismatch = profile(2, "Mismatch", *[ev(f"x{i}", DOCS, 1 + i * 0.1, ("Other",)) for i in range(5)])

    decision = rank([mismatch, match])

    assert [c.work_thread.name for c in decision.candidates] == ["Match", "Mismatch"]
    assert [c.rank for c in decision.candidates] == [1, 2]


def test_recent_evidence_beats_stale_but_otherwise_identical_evidence() -> None:
    fresh = profile(2, "Fresh", *[ev(f"f{i}", DEV, 1 + i * 0.1, ("VS Code", "Python")) for i in range(4)])
    stale = profile(1, "Stale", *[ev(f"s{i}", DEV, 120 + i * 0.1, ("VS Code", "Python")) for i in range(4)])  # lower id

    decision = rank([stale, fresh])

    assert decision.candidates[0].work_thread.name == "Fresh"
    assert decision.candidates[0].association_score > decision.candidates[1].association_score


def test_application_overlap_raises_the_score_all_else_equal() -> None:
    same_apps = profile(1, "Overlap", *[ev(f"a{i}", DEV, 2 + i * 0.1, ("VS Code", "Python")) for i in range(4)])
    other_apps = profile(2, "NoOverlap", *[ev(f"b{i}", DEV, 2 + i * 0.1, ("Notepad",)) for i in range(4)])

    decision = rank([other_apps, same_apps])

    top, second = decision.candidates
    assert top.work_thread.name == "Overlap"
    app = lambda c: [x for x in c.contributions if x.component == "application"][0].contribution
    assert app(top) > 0 and app(second) == 0


def test_more_historical_evidence_scores_higher_all_else_equal() -> None:
    many = profile(2, "Many", *[ev(f"m{i}", DEV, 1.0, ("VS Code",)) for i in range(9)])
    few = profile(1, "Few", ev("f0", DEV, 1.0, ("VS Code",)))

    decision = rank([few, many], ctx(apps=("VS Code",)))

    assert [c.work_thread.name for c in decision.candidates] == ["Many", "Few"]


def test_three_thread_scenario_follows_the_documented_rules_not_accidental_order() -> None:
    thread_a = profile(  # strong context match, moderate recency (~14 days)
        1, "A: context match",
        *[ev(f"a{i}", DEV, 14 + i * 0.2, ("Terminal",)) for i in range(6)],
    )
    thread_b = profile(  # weak context match, very recent
        2, "B: very recent",
        ev("b0", DEV, 0.1, ("Notepad",)),
        *[ev(f"b{i}", DOCS, 0.2 + i * 0.1, ("Notepad",)) for i in range(1, 8)],
    )
    thread_c = profile(  # strong application overlap, stale and different context
        3, "C: app overlap",
        *[ev(f"c{i}", DOCS, 150 + i, ("VS Code", "Python")) for i in range(6)],
    )
    context = ctx(probability=0.9, apps=("VS Code", "Python"))

    decision = rank([thread_c, thread_b, thread_a], context)

    # Independently recompute each score from the documented formula.
    w = AssociationWeights()
    expected = {}
    for prof in (thread_a, thread_b, thread_c):
        f = extract(prof, context).features
        r = w.recent_weight
        expected[prof.work_thread.name] = (
            w.context * f.current_context_probability * ((1 - r) * f.context_share + r * f.recent_context_share)
            + w.application * ((1 - r) * f.application_overlap + r * f.recent_application_overlap)
            + w.recency * f.recency
            + w.history * (f.historical_session_count / (f.historical_session_count + w.history_saturation))
        )
    for candidate in decision.candidates:
        assert candidate.association_score == pytest.approx(expected[candidate.work_thread.name])
    assert [c.work_thread.name for c in decision.candidates] == sorted(expected, key=lambda n: -expected[n])
    assert decision.candidates[0].work_thread.name == "A: context match"
    assert decision.outcome is AssociationOutcome.SUGGEST


# ---------------------------------------------------------------- candidate generation / sparsity


def test_sparse_and_empty_threads_remain_candidates() -> None:
    rich = profile(1, "Rich", *strong_dev_evidence("r", 6))
    empty = profile(2, "Empty")
    one_session = profile(3, "OneSession", ev("o", DEV, 1, ("VS Code",)))

    decision = rank([empty, one_session, rich])

    assert {c.work_thread.name for c in decision.candidates} == {"Rich", "Empty", "OneSession"}
    assert decision.candidates[0].work_thread.name == "Rich"
    assert decision.candidates[-1].work_thread.name == "Empty"


def test_a_thread_with_tasks_but_no_observations_is_just_an_empty_profile() -> None:
    decision = rank([profile(1, "Only tasks")])

    assert [c.work_thread.name for c in decision.candidates] == ["Only tasks"]
    assert decision.candidates[0].features.historical_session_count == 0.0


def test_old_threads_are_never_excluded_and_can_still_win() -> None:
    ancient = profile(1, "Ancient", *[ev(f"a{i}", DEV, 1500 + i, ("VS Code", "Python")) for i in range(8)])
    unrelated_recent = profile(2, "Recent", *[ev(f"r{i}", DOCS, 0.5 + i * 0.1, ("Notepad",)) for i in range(8)])

    decision = rank([unrelated_recent, ancient])

    assert {c.work_thread.name for c in decision.candidates} == {"Ancient", "Recent"}  # nobody dropped for age
    assert decision.candidates[0].work_thread.name == "Ancient"  # strong content match outweighs staleness


def test_custom_candidate_generator_is_honoured() -> None:
    class OnlyEven:
        def generate(self, context, profiles):
            return tuple(p for p in profiles if p.work_thread.id % 2 == 0)

    decision = rank([profile(1, "Odd"), profile(2, "Even")], candidate_generator=OnlyEven())

    assert [c.work_thread.name for c in decision.candidates] == ["Even"]


def test_default_generator_returns_every_thread_in_stable_id_order() -> None:
    profiles = [profile(3, "c"), profile(1, "a"), profile(2, "b")]

    generated = AllWorkThreadsCandidateGenerator().generate(ctx(), tuple(profiles))

    assert [p.work_thread.id for p in generated] == [1, 2, 3]


# ============================================================================
# Abstention
# ============================================================================


def test_no_work_threads_means_no_candidates_and_an_explicit_abstention() -> None:
    decision = rank([])

    assert decision.candidates == ()
    assert decision.outcome is AssociationOutcome.NO_MATCH
    assert decision.abstain_reason is AbstainReason.NO_WORK_THREADS
    assert decision.suggested is None


def test_weak_evidence_abstains_instead_of_forcing_the_best_candidate() -> None:
    weak = profile(1, "Weak", ev("w0", DOCS, 90, ("Notepad",)), ev("w1", ML, 95, ("Notepad",)))

    decision = rank([weak])

    assert decision.outcome is AssociationOutcome.NO_MATCH
    assert decision.abstain_reason is AbstainReason.LOW_SCORE
    assert [c.work_thread.name for c in decision.candidates] == ["Weak"]  # still ranked and inspectable
    assert decision.suggested is None


def test_close_scores_trigger_abstention_as_ambiguous() -> None:
    a = profile(1, "A", *strong_dev_evidence("a", 6))
    b = profile(2, "B", *strong_dev_evidence("b", 6))

    decision = rank([a, b])

    assert decision.candidates[0].association_score >= AbstentionPolicy().min_top_score  # the top IS strong
    assert decision.outcome is AssociationOutcome.NO_MATCH
    assert decision.abstain_reason is AbstainReason.AMBIGUOUS


def test_a_clear_winner_is_suggested() -> None:
    a = profile(1, "A", *strong_dev_evidence("a", 8))
    b = profile(2, "B", *[ev(f"b{i}", DOCS, 40 + i, ("Notepad",)) for i in range(2)])

    decision = rank([a, b])

    assert decision.outcome is AssociationOutcome.SUGGEST
    assert decision.abstain_reason is None
    assert decision.suggested is decision.candidates[0]
    assert decision.suggested.work_thread.name == "A"


def test_a_single_strong_candidate_needs_no_margin() -> None:
    decision = rank([profile(1, "Only", *strong_dev_evidence("a", 8))])

    assert decision.outcome is AssociationOutcome.SUGGEST


def test_an_uninformative_current_context_abstains() -> None:
    strong_thread = profile(1, "A", *strong_dev_evidence("a", 8))

    decision = rank([strong_thread], ctx(probability=0.05, apps=()))

    assert decision.outcome is AssociationOutcome.NO_MATCH
    assert decision.abstain_reason is AbstainReason.UNINFORMATIVE_CURRENT_CONTEXT


def test_low_context_probability_alone_is_not_uninformative_when_applications_are_known() -> None:
    strong_thread = profile(1, "A", *strong_dev_evidence("a", 8))

    decision = rank([strong_thread], ctx(probability=0.05, apps=("VS Code", "Python")))

    assert decision.abstain_reason is not AbstainReason.UNINFORMATIVE_CURRENT_CONTEXT


def test_recency_and_history_alone_can_never_justify_a_suggestion() -> None:
    # No context match, no application overlap -- only a very recent, well-populated thread.
    busy_but_unrelated = profile(1, "Busy", *[ev(f"b{i}", DOCS, 0.01 * (i + 1), ("Notepad",)) for i in range(50)])

    decision = rank([busy_but_unrelated], policy=AbstentionPolicy(min_top_score=0.0, min_margin=0.0))

    assert decision.outcome is AssociationOutcome.NO_MATCH
    assert decision.abstain_reason is AbstainReason.NO_COMPATIBILITY_EVIDENCE


def test_abstention_policy_is_configurable_and_validated() -> None:
    medium = profile(1, "Medium", *[ev(f"m{i}", DEV, 10 + i, ("VS Code",)) for i in range(2)])
    assert rank([medium], policy=AbstentionPolicy(min_top_score=0.99)).abstain_reason is AbstainReason.LOW_SCORE
    assert rank([medium], policy=AbstentionPolicy(min_top_score=0.05)).outcome is AssociationOutcome.SUGGEST

    for bad in (dict(min_top_score=1.2), dict(min_margin=-0.1), dict(min_context_probability=2.0)):
        with pytest.raises(ValueError):
            AbstentionPolicy(**bad)


# ============================================================================
# Deterministic ordering
# ============================================================================


class ConstantScorer:
    """A stand-in for a future learned scorer: every candidate gets the same score."""

    def score(self, features, detail, context):
        return ScoreBreakdown(association_score=0.5, contributions=(), reasons=())


def test_equal_scores_are_ordered_by_evidence_then_recency_then_thread_id() -> None:
    strongest_evidence = profile(7, "MoreSessions", *[ev(f"a{i}", DEV, 30, ()) for i in range(5)])
    more_recent = profile(8, "MoreRecent", *[ev(f"b{i}", DEV, 1, ()) for i in range(3)])
    older_low_id = profile(2, "OlderLowId", *[ev(f"c{i}", DEV, 40, ()) for i in range(3)])
    twin_high_id = profile(9, "TwinHighId", *[ev(f"d{i}", DEV, 40, ()) for i in range(3)])
    twin_low_id = profile(3, "TwinLowId", *[ev(f"e{i}", DEV, 40, ()) for i in range(3)])

    decision = rank(
        [twin_high_id, older_low_id, more_recent, strongest_evidence, twin_low_id],
        scorer=ConstantScorer(),
        policy=AbstentionPolicy(min_top_score=0.0, min_margin=0.0),
    )

    assert [c.work_thread.name for c in decision.candidates] == [
        "MoreSessions",  # 1) stronger evidence
        "MoreRecent",  # 2) then more recent evidence
        "OlderLowId",  # 3) then stable id (2 < 3 < 9)
        "TwinLowId",
        "TwinHighId",
    ]


def test_the_id_tie_break_is_explicit_and_does_not_rely_on_generator_or_input_order() -> None:
    """A future generator may return threads in any order; equal candidates must still be ordered by id."""

    class HostileOrder:
        def generate(self, context, profiles):
            return tuple(sorted(profiles, key=lambda p: -p.work_thread.id))  # highest id first

    twins = [profile(i, f"T{i}", *[ev(f"t{i}-{j}", DEV, 5, ()) for j in range(3)]) for i in (4, 2, 9, 1)]

    decision = rank(
        twins,
        scorer=ConstantScorer(),
        policy=AbstentionPolicy(min_top_score=0.0, min_margin=0.0),
        candidate_generator=HostileOrder(),
    )

    assert [c.work_thread.id for c in decision.candidates] == [1, 2, 4, 9]


def test_ranking_is_independent_of_input_order_and_repeatable() -> None:
    profiles = [
        profile(i, f"T{i}", *[ev(f"t{i}-{j}", DEV if (i + j) % 2 else DOCS, (i * 3 + j) % 17, ("VS Code",) if j % 2 else ("Python",)) for j in range(i % 5 + 1)])
        for i in range(1, 9)
    ]
    reference = rank(profiles)

    for seed in range(10):
        shuffled = profiles[:]
        random.Random(seed).shuffle(shuffled)
        assert rank(shuffled) == reference
    assert rank(profiles) == rank(profiles)


def test_ranking_does_not_mutate_its_inputs() -> None:
    profiles = tuple(profile(i, f"T{i}", *strong_dev_evidence(f"t{i}", 3)) for i in range(1, 4))
    context = ctx()
    snapshot = (profiles, context)
    copy_before = repr(snapshot)

    WorkThreadCandidateRanker().rank(context, profiles)

    assert repr(snapshot) == copy_before
    with pytest.raises(dataclasses.FrozenInstanceError):
        profiles[0].work_thread.name = "mutated"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.context_label = "mutated"  # type: ignore[misc]


def test_a_learned_scorer_can_replace_the_baseline_without_changing_the_ranker() -> None:
    class PreferHighestId:
        def score(self, features, detail, context):
            return ScoreBreakdown(association_score=0.0, contributions=(), reasons=())

    profiles = [profile(1, "a", *strong_dev_evidence("a", 4)), profile(2, "b")]

    decision = rank(profiles, scorer=PreferHighestId(), policy=AbstentionPolicy(min_top_score=0.0, min_margin=0.0))

    assert len(decision.candidates) == 2 and decision.candidates[0].association_score == 0.0


# ============================================================================
# Explanations are grounded in real contributions
# ============================================================================


def test_every_reason_corresponds_to_a_positive_contribution() -> None:
    decision = rank([_reference_profile()])
    candidate = decision.candidates[0]

    positive = {c.component for c in candidate.contributions if c.contribution > 0}
    assert positive == {"context", "application", "recency", "history"}
    text = " | ".join(candidate.reasons)
    assert f"'{DEV}' matched 3 of 4 earlier sessions" in text
    assert "VS Code (3 of 4 sessions)" in text and "Python (2 of 4 sessions)" in text
    assert "Most recent session in this thread was today" in text
    assert "Based on 4 earlier sessions" in text


def test_zero_contribution_components_produce_no_reason() -> None:
    apps_only = profile(1, "AppsOnly", *[ev(f"a{i}", DOCS, 2 + i, ("VS Code", "Python")) for i in range(4)])

    candidate = rank([apps_only]).candidates[0]

    contributions = {c.component: c.contribution for c in candidate.contributions}
    assert contributions["context"] == 0.0 and contributions["application"] > 0
    text = " | ".join(candidate.reasons)
    assert "matched" not in text and "Applications also used" in text  # context reason absent, application present


def test_reasons_only_name_applications_that_were_really_matched() -> None:
    thread = profile(1, "T", *[ev(f"t{i}", DEV, 1 + i, ("VS Code", "Terminal")) for i in range(3)])

    candidate = rank([thread], ctx(apps=("VS Code", "Spotify"))).candidates[0]

    text = " | ".join(candidate.reasons)
    assert "VS Code (3 of 3 sessions)" in text
    assert "Spotify" not in text and "Terminal" not in text  # not in the current session / not shared


def test_reasons_are_ordered_by_contribution_and_regenerated_identically() -> None:
    candidate = rank([_reference_profile()]).candidates[0]
    order = [c.component for c in sorted(candidate.contributions, key=lambda c: -c.contribution) if c.contribution > 0]
    keyword = {"context": "matched", "application": "Applications also used", "recency": "Most recent session", "history": "Based on"}

    positions = [next(i for i, r in enumerate(candidate.reasons) if keyword[comp] in r) for comp in order]

    assert positions == sorted(positions)
    assert rank([_reference_profile()]).candidates[0].reasons == candidate.reasons


def test_reasons_follow_contribution_size_even_when_that_differs_from_weight_order() -> None:
    # Weight order is context > application > recency = history, but here recency and history contribute
    # far more than the (weak) context match, so they must be listed first.
    thread = profile(
        1, "T",
        ev("m0", DEV, 0.1, ("Notepad",)),
        *[ev(f"o{i}", DOCS, 0.1 + i * 0.01, ("Notepad",)) for i in range(1, 10)],
    )

    candidate = rank([thread], ctx(apps=("VS Code",))).candidates[0]

    by_size = [c.component for c in sorted(candidate.contributions, key=lambda c: -c.contribution) if c.contribution > 0]
    assert by_size == ["recency", "history", "context"]  # not the weight order (context first)
    keyword = {"context": "matched", "recency": "Most recent session", "history": "Based on"}
    positions = [next(i for i, r in enumerate(candidate.reasons) if keyword[comp] in r) for comp in by_size]
    assert positions == sorted(positions)


def test_sparse_history_is_flagged_as_limited_evidence_in_the_reasons() -> None:
    candidate = rank([profile(1, "T", ev("only", DEV, 1, ("VS Code",)))]).candidates[0]

    assert any("Based on 1 earlier session" in r and "limited evidence" in r for r in candidate.reasons)


def test_the_reason_for_age_uses_whole_days() -> None:
    candidate = rank([profile(1, "T", ev("a", DEV, 9.4, ("VS Code",)))]).candidates[0]

    assert any("was 9 days ago" in r for r in candidate.reasons)


# ============================================================================
# Architecture guards: a deterministic baseline, not a model; independent of Qt/DB
# ============================================================================

APP = Path(__file__).resolve().parents[1] / "app" / "ml"
PURE_MODULES = ("thread_association_features.py", "thread_association_ranking.py")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


@pytest.mark.parametrize(
    "module", PURE_MODULES + ("work_thread_profile.py", "thread_association_confirmation.py", "thread_association_advisor.py")
)
def test_no_new_module_depends_on_qt_ml_libraries_sql_or_the_monitor(module: str) -> None:
    imported = {name.split(".")[0] for name in _imports(APP / module)}
    for forbidden in ("PySide6", "sklearn", "numpy", "pandas", "joblib", "torch", "sqlite3", "win32gui", "psutil", "openai"):
        assert forbidden not in imported, f"{module} imports {forbidden}"


@pytest.mark.parametrize("module", PURE_MODULES)
def test_scoring_core_does_not_touch_stores_repositories_or_the_context_pipeline(module: str) -> None:
    imported = _imports(APP / module)
    for forbidden in (
        "app.database.activity_repository",
        "app.ml.context_observation_store",
        "app.ml.task_store",
        "app.ml.context_classifier",
        "app.ml.context_inference",
        "app.ml.feature_engineering",
        "app.ml.training_examples",
        "app.ml.training_pipeline",
    ):
        assert forbidden not in imported, f"{module} imports {forbidden}"


@pytest.mark.parametrize("module", PURE_MODULES)
def test_scoring_core_reads_no_clock_and_no_randomness(module: str) -> None:
    source = (APP / module).read_text(encoding="utf-8")
    for forbidden in ("datetime.now(", "datetime.utcnow(", "time.time(", "import random", "random."):
        assert forbidden not in source, f"{module} uses {forbidden}"


def test_no_module_writes_sql() -> None:
    for module in PURE_MODULES + ("work_thread_profile.py", "thread_association_confirmation.py", "thread_association_advisor.py"):
        lowered = (APP / module).read_text(encoding="utf-8").lower()
        for keyword in ("insert into", "delete from", "create table", "alter table", "update work_thread"):
            assert keyword not in lowered, f"{module} contains {keyword}"


def test_result_types_do_not_call_the_score_a_probability_or_confidence() -> None:
    from app.ml.thread_association_ranking import AssociationDecision, CandidateScore

    for cls in (CandidateScore, AssociationDecision, ScoreBreakdown):
        for field in dataclasses.fields(cls):
            assert "confidence" not in field.name and "probability" not in field.name
