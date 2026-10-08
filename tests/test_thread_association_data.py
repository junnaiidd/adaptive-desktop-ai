"""
Work Thread association: persistence-facing contracts.

  WorkThreadProfileBuilder      derives profiles from existing stores (no new schema, never writes)
  WorkThreadAssociationAdvisor  completed-session context -> ranked, advisory assessment (never writes)
  AssociationConfirmer          the ONLY write path, and only on an explicit user confirmation

The pure ranking rules are covered in test_thread_association_engine.py.
"""

from __future__ import annotations

import dataclasses
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.database.activity_repository import ActivityRepository, ActivitySegment, StoredSession
from app.ml.context_observation_store import ContextObservationStore
from app.ml.task_store import TaskStore
from app.ml.thread_association_advisor import WorkThreadAssociationAdvisor
from app.ml.thread_association_confirmation import (
    AssociationConfirmer,
    AssociationFeedback,
    AssociationFeedbackKind,
    AssociationSuggestion,
)
from app.ml.thread_association_ranking import AssociationOutcome, AbstainReason
from app.ml.work_thread_profile import WorkThreadProfileBuilder
from app.ml.work_thread_store import WorkThreadHasAssociationsError, WorkThreadStore

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
DEV = "Software Development"
DOCS = "Documentation"


def ago(days: float) -> datetime:
    return NOW - timedelta(days=days)


class Env:
    """Every store points at ONE database file, exactly like the running application."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.repo = ActivityRepository(path)
        self.observations = ContextObservationStore(path)
        self.threads = WorkThreadStore(path)
        self.tasks = TaskStore(path)

    def builder(self, **overrides) -> WorkThreadProfileBuilder:
        kwargs = dict(activity_repository=self.repo)
        kwargs.update(overrides)
        return WorkThreadProfileBuilder(self.threads, self.observations, **kwargs)

    def advisor(self, **overrides) -> WorkThreadAssociationAdvisor:
        return WorkThreadAssociationAdvisor(self.threads, self.observations, self.repo, **overrides)

    def confirmer(self) -> AssociationConfirmer:
        return AssociationConfirmer(self.threads)

    def session(self, sid: str, days: float, *segments) -> str:
        """segments: (application, process_name, window_title)"""
        start = ago(days)
        self.repo.start_session(StoredSession(id=sid, started_at=start, ended_at=None))
        for index, (application, process, title) in enumerate(segments):
            self.repo.insert_activity(
                ActivitySegment(
                    session_id=sid,
                    started_at=start + timedelta(minutes=index * 10),
                    ended_at=start + timedelta(minutes=index * 10 + 9),
                    application=application,
                    process_name=process,
                    window_title=title,
                    duration_seconds=540.0,
                )
            )
        return sid

    def observe(self, sid: str, label: str, days: float, probability: float = 0.9):
        return self.observations.record_observation(
            sid, label, {label: probability, "Other": 1 - probability}, "test-model", observed_at=ago(days)
        )

    def thread(self, name: str, days_old: float = 100):
        return self.threads.create_work_thread(name, created_at=ago(days_old))

    def link(self, thread, observation, days: float):
        return self.threads.associate_observation(thread.id, observation.id, associated_at=ago(days))

    def history_thread(self, name: str, sessions: int, label: str = DEV, apps=(("VS Code", "code.exe"), ("Python", "python.exe")), first_days_ago: float = 2.0):
        """A thread with `sessions` linked, completed sessions (newest first_days_ago, then older)."""
        thread = self.thread(name)
        for i in range(sessions):
            days = first_days_ago + i * 0.5
            sid = self.session(f"{name}-{i}", days, *[(a, p, f"secret title {name}-{i}") for a, p in apps])
            self.link(thread, self.observe(sid, label, days - 0.01), days - 0.02)
        return thread

    def current(self, label: str = DEV, probability: float = 0.9, apps=(("VS Code", "code.exe"), ("Python", "python.exe")), sid: str = "current"):
        self.session(sid, 0.05, *[(a, p, "current secret title") for a, p in apps])
        return self.observe(sid, label, 0.04, probability)


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path / "activity.db")


def _snapshot(path: Path):
    connection = sqlite3.connect(path)
    try:
        tables = sorted(r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'"))
        counts = {t: connection.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in tables}
        columns = {t: [r[1] for r in connection.execute(f'PRAGMA table_info("{t}")')] for t in tables}
    finally:
        connection.close()
    return tables, counts, columns


# ============================================================================
# Profile builder
# ============================================================================


def test_profiles_cover_every_thread_ordered_by_id_including_empty_and_tasks_only(env: Env) -> None:
    rich = env.history_thread("Rich", 2)
    empty = env.thread("Empty")
    tasks_only = env.thread("TasksOnly")
    env.tasks.create_task(tasks_only.id, "a task", created_at=ago(1))

    profiles = env.builder().build_profiles()

    assert [p.work_thread.id for p in profiles] == sorted([rich.id, empty.id, tasks_only.id])
    by_name = {p.work_thread.name: p for p in profiles}
    assert len(by_name["Rich"].evidence) == 2
    assert by_name["Empty"].evidence == () and by_name["TasksOnly"].evidence == ()


def test_no_threads_gives_no_profiles_and_unknown_thread_gives_none(env: Env) -> None:
    assert env.builder().build_profiles() == ()
    assert env.builder().build_profile(404) is None


def test_an_evidence_unit_carries_exactly_the_documented_facts(env: Env) -> None:
    thread = env.thread("T")
    sid = env.session("s1", 3, ("VS Code", "code.exe", "Report.docx - secret"), ("Python", "python.exe", "x"))
    observation = env.observe(sid, DEV, 2.9)
    env.link(thread, observation, 2.8)

    (unit,) = env.builder().build_profile(thread.id).evidence

    assert unit.observation_id == observation.id and unit.session_id == "s1"
    assert unit.context_label == DEV
    assert unit.observed_at == ago(2.9) and unit.associated_at == ago(2.8)
    assert unit.applications == ("Python", "VS Code")  # distinct, sorted


def test_window_titles_never_enter_a_profile(env: Env) -> None:
    thread = env.history_thread("T", 2)

    text = repr(env.builder().build_profile(thread.id))

    assert "secret title" not in text  # privacy: titles are not semantic features and are never read into the profile


def test_application_name_falls_back_to_process_and_nameless_segments_are_skipped(env: Env) -> None:
    thread = env.thread("T")
    sid = env.session("s1", 2, (None, "tool.exe", "t"), (None, None, "t"), ("  ", None, "t"))
    env.link(thread, env.observe(sid, DEV, 1.9), 1.8)

    (unit,) = env.builder().build_profile(thread.id).evidence

    assert unit.applications == ("tool.exe",)


def test_missing_activity_repository_or_segments_yield_empty_applications(env: Env) -> None:
    thread = env.thread("T")
    with_segments = env.session("s1", 2, ("VS Code", "code.exe", "t"))
    without_segments = env.session("s2", 3)
    env.link(thread, env.observe(with_segments, DEV, 1.9), 1.8)
    env.link(thread, env.observe(without_segments, DEV, 2.9), 2.8)

    no_repo = env.builder(activity_repository=None).build_profile(thread.id)
    with_repo = env.builder().build_profile(thread.id)

    assert all(unit.applications == () for unit in no_repo.evidence)
    by_session = {u.session_id: u for u in with_repo.evidence}
    assert by_session["s1"].applications == ("VS Code",) and by_session["s2"].applications == ()


def test_duplicate_associations_collapse_to_one_unit_with_the_earliest_confirmation(env: Env) -> None:
    thread = env.thread("T")
    sid = env.session("s1", 5, ("VS Code", "code.exe", "t"))
    observation = env.observe(sid, DEV, 4.9)
    env.link(thread, observation, 4.0)
    env.link(thread, observation, 1.0)  # the store permits duplicates

    (unit,) = env.builder().build_profile(thread.id).evidence

    assert unit.associated_at == ago(4.0)  # first confirmation decides when it became known


def test_several_observations_of_one_session_count_once_using_the_latest(env: Env) -> None:
    thread = env.thread("T")
    sid = env.session("s1", 5, ("VS Code", "code.exe", "t"))
    first = env.observe(sid, DOCS, 4.9)
    latest = env.observe(sid, DEV, 4.5)
    env.link(thread, first, 4.0)
    env.link(thread, latest, 4.0)

    (unit,) = env.builder().build_profile(thread.id).evidence

    assert unit.context_label == DEV and unit.observation_id == latest.id


def test_multiple_sessions_are_separate_units_in_chronological_order(env: Env) -> None:
    thread = env.history_thread("T", 4)

    evidence = env.builder().build_profile(thread.id).evidence

    assert len(evidence) == 4 and len({e.session_id for e in evidence}) == 4
    assert [e.observed_at for e in evidence] == sorted(e.observed_at for e in evidence)


def test_threads_are_isolated_and_one_observation_can_support_two_threads(env: Env) -> None:
    first = env.history_thread("First", 2, label=DEV)
    second = env.history_thread("Second", 3, label=DOCS)
    shared_session = env.session("shared", 9, ("VS Code", "code.exe", "t"))
    shared = env.observe(shared_session, DEV, 8.9)
    env.link(first, shared, 8.0)
    env.link(second, shared, 8.0)

    profiles = {p.work_thread.name: p for p in env.builder().build_profiles()}

    assert {e.context_label for e in profiles["First"].evidence} == {DEV}
    assert {e.context_label for e in profiles["Second"].evidence} == {DOCS, DEV}
    assert len(profiles["First"].evidence) == 3 and len(profiles["Second"].evidence) == 4


def test_old_threads_are_still_profiled(env: Env) -> None:
    old = env.thread("Old", days_old=2000)
    sid = env.session("ancient", 1500, ("VS Code", "code.exe", "t"))
    env.link(old, env.observe(sid, DEV, 1499), 1498)

    (profile,) = env.builder().build_profiles()

    assert profile.work_thread.id == old.id and len(profile.evidence) == 1


def test_profiles_are_identical_after_a_restart(tmp_path: Path) -> None:
    first_run = Env(tmp_path / "activity.db")
    first_run.history_thread("T", 3)
    before = first_run.builder().build_profiles()

    restarted = Env(tmp_path / "activity.db")  # brand-new instances, same file

    assert restarted.builder().build_profiles() == before


def test_building_profiles_never_changes_the_database(env: Env) -> None:
    env.history_thread("T", 3)
    env.current()
    before = _snapshot(env.path)

    env.builder().build_profiles()
    env.builder().build_profile(1)

    assert _snapshot(env.path) == before


# ============================================================================
# Advisor: current completed-session context -> advisory assessment
# ============================================================================


def test_a_clear_match_yields_a_suggestion_with_the_evidence_a_popup_needs(env: Env) -> None:
    thread = env.history_thread("Adaptive Desktop AI", 6)
    env.history_thread("Unrelated", 2, label=DOCS, apps=(("Notepad", "notepad.exe"),), first_days_ago=60)
    observation = env.current()

    assessment = env.advisor().assess(observation.id, as_of=NOW)

    assert assessment.decision.outcome is AssociationOutcome.SUGGEST
    suggestion = assessment.suggestion
    assert isinstance(suggestion, AssociationSuggestion)
    assert suggestion.work_thread_id == thread.id and suggestion.work_thread_name == "Adaptive Desktop AI"
    assert suggestion.observation_id == observation.id and suggestion.session_id == "current"
    assert suggestion.association_score == assessment.decision.suggested.association_score
    assert suggestion.applications == ("Python", "VS Code")
    assert set(suggestion.matched_applications) == {"Python", "VS Code"}
    assert suggestion.reasons and suggestion.as_of == NOW
    assert len(suggestion.features.as_vector()) == 7


def test_assessing_never_writes_anything(env: Env) -> None:
    env.history_thread("A", 6)
    observation = env.current()
    before = _snapshot(env.path)

    env.advisor().assess(observation.id, as_of=NOW)
    env.advisor().assess(observation.id, as_of=NOW)

    assert _snapshot(env.path) == before  # no association, thread, task, observation or activity row appeared
    assert env.threads.list_work_threads_for_observation(observation.id) == []


def test_no_threads_gives_no_suggestion(env: Env) -> None:
    observation = env.current()

    assessment = env.advisor().assess(observation.id, as_of=NOW)

    assert assessment.suggestion is None
    assert assessment.decision.abstain_reason is AbstainReason.NO_WORK_THREADS


def test_an_ambiguous_match_gives_no_suggestion(env: Env) -> None:
    env.history_thread("A", 6)
    env.history_thread("B", 6)
    observation = env.current()

    assessment = env.advisor().assess(observation.id, as_of=NOW)

    assert assessment.suggestion is None
    assert assessment.decision.abstain_reason is AbstainReason.AMBIGUOUS
    assert len(assessment.decision.candidates) == 2  # the ranking is still available for inspection


def test_a_session_without_application_data_is_assessed_safely(env: Env) -> None:
    env.history_thread("A", 6)
    env.session("bare", 0.05)  # no activity segments
    observation = env.observe("bare", DEV, 0.04)

    assessment = env.advisor().assess(observation.id, as_of=NOW)

    assert assessment.context.applications == ()
    assert assessment.decision.candidates[0].features.application_overlap == 0.0


def test_unknown_observation_is_rejected_and_as_of_is_required(env: Env) -> None:
    with pytest.raises(ValueError):
        env.advisor().assess(404, as_of=NOW)
    observation = env.current()
    with pytest.raises(TypeError):
        env.advisor().assess(observation.id, NOW)  # type: ignore[misc]  # as_of must be explicit, never a hidden clock read
    with pytest.raises(ValueError):
        env.advisor().assess(observation.id, as_of=datetime(2026, 10, 1, 12, 0))  # naive


def test_the_advisor_adds_no_database_schema(tmp_path: Path) -> None:
    baseline = Env(tmp_path / "baseline.db")
    tables_without_advisor = _snapshot(baseline.path)[0]

    env = Env(tmp_path / "with.db")
    env.history_thread("A", 3)
    env.advisor().assess(env.current().id, as_of=NOW)

    assert _snapshot(env.path)[0] == tables_without_advisor


# ---------------------------------------------------------------- leakage


def test_confirming_a_session_does_not_change_how_that_same_session_is_scored(env: Env) -> None:
    thread = env.history_thread("A", 4)
    observation = env.current()
    advisor = env.advisor()

    before = advisor.assess(observation.id, as_of=NOW)
    env.link(thread, observation, 0.01)  # the target association now exists in the database
    after = advisor.assess(observation.id, as_of=NOW)

    assert {c.work_thread.id: c.features for c in after.decision.candidates} == {
        c.work_thread.id: c.features for c in before.decision.candidates
    }
    assert after.decision.candidates[0].association_score == before.decision.candidates[0].association_score


def test_associations_confirmed_after_the_decision_time_are_not_used(env: Env) -> None:
    thread = env.history_thread("A", 3)
    observation = env.current()
    decision_time = ago(0.03)
    baseline = env.advisor().assess(observation.id, as_of=decision_time)

    late_session = env.session("late", 0.04, ("VS Code", "code.exe", "t"))
    late = env.observe(late_session, DEV, 0.035)  # happened before the decision...
    env.link(thread, late, -1.0)  # ...but the user only linked it AFTER (a future time)
    replay = env.advisor().assess(observation.id, as_of=decision_time)

    assert {c.work_thread.id: c.features for c in replay.decision.candidates} == {
        c.work_thread.id: c.features for c in baseline.decision.candidates
    }


# ============================================================================
# Confirmation contract: explicit, user-initiated, and the only write path
# ============================================================================


def _suggestion(env: Env) -> AssociationSuggestion:
    env.history_thread("A", 6)
    observation = env.current()
    suggestion = env.advisor().assess(observation.id, as_of=NOW).suggestion
    assert suggestion is not None
    return suggestion


def test_confirming_creates_exactly_one_association_through_the_existing_store(env: Env) -> None:
    suggestion = _suggestion(env)
    before = _snapshot(env.path)[1]["work_thread_observations"]

    feedback = env.confirmer().confirm(suggestion, confirmed_at=NOW)

    assert isinstance(feedback, AssociationFeedback) and feedback.kind is AssociationFeedbackKind.CONFIRMED
    assert _snapshot(env.path)[1]["work_thread_observations"] == before + 1
    linked = env.threads.list_observations_for_work_thread(suggestion.work_thread_id)
    assert feedback.association in linked
    assert feedback.association.observation_id == suggestion.observation_id
    assert feedback.association.associated_at == NOW and feedback.decided_at == NOW
    assert feedback.suggestion is suggestion


def test_confirming_twice_does_not_create_a_duplicate_association(env: Env) -> None:
    suggestion = _suggestion(env)
    confirmer = env.confirmer()
    count_before = _snapshot(env.path)[1]["work_thread_observations"]

    first = confirmer.confirm(suggestion, confirmed_at=NOW)
    second = confirmer.confirm(suggestion, confirmed_at=NOW + timedelta(minutes=5))

    assert _snapshot(env.path)[1]["work_thread_observations"] == count_before + 1
    assert second.association == first.association


def test_rejecting_writes_nothing_and_records_the_decision_in_the_returned_feedback(env: Env) -> None:
    suggestion = _suggestion(env)
    before = _snapshot(env.path)

    feedback = env.confirmer().reject(suggestion, rejected_at=NOW)

    assert feedback.kind is AssociationFeedbackKind.REJECTED and feedback.association is None
    assert feedback.suggestion is suggestion and feedback.decided_at == NOW
    assert _snapshot(env.path) == before  # persisting negative examples is deliberately deferred (needs a schema)


def test_confirming_a_suggestion_without_a_persisted_observation_is_refused_and_writes_nothing(env: Env) -> None:
    suggestion = dataclasses.replace(_suggestion(env), observation_id=None)
    before = _snapshot(env.path)

    with pytest.raises(ValueError):
        env.confirmer().confirm(suggestion, confirmed_at=NOW)

    assert _snapshot(env.path) == before


def test_confirm_requires_a_timezone_aware_time(env: Env) -> None:
    suggestion = _suggestion(env)

    with pytest.raises(ValueError):
        env.confirmer().confirm(suggestion, confirmed_at=datetime(2026, 10, 1, 12, 0))
    with pytest.raises(ValueError):
        env.confirmer().reject(suggestion, rejected_at=datetime(2026, 10, 1, 12, 0))


def test_suggestions_and_feedback_are_immutable_and_carry_a_training_ready_feature_snapshot(env: Env) -> None:
    suggestion = _suggestion(env)
    feedback = env.confirmer().reject(suggestion, rejected_at=NOW)

    with pytest.raises(dataclasses.FrozenInstanceError):
        suggestion.association_score = 1.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        feedback.kind = AssociationFeedbackKind.CONFIRMED  # type: ignore[misc]
    vector = feedback.suggestion.features.as_vector()
    assert all(isinstance(v, float) for v in vector) and len(vector) == len(suggestion.features.FEATURE_NAMES)


def test_existing_work_thread_deletion_rules_are_unchanged_by_confirmation(env: Env) -> None:
    suggestion = _suggestion(env)
    env.confirmer().confirm(suggestion, confirmed_at=NOW)

    with pytest.raises(WorkThreadHasAssociationsError):
        env.threads.delete_work_thread(suggestion.work_thread_id)


def test_the_confirmer_only_touches_the_association_table(env: Env) -> None:
    suggestion = _suggestion(env)
    before = _snapshot(env.path)

    env.confirmer().confirm(suggestion, confirmed_at=NOW)

    after = _snapshot(env.path)
    changed = {t for t in before[1] if before[1][t] != after[1][t]}
    assert changed == {"work_thread_observations"}
    assert after[0] == before[0] and after[2] == before[2]  # no schema change
