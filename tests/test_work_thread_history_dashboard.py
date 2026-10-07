"""
Dashboard integration for the Work Thread history read model.

The core model is covered in test_work_thread_history.py. These tests cover only the thin presentation
path: DashboardController composition, the compact text, and the single label in the Work panel.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel

from app.database.activity_repository import ActivityRepository, ActivitySegment, StoredSession
from app.ml.context_observation_store import ContextObservationStore
from app.ml.task_store import TaskStore
from app.ml.work_thread_store import WorkThreadStore
from app.ui.dashboard_controller import DashboardController, describe_work_thread_history
from app.ui.main_window import MainWindow
from app.workspace.workspace_snapshot_store import CapturedForegroundWindow, WorkspaceSnapshotStore

BASE = datetime(2026, 9, 20, 9, 0, tzinfo=timezone.utc)


def at(hours: float = 0.0) -> datetime:
    return BASE + timedelta(hours=hours)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class Fixture:
    def __init__(self, path: Path) -> None:
        self.repo = ActivityRepository(path)
        self.observations = ContextObservationStore(path)
        self.threads = WorkThreadStore(path)
        self.tasks = TaskStore(path)
        self.snapshots = WorkspaceSnapshotStore(path)
        service = SimpleNamespace(
            poll_interval_seconds=5.0,
            current_activity=None,
            current_duration_seconds=None,
            session_manager=SimpleNamespace(current_session=None),
        )
        self.controller = DashboardController(
            self.repo,
            service,
            context_observation_store=self.observations,
            work_thread_store=self.threads,
            task_store=self.tasks,
            workspace_snapshot_store=self.snapshots,
        )

    def thread_with_history(self, name: str = "Adaptive Desktop AI"):
        thread = self.threads.create_work_thread(name, created_at=at(-24))
        for index, (label, app_name) in enumerate([("Software Development", "VS Code"), ("ML / Model Development", "Python")]):
            sid = f"s{index}"
            self.repo.start_session(StoredSession(id=sid, started_at=at(index * 3), ended_at=None))
            self.repo.insert_activity(
                ActivitySegment(sid, at(index * 3), at(index * 3 + 1), app_name, f"{app_name}.exe", None, 3600.0)
            )
            observation = self.observations.record_observation(
                sid, label, {label: 1.0}, "m", observed_at=at(index * 3 + 1.1)
            )
            self.threads.associate_observation(thread.id, observation.id, associated_at=at(10))
        self.tasks.create_task(thread.id, "open", created_at=at(0))
        self.tasks.complete_task(self.tasks.create_task(thread.id, "done", created_at=at(1)).id)
        return thread


@pytest.fixture
def fx(tmp_path: Path) -> Fixture:
    return Fixture(tmp_path / "activity.db")


# ----------------------------------------------------------------------------
# Controller
# ----------------------------------------------------------------------------


def test_controller_composes_history_from_its_own_stores(fx: Fixture) -> None:
    thread = fx.thread_with_history()

    history = fx.controller.work_thread_history(thread.id)

    assert history.work_thread == thread
    assert history.session_count == 2 and history.associated_observation_count == 2
    assert history.open_task_count == 1 and history.completed_task_count == 1
    assert [a.name for a in history.recent_applications] == ["Python", "VS Code"]


def test_controller_returns_none_for_unknown_thread_or_missing_stores(fx: Fixture, tmp_path: Path) -> None:
    assert fx.controller.work_thread_history(404) is None

    bare = DashboardController(fx.repo, SimpleNamespace())  # no Work Thread / observation stores
    assert bare.work_thread_history(1) is None
    assert bare.work_thread_history_stamp(1) is None


def test_stamp_changes_only_when_a_history_input_changes(fx: Fixture) -> None:
    thread = fx.thread_with_history()
    stamp = fx.controller.work_thread_history_stamp(thread.id)

    assert fx.controller.work_thread_history_stamp(thread.id) == stamp  # stable while nothing changes

    task = fx.tasks.create_task(thread.id, "new", created_at=at(5))
    after_task = fx.controller.work_thread_history_stamp(thread.id)
    assert after_task != stamp

    fx.tasks.complete_task(task.id)
    after_toggle = fx.controller.work_thread_history_stamp(thread.id)
    assert after_toggle != after_task

    fx.snapshots.create_snapshot(
        thread.id, CapturedForegroundWindow("Notepad", "notepad.exe", r"C:\Windows\notepad.exe", None), captured_at=at(6)
    )
    assert fx.controller.work_thread_history_stamp(thread.id) != after_toggle


# ----------------------------------------------------------------------------
# Compact text
# ----------------------------------------------------------------------------


def test_description_is_compact_and_only_states_known_facts(fx: Fixture) -> None:
    thread = fx.thread_with_history()
    fx.snapshots.create_snapshot(
        thread.id, CapturedForegroundWindow("Notepad", "notepad.exe", r"C:\Windows\notepad.exe", None), captured_at=at(6)
    )

    text = describe_work_thread_history(fx.controller.work_thread_history(thread.id))
    lines = text.split("\n")

    assert len(lines) <= 3
    assert lines[0] == "Contexts: Software Development, ML / Model Development"
    assert lines[1] == "2 sessions · 2 linked observations · 1 open / 1 done tasks"
    assert lines[2].startswith("Recent: Python, VS Code · last active ")
    assert lines[2].endswith("workspace snapshot saved")


def test_description_for_an_empty_thread_does_not_invent_anything(fx: Fixture) -> None:
    thread = fx.threads.create_work_thread("Empty", created_at=at(0))

    text = describe_work_thread_history(fx.controller.work_thread_history(thread.id))

    assert text == "No linked observations yet"
    assert "Contexts" not in text and "Recent" not in text and "last active" not in text


def test_description_pluralises_and_caps_long_context_lists(fx: Fixture) -> None:
    thread = fx.threads.create_work_thread("Many", created_at=at(0))
    for index in range(5):
        sid = f"s{index}"
        fx.repo.start_session(StoredSession(id=sid, started_at=at(index), ended_at=None))
        observation = fx.observations.record_observation(sid, f"Context {index}", {"x": 1.0}, "m", observed_at=at(index))
        fx.threads.associate_observation(thread.id, observation.id, associated_at=at(10))

    first_line = describe_work_thread_history(fx.controller.work_thread_history(thread.id)).split("\n")[0]
    assert first_line == "Contexts: Context 0, Context 1, Context 2 (+2 more)"

    one = fx.threads.create_work_thread("One", created_at=at(0))
    observation = fx.observations.record_observation("s0", "Context 0", {"x": 1.0}, "m", observed_at=at(0))
    fx.threads.associate_observation(one.id, observation.id, associated_at=at(1))
    assert "1 session · 1 linked observation" in describe_work_thread_history(fx.controller.work_thread_history(one.id))


# ----------------------------------------------------------------------------
# Window
# ----------------------------------------------------------------------------


def _window(fx: Fixture) -> MainWindow:
    window = MainWindow(
        fx.controller,
        work_thread_store=fx.threads,
        task_store=fx.tasks,
        workspace_snapshot_store=fx.snapshots,
    )
    window.resize(1180, 760)
    window.show()
    window.refresh()
    return window


def test_selected_thread_shows_its_history_in_the_work_panel(fx: Fixture, qapp) -> None:
    thread = fx.thread_with_history()
    window = _window(fx)
    qapp.processEvents()

    label = window.thread_history_label
    assert label.isVisibleTo(window)
    assert "Software Development" in label.text() and "2 sessions" in label.text()
    assert window.findChild(QLabel, "pageTitle") is not None
    window.close()


def test_history_label_is_hidden_when_no_thread_exists(fx: Fixture, qapp) -> None:
    window = _window(fx)
    qapp.processEvents()

    assert window.thread_history_label.text() == ""
    assert not window.thread_history_label.isVisibleTo(window)
    window.close()


def test_history_follows_the_selection_and_later_changes(fx: Fixture, qapp) -> None:
    first = fx.thread_with_history("First")
    second = fx.threads.create_work_thread("Second", created_at=at(0))
    window = _window(fx)
    qapp.processEvents()
    assert "linked observation" in window.thread_history_label.text()

    window.thread_dropdown.setCurrentIndex(window.thread_dropdown.findData(second.id))
    qapp.processEvents()
    assert window.thread_history_label.text() == "No linked observations yet"

    fx.tasks.create_task(second.id, "now there is a task", created_at=at(2))
    window.refresh()  # the periodic tick picks up the changed input
    assert window.thread_history_label.text() == "No linked observations yet · 1 open / 0 done tasks"
    assert first.id != second.id
    window.close()


def test_unchanged_inputs_do_not_rebuild_the_history_on_every_tick(fx: Fixture, qapp, monkeypatch) -> None:
    fx.thread_with_history()
    window = _window(fx)
    qapp.processEvents()

    calls = []
    original = fx.controller.work_thread_history
    monkeypatch.setattr(fx.controller, "work_thread_history", lambda *a, **k: calls.append(a) or original(*a, **k))
    for _ in range(5):
        window.refresh()

    assert calls == []  # only the cheap fingerprint ran; the full history was not recomputed
    window.close()
