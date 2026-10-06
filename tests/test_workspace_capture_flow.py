"""Capture-flow ordering: minimize first, read the foreground afterwards, always restore the window.

The window's show/raise/activate methods are replaced by recorders so these tests never flash a real
window, and the focus-delay timer is replaced so the deferred step can be run deterministically.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from app.database.activity_repository import ActivityRepository
from app.ml.work_thread_store import WorkThreadStore
from app.ui import main_window as main_window_module
from app.ui.dashboard_controller import DashboardController
from app.ui.main_window import MainWindow
from app.workspace.workspace_snapshot_store import CapturedForegroundWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _Probe:
    def __init__(self, events, error=None):
        self.events = events
        self.error = error

    def capture_foreground_window(self):
        self.events.append("capture")
        if self.error:
            raise self.error
        return CapturedForegroundWindow("Notepad", "notepad.exe", r"C:\Windows\notepad.exe", "Notes")


def _build(tmp_path, monkeypatch, qapp, *, probe_error=None, state=Qt.WindowState.WindowNoState):
    events: list[str] = []
    scheduled: list[tuple[int, object]] = []
    monkeypatch.setattr(main_window_module.QTimer, "singleShot", lambda ms, fn: scheduled.append((ms, fn)))

    database_path = tmp_path / "activity.db"
    repository = ActivityRepository(database_path)
    WorkThreadStore(database_path).create_work_thread("Workspace", created_at=datetime.now(timezone.utc))
    service = SimpleNamespace(
        poll_interval_seconds=5.0,
        current_activity=None,
        current_duration_seconds=None,
        session_manager=SimpleNamespace(current_session=None),
    )
    controller = DashboardController(repository, service)
    window = MainWindow(controller, workspace_probe=_Probe(events, probe_error))
    window.refresh()
    assert window.thread_dropdown.currentData() is not None
    scheduled.clear()  # drop the startup context-inference singleShot; keep only capture scheduling

    monkeypatch.setattr(window, "windowState", lambda: state)
    monkeypatch.setattr(window, "showMinimized", lambda: events.append("minimize"))
    monkeypatch.setattr(window, "showNormal", lambda: events.append("show_normal"))
    monkeypatch.setattr(window, "showMaximized", lambda: events.append("show_maximized"))
    monkeypatch.setattr(window, "raise_", lambda: events.append("raise"))
    monkeypatch.setattr(window, "activateWindow", lambda: events.append("activate"))
    return window, controller, events, scheduled


def test_capture_minimizes_first_and_reads_foreground_only_after_the_focus_delay(tmp_path, monkeypatch, qapp) -> None:
    window, controller, events, scheduled = _build(tmp_path, monkeypatch, qapp)
    thread_id = window.thread_dropdown.currentData()

    window._capture_workspace_snapshot()

    assert events == ["minimize"]  # nothing captured yet: this app is still handing focus back
    assert [ms for ms, _ in scheduled] == [main_window_module.WORKSPACE_CAPTURE_FOCUS_DELAY_MS]
    assert window.capture_workspace_btn.isEnabled() is False

    scheduled[0][1]()  # the focus delay elapses

    assert events == ["minimize", "capture", "show_normal", "raise", "activate"]
    snapshots = controller.list_workspace_snapshots(thread_id)
    assert len(snapshots) == 1 and snapshots[0].process_name == "notepad.exe"
    assert "Captured Notepad" in window.workspace_feedback.text()
    assert window.capture_workspace_btn.isEnabled() is True
    assert window.restore_workspace_btn.isEnabled() is True


def test_capture_failure_still_restores_the_window_and_reports_the_error(tmp_path, monkeypatch, qapp) -> None:
    window, controller, events, scheduled = _build(
        tmp_path, monkeypatch, qapp, probe_error=RuntimeError("Adaptive Desktop AI was the foreground window")
    )
    thread_id = window.thread_dropdown.currentData()

    window._capture_workspace_snapshot()
    scheduled[0][1]()

    assert events == ["minimize", "capture", "show_normal", "raise", "activate"]
    assert controller.list_workspace_snapshots(thread_id) == ()
    assert "Could not capture current window" in window.workspace_feedback.text()
    assert window.capture_workspace_btn.isEnabled() is True


def test_second_capture_click_while_pending_is_ignored(tmp_path, monkeypatch, qapp) -> None:
    window, _controller, events, scheduled = _build(tmp_path, monkeypatch, qapp)

    window._capture_workspace_snapshot()
    window._capture_workspace_snapshot()

    assert events == ["minimize"]
    assert len(scheduled) == 1


def test_capture_without_a_selected_work_thread_does_not_minimize(tmp_path, monkeypatch, qapp) -> None:
    window, _controller, events, scheduled = _build(tmp_path, monkeypatch, qapp)
    window.thread_dropdown.clear()

    window._capture_workspace_snapshot()

    assert events == [] and scheduled == []
    assert "select a Work Thread" in window.workspace_feedback.text()


def test_maximized_window_is_restored_maximized(tmp_path, monkeypatch, qapp) -> None:
    window, _controller, events, scheduled = _build(tmp_path, monkeypatch, qapp, state=Qt.WindowState.WindowMaximized)

    window._capture_workspace_snapshot()
    scheduled[0][1]()

    assert events == ["minimize", "capture", "show_maximized", "raise", "activate"]
