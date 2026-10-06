"""Planner and executor tests using ports only; no process or window is touched."""

from datetime import datetime, timezone

from app.workspace.workspace_restoration import WorkspaceRestorationPlanner, WorkspaceRestorer
from app.workspace.workspace_snapshot_store import WorkspaceSnapshot


def _snapshot() -> WorkspaceSnapshot:
    return WorkspaceSnapshot(
        id=1,
        work_thread_id=2,
        captured_at=datetime(2026, 10, 4, tzinfo=timezone.utc),
        application="Notepad",
        process_name="notepad.exe",
        executable_path=r"C:\\Windows\\notepad.exe",
        window_title="Notes",
    )


class _Finder:
    def __init__(self, matches=()):
        self.matches = matches
        self.paths = []

    def find_visible_windows_for_executable(self, executable_path):
        self.paths.append(executable_path)
        return self.matches


class _Activator:
    def __init__(self, error=None):
        self.handles = []
        self.error = error

    def activate(self, window_handle):
        self.handles.append(window_handle)
        if self.error:
            raise self.error


class _Launcher:
    def __init__(self, error=None):
        self.paths = []
        self.error = error

    def launch(self, executable_path):
        self.paths.append(executable_path)
        if self.error:
            raise self.error


def test_planner_prefers_existing_window_without_os_access() -> None:
    plan = WorkspaceRestorationPlanner().plan(_snapshot(), (101,))
    assert plan.action == "activate_existing"
    assert plan.window_handle == 101


def test_restorer_activates_match_or_launches_executable_with_no_arguments() -> None:
    activator = _Activator()
    launcher = _Launcher()
    activation_result = WorkspaceRestorer(_Finder((101,)), activator, launcher).restore(_snapshot())
    assert activation_result.success is True
    assert activation_result.action == "activate_existing"
    assert activator.handles == [101]
    assert launcher.paths == []

    launcher_result = WorkspaceRestorer(_Finder(), _Activator(), launcher).restore(_snapshot())
    assert launcher_result.success is True
    assert launcher_result.action == "launch_executable"
    assert launcher.paths == [_snapshot().executable_path]


def test_restorer_reports_os_failures_without_raising() -> None:
    result = WorkspaceRestorer(_Finder(), _Activator(), _Launcher(PermissionError("denied"))).restore(_snapshot())
    assert result.success is False
    assert result.action == "launch_executable"
    assert "denied" in result.message


def test_restorer_reports_window_inventory_failures_without_raising() -> None:
    class BrokenFinder:
        def find_visible_windows_for_executable(self, executable_path):
            raise PermissionError("cannot inspect windows")

    result = WorkspaceRestorer(BrokenFinder(), _Activator(), _Launcher()).restore(_snapshot())
    assert result.success is False
    assert result.action == "inspect_or_restore"
    assert "cannot inspect windows" in result.message
