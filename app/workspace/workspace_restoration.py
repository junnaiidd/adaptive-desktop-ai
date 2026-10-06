"""Pure planning plus injected execution for best-effort workspace restoration."""

from __future__ import annotations

from dataclasses import dataclass

from app.workspace.windows_workspace_adapter import (
    ExecutableLauncher,
    ExistingWindowFinder,
    WindowActivator,
)
from app.workspace.workspace_snapshot_store import WorkspaceSnapshot


@dataclass(frozen=True, slots=True)
class WorkspaceRestorePlan:
    snapshot_id: int
    executable_path: str
    action: str
    window_handle: int | None = None


@dataclass(frozen=True, slots=True)
class WorkspaceRestoreResult:
    success: bool
    action: str
    message: str


class WorkspaceRestorationPlanner:
    """Choose an activation or no-argument launch without touching the desktop."""

    def plan(
        self, snapshot: WorkspaceSnapshot, matching_windows: tuple[int, ...]
    ) -> WorkspaceRestorePlan:
        if matching_windows:
            return WorkspaceRestorePlan(
                snapshot_id=snapshot.id,
                executable_path=snapshot.executable_path,
                action="activate_existing",
                window_handle=matching_windows[0],
            )
        return WorkspaceRestorePlan(
            snapshot_id=snapshot.id,
            executable_path=snapshot.executable_path,
            action="launch_executable",
        )


class WorkspaceRestorer:
    """Execute an already-approved plan through ports that tests can safely fake."""

    def __init__(
        self,
        window_finder: ExistingWindowFinder,
        window_activator: WindowActivator,
        executable_launcher: ExecutableLauncher,
        planner: WorkspaceRestorationPlanner | None = None,
    ) -> None:
        self.window_finder = window_finder
        self.window_activator = window_activator
        self.executable_launcher = executable_launcher
        self.planner = planner or WorkspaceRestorationPlanner()

    def restore(self, snapshot: WorkspaceSnapshot) -> WorkspaceRestoreResult:
        plan: WorkspaceRestorePlan | None = None
        try:
            plan = self.planner.plan(
                snapshot, self.window_finder.find_visible_windows_for_executable(snapshot.executable_path)
            )
            if plan.action == "activate_existing":
                assert plan.window_handle is not None
                self.window_activator.activate(plan.window_handle)
                return WorkspaceRestoreResult(
                    success=True,
                    action=plan.action,
                    message="Requested activation of an already running application window.",
                )
            self.executable_launcher.launch(plan.executable_path)
            return WorkspaceRestoreResult(
                success=True,
                action=plan.action,
                message="Requested a no-argument application launch.",
            )
        except Exception as error:  # OS launch/focus restrictions are expected best-effort outcomes.
            return WorkspaceRestoreResult(
                success=False,
                action=plan.action if plan is not None else "inspect_or_restore",
                message=f"Could not restore this application: {error}",
            )
