"""Small Windows-only adapters for explicit foreground-app capture and restore."""

from __future__ import annotations

import os
from pathlib import Path
import platform
import subprocess
from typing import Protocol

from app.core.activity_monitor import sanitize_window_title
from app.workspace.workspace_snapshot_store import CapturedForegroundWindow


class WorkspaceCaptureError(RuntimeError):
    """A user-safe failure while reading the explicitly requested foreground window."""


class ForegroundWindowProbe(Protocol):
    def capture_foreground_window(self) -> CapturedForegroundWindow: ...


class ExistingWindowFinder(Protocol):
    def find_visible_windows_for_executable(self, executable_path: str) -> tuple[int, ...]: ...


class WindowActivator(Protocol):
    def activate(self, window_handle: int) -> None: ...


class ExecutableLauncher(Protocol):
    def launch(self, executable_path: str) -> None: ...


_BROWSER_EXECUTABLES = {"chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe"}


class WindowsForegroundWindowProbe:
    """Capture only the current conventional desktop app after a user clicks Capture."""

    def capture_foreground_window(self) -> CapturedForegroundWindow:
        if platform.system() != "Windows":
            raise WorkspaceCaptureError("Workspace capture is available only on Windows.")
        try:
            import psutil
            import win32gui
            import win32process

            window_handle = win32gui.GetForegroundWindow()
            if not window_handle or not win32gui.IsWindowVisible(window_handle):
                raise WorkspaceCaptureError("No visible foreground desktop window is available to capture.")
            raw_title = win32gui.GetWindowText(window_handle)
            _, process_id = win32process.GetWindowThreadProcessId(window_handle)
            if not process_id:
                raise WorkspaceCaptureError("The foreground window did not expose a process.")
            if process_id == os.getpid():
                # Safety net: never snapshot Adaptive Desktop AI itself (e.g. python.exe).
                raise WorkspaceCaptureError(
                    "Adaptive Desktop AI was the foreground window, so no other application could be captured. "
                    "Switch to the application you want to capture and try again."
                )
            process = psutil.Process(process_id)
            process_name = process.name().strip()
            executable_path = process.exe().strip()
        except WorkspaceCaptureError:
            raise
        except Exception as error:  # Win32 and process metadata can legitimately disappear or deny access.
            raise WorkspaceCaptureError("Could not access the foreground application.") from error

        if not process_name or not executable_path or not Path(executable_path).is_file():
            raise WorkspaceCaptureError("The foreground application does not expose a launchable executable path.")
        if process_name.casefold() in _BROWSER_EXECUTABLES:
            raise WorkspaceCaptureError("Browser windows are not supported by Workspace Restore v1.")

        application, _ = os.path.splitext(process_name)
        return CapturedForegroundWindow(
            application=application or process_name,
            process_name=process_name,
            executable_path=executable_path,
            window_title=sanitize_window_title(raw_title),
        )


class WindowsExistingWindowFinder:
    """Find visible top-level windows belonging to a captured executable."""

    def find_visible_windows_for_executable(self, executable_path: str) -> tuple[int, ...]:
        if platform.system() != "Windows":
            return ()
        try:
            import psutil
            import win32gui
            import win32process
        except Exception:
            return ()

        target = _normalized_path(executable_path)
        matches: list[int] = []

        def consider(window_handle: int, _unused: object) -> bool:
            try:
                if not win32gui.IsWindowVisible(window_handle):
                    return True
                _, process_id = win32process.GetWindowThreadProcessId(window_handle)
                if process_id and _normalized_path(psutil.Process(process_id).exe()) == target:
                    matches.append(window_handle)
            except Exception:
                pass
            return True

        try:
            win32gui.EnumWindows(consider, None)
        except Exception:
            return ()
        return tuple(matches)


class WindowsWindowActivator:
    """Attempt foreground activation; Windows focus policy may still refuse it."""

    def activate(self, window_handle: int) -> None:
        import win32gui

        win32gui.SetForegroundWindow(window_handle)


class SubprocessExecutableLauncher:
    """Launch only the captured executable, never command-line arguments or documents."""

    def launch(self, executable_path: str) -> None:
        subprocess.Popen([executable_path])  # noqa: S603 -- path was explicitly captured from a running process


def _normalized_path(value: str) -> str:
    return os.path.normcase(os.path.normpath(value))
