"""Mocked tests for the Windows-only foreground capture adapter."""

from types import SimpleNamespace
import sys

import pytest

from app.workspace.windows_workspace_adapter import WindowsForegroundWindowProbe, WorkspaceCaptureError


def _install_windows_modules(monkeypatch, process, *, title="Notes", visible=True) -> None:
    monkeypatch.setattr("app.workspace.windows_workspace_adapter.platform.system", lambda: "Windows")
    monkeypatch.setitem(
        sys.modules,
        "win32gui",
        SimpleNamespace(
            GetForegroundWindow=lambda: 42,
            IsWindowVisible=lambda _handle: visible,
            GetWindowText=lambda _handle: title,
        ),
    )
    monkeypatch.setitem(sys.modules, "win32process", SimpleNamespace(GetWindowThreadProcessId=lambda _h: (1, 99)))
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=lambda _pid: process))


def test_capture_reads_one_foreground_window_and_sanitizes_title(tmp_path, monkeypatch) -> None:
    executable = tmp_path / "notepad.exe"
    executable.touch()
    process = SimpleNamespace(name=lambda: "notepad.exe", exe=lambda: str(executable))
    _install_windows_modules(monkeypatch, process, title="Reset password - Notes")

    captured = WindowsForegroundWindowProbe().capture_foreground_window()

    assert captured.application == "notepad"
    assert captured.process_name == "notepad.exe"
    assert captured.executable_path == str(executable)
    assert captured.window_title == "[Sensitive window title hidden]"


def test_capture_rejects_browser_and_unavailable_windows(tmp_path, monkeypatch) -> None:
    executable = tmp_path / "chrome.exe"
    executable.touch()
    process = SimpleNamespace(name=lambda: "chrome.exe", exe=lambda: str(executable))
    _install_windows_modules(monkeypatch, process)
    with pytest.raises(WorkspaceCaptureError, match="Browser windows"):
        WindowsForegroundWindowProbe().capture_foreground_window()

    _install_windows_modules(monkeypatch, process, visible=False)
    with pytest.raises(WorkspaceCaptureError, match="No visible foreground"):
        WindowsForegroundWindowProbe().capture_foreground_window()


def test_capture_refuses_adaptive_desktop_ai_itself(tmp_path, monkeypatch) -> None:
    """Regression: clicking Capture makes this app foreground; it must never snapshot itself."""
    import os

    executable = tmp_path / "python.exe"
    executable.touch()
    process = SimpleNamespace(name=lambda: "python.exe", exe=lambda: str(executable))
    _install_windows_modules(monkeypatch, process, title="Adaptive Desktop AI")
    monkeypatch.setitem(sys.modules, "win32process", SimpleNamespace(GetWindowThreadProcessId=lambda _h: (1, os.getpid())))

    with pytest.raises(WorkspaceCaptureError, match="Adaptive Desktop AI was the foreground window"):
        WindowsForegroundWindowProbe().capture_foreground_window()
