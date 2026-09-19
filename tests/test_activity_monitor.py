"""Tests for behavior that does not require a live Windows desktop."""

from dataclasses import asdict
from datetime import datetime, timezone

from app.core import activity_monitor
from app.core.activity_monitor import ActivityEvent, WindowsActivityMonitor, sanitize_window_title


def test_activity_event_has_the_expected_structure() -> None:
    event = ActivityEvent(
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        application="Code",
        process_name="Code.exe",
        window_title="main.py - adaptive-desktop-ai",
    )

    assert tuple(asdict(event)) == (
        "timestamp",
        "application",
        "process_name",
        "window_title",
    )
    assert event.process_name == "Code.exe"


def test_sanitize_window_title_handles_empty_and_invalid_values() -> None:
    assert sanitize_window_title(None) is None
    assert sanitize_window_title("") is None
    assert sanitize_window_title("   \t\n") is None
    assert sanitize_window_title(42) is None  # type: ignore[arg-type]


def test_sanitize_window_title_normalizes_and_redacts_sensitive_content() -> None:
    assert sanitize_window_title("  Project   notes  ") == "Project notes"
    assert sanitize_window_title("Reset password - Browser") == "[Sensitive window title hidden]"
    assert sanitize_window_title("Inbox - Mail") == "[Sensitive window title hidden]"


def test_monitor_creates_event_when_process_information_is_missing(monkeypatch) -> None:
    monitor = WindowsActivityMonitor()
    monkeypatch.setattr(activity_monitor.platform, "system", lambda: "Windows")
    monkeypatch.setattr(monitor, "_read_foreground_window", lambda: ("Untitled - Editor", 123))
    monkeypatch.setattr(monitor, "_get_process_name", lambda _process_id: None)

    event = monitor.get_current_activity()

    assert event is not None
    assert event.application is None
    assert event.process_name is None
    assert event.window_title == "Untitled - Editor"
    assert event.timestamp.tzinfo is timezone.utc


def test_monitor_handles_unavailable_foreground_window(monkeypatch) -> None:
    monitor = WindowsActivityMonitor()
    monkeypatch.setattr(activity_monitor.platform, "system", lambda: "Windows")
    monkeypatch.setattr(monitor, "_read_foreground_window", lambda: None)

    assert monitor.get_current_activity() is None


def test_monitor_is_graceful_on_non_windows(monkeypatch) -> None:
    monitor = WindowsActivityMonitor()
    monkeypatch.setattr(activity_monitor.platform, "system", lambda: "Linux")

    assert monitor.get_current_activity() is None
