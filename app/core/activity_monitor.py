"""Windows foreground-window observation with privacy-aware event creation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
import platform
import re


_SENSITIVE_TITLE_PATTERNS = (
    re.compile(
        r"\b(password|passcode|one[- ]time code|otp|secret|private key|"
        r"recovery code|security code|api key|access token|cvv|credit card|"
        r"social security)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(messages?|chat|inbox|direct messages?|dm|mail)\b", re.IGNORECASE),
)
_REDACTED_TITLE = "[Sensitive window title hidden]"


@dataclass(frozen=True, slots=True)
class ActivityEvent:
    """A single, in-memory observation of the active desktop application."""

    timestamp: datetime
    application: str | None
    process_name: str | None
    window_title: str | None


def sanitize_window_title(title: str | None) -> str | None:
    """Normalize a title and suppress titles with obvious sensitive indicators.

    This is a small first privacy boundary, not a guarantee that all sensitive
    information can be detected from a window title. Future policy can extend
    this function without affecting consumers of :class:`ActivityEvent`.
    """
    if not isinstance(title, str):
        return None

    normalized_title = " ".join(title.split())
    if not normalized_title:
        return None

    if any(pattern.search(normalized_title) for pattern in _SENSITIVE_TITLE_PATTERNS):
        return _REDACTED_TITLE

    return normalized_title


class WindowsActivityMonitor:
    """Read the current Windows foreground window without recording input."""

    def get_current_activity(self) -> ActivityEvent | None:
        """Return the current foreground activity, or ``None`` when unavailable."""
        if platform.system() != "Windows":
            return None

        foreground_window = self._read_foreground_window()
        if foreground_window is None:
            return None

        window_title, process_id = foreground_window
        process_name = self._get_process_name(process_id)
        return ActivityEvent(
            timestamp=datetime.now(timezone.utc),
            application=self._application_name(process_name),
            process_name=process_name,
            window_title=sanitize_window_title(window_title),
        )

    @staticmethod
    def _read_foreground_window() -> tuple[str | None, int] | None:
        """Retrieve foreground-window metadata through the Windows API."""
        try:
            import win32gui
            import win32process

            window_handle = win32gui.GetForegroundWindow()
            if not window_handle:
                return None

            window_title = win32gui.GetWindowText(window_handle)
            _, process_id = win32process.GetWindowThreadProcessId(window_handle)
            if not process_id:
                return None

            return window_title, process_id
        except Exception:
            # Win32 calls can fail for an invalid handle or protected process.
            return None

    @staticmethod
    def _get_process_name(process_id: int) -> str | None:
        """Return a process name while tolerating disappearing or protected processes."""
        try:
            import psutil

            process_name = psutil.Process(process_id).name()
            return process_name.strip() or None
        except Exception:
            # A process can exit or deny metadata access between API calls.
            return None

    @staticmethod
    def _application_name(process_name: str | None) -> str | None:
        """Derive a simple display name from the process executable name."""
        if not process_name:
            return None

        application_name, _ = os.path.splitext(process_name)
        return application_name or process_name
