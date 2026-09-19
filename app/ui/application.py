"""Qt application assembly for the Adaptive Desktop AI desktop interface."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtWidgets import QApplication

from app.core.activity_monitor import WindowsActivityMonitor
from app.core.monitoring_service import ActivityMonitoringService
from app.database.activity_repository import ActivityRepository
from app.ui.dashboard_controller import DashboardController
from app.ui.main_window import MainWindow


def run_application(arguments: Sequence[str]) -> int:
    """Create and show the native Activity dashboard."""
    application = QApplication(list(arguments))
    repository = ActivityRepository()
    service = ActivityMonitoringService(WindowsActivityMonitor(), repository)
    window = MainWindow(DashboardController(repository, service))
    window.show()
    return application.exec()
