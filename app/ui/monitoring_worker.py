"""Qt worker that runs monitoring without blocking the interface thread."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from app.core.monitoring_service import ActivityMonitoringService


class MonitoringWorker(QThread):
    """Own the blocking monitoring loop on a dedicated Qt thread."""

    started_monitoring = Signal()
    stopped_monitoring = Signal()
    monitoring_error = Signal(str)

    def __init__(self, service: ActivityMonitoringService) -> None:
        super().__init__()
        self._service = service

    def run(self) -> None:
        self.started_monitoring.emit()
        try:
            self._service.run()
        except Exception as error:
            self.monitoring_error.emit(str(error))
        finally:
            self.stopped_monitoring.emit()

    def stop(self) -> None:
        """Request service shutdown; the worker exits after the current poll."""
        self._service.request_stop()
