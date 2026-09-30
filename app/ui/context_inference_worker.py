"""Qt worker that runs context-observation inference without blocking the interface thread."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from app.ui.context_observation_coordinator import ContextObservationCoordinator


class ContextInferenceWorker(QThread):
    """
    Run one `ContextObservationCoordinator.run_once()` pass on a
    dedicated Qt thread, separate from `MonitoringWorker`'s thread, so a
    slow or failing inference call can never delay activity polling or
    block the UI.

    A single instance performs exactly one pass then finishes (mirroring
    a one-shot background task); the caller (`MainWindow`) is responsible
    for starting a new instance on its own periodic timer, guarded
    against overlap by checking `isRunning()` before starting another --
    the same pattern already used for `MonitoringWorker`.
    """

    observations_recorded = Signal(int)
    coordinator_error = Signal(str)

    def __init__(self, coordinator: ContextObservationCoordinator) -> None:
        super().__init__()
        self._coordinator = coordinator

    def run(self) -> None:
        try:
            result = self._coordinator.run_once()
        except Exception as error:  # noqa: BLE001 -- must never crash the app; report instead
            self.coordinator_error.emit(str(error))
            return
        self.observations_recorded.emit(result.recorded_count)
