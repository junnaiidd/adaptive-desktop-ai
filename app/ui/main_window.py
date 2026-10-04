"""PySide6 Activity dashboard for the local Adaptive Desktop AI service."""

from __future__ import annotations

from datetime import datetime, timezone

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.ml.context_observation_store import ContextObservationStore
from app.ml.work_thread_store import WorkThreadStore
from app.ui.context_inference_worker import ContextInferenceWorker
from app.ui.context_observation_coordinator import ContextObservationCoordinator
from app.ui.dashboard_controller import DashboardController
from app.ui.monitoring_worker import MonitoringWorker


# How often to look for newly completed sessions that have no context observation yet.
CONTEXT_INFERENCE_INTERVAL_MS = 30_000


class MainWindow(QMainWindow):
    """The first native Activity interface, backed by the local monitoring service."""

    def __init__(
        self,
        controller: DashboardController,
        context_coordinator: ContextObservationCoordinator | None = None,
        work_thread_store: WorkThreadStore | None = None,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.context_coordinator = context_coordinator or self._build_default_coordinator(controller)
        self.work_thread_store = work_thread_store or self._build_default_work_thread_store(controller)
        self.worker: MonitoringWorker | None = None
        self.inference_worker: ContextInferenceWorker | None = None
        self.setWindowTitle("Adaptive Desktop AI")
        self.resize(1180, 760)
        self.setMinimumSize(960, 640)
        self._build_ui()
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self.refresh)
        self._refresh_timer.start(1_000)
        self._inference_timer = QTimer(self)
        self._inference_timer.timeout.connect(self._run_context_inference)
        self._inference_timer.start(CONTEXT_INFERENCE_INTERVAL_MS)
        QTimer.singleShot(0, self._run_context_inference)  # one pass at startup, then every interval
        self.refresh()

    @staticmethod
    def _build_default_coordinator(controller: DashboardController) -> ContextObservationCoordinator:
        """
        Self-wire the context-observation integration when the caller
        (e.g. `application.py`, unmodified by this milestone) does not
        supply a coordinator explicitly. Reuses the SAME database file
        `controller.repository` already points at, so the coordinator's
        writes and the dashboard's reads always agree on one store --
        and shares that one store with `controller` itself when it
        wasn't given one, so the "Latest context" card reflects what the
        coordinator actually persists.
        """
        store = controller.context_observation_store
        if store is None:
            store = ContextObservationStore(controller.repository.database_path)
            controller.context_observation_store = store
        return ContextObservationCoordinator(controller.repository, store)

    @staticmethod
    def _build_default_work_thread_store(controller: DashboardController) -> WorkThreadStore:
        """
        Self-wire the WorkThreadStore when the caller does not supply one explicitly.
        Reuses the SAME database file `controller.repository` already points at,
        and shares that store with `controller` so UI actions and dashboard snapshots
        read and write the identical database.
        """
        store = controller.work_thread_store
        if store is None:
            store = WorkThreadStore(controller.repository.database_path)
            controller.work_thread_store = store
        return store

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_sidebar())
        layout.addWidget(self._build_activity_page(), 1)
        self.setStyleSheet(_DARK_STYLESHEET)

    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(204)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(20, 24, 20, 24)
        layout.setSpacing(12)
        brand = QLabel("ADAPTIVE\nDESKTOP AI")
        brand.setObjectName("brand")
        layout.addWidget(brand)
        layout.addSpacing(32)
        for name, active in (("Activity", True), ("Context", False), ("Routines", False), ("Workspaces", False), ("Memory", False), ("Settings", False)):
            item = QLabel(name if active else f"{name}  ·  later")
            item.setObjectName("navActive" if active else "navMuted")
            layout.addWidget(item)
        layout.addStretch()
        privacy = QLabel("LOCAL ONLY\nNo input capture")
        privacy.setObjectName("privacy")
        layout.addWidget(privacy)
        return sidebar

    def _build_activity_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(34, 28, 34, 28)
        layout.setSpacing(18)

        header = QHBoxLayout()
        title_group = QVBoxLayout()
        title = QLabel("Activity")
        title.setObjectName("pageTitle")
        title_group.addWidget(title)
        subtitle = QLabel("A private, local view of your observed desktop flow.")
        subtitle.setObjectName("subtitle")
        title_group.addWidget(subtitle)
        header.addLayout(title_group)
        header.addStretch()
        self.status = QLabel()
        self.status.setObjectName("status")
        header.addWidget(self.status)
        self.toggle_button = QPushButton("Start monitoring")
        self.toggle_button.clicked.connect(self.toggle_monitoring)
        header.addWidget(self.toggle_button)
        layout.addLayout(header)

        cards = QHBoxLayout()
        cards.setSpacing(14)
        self.current_card, self.current_values = _card("Current activity", ("Application", "Process", "Window title", "Duration"))
        self.today_card, self.today_values = _card("Today", ("Observed", "Segments", "Sessions"))
        self.session_card, self.session_values = _card("Session", ("Status", "Duration", "Activities"))
        self.context_card, self.context_values = _card("Latest context", ("Context", "Session", "Observed"))
        cards.addWidget(self.current_card, 2)
        cards.addWidget(self.today_card, 1)
        cards.addWidget(self.session_card, 1)
        cards.addWidget(self.context_card, 1)
        layout.addLayout(cards)

        threads_panel = QFrame()
        threads_panel.setObjectName("panel")
        threads_layout = QVBoxLayout(threads_panel)
        threads_layout.setContentsMargins(18, 16, 18, 14)
        threads_layout.setSpacing(10)

        threads_title = QLabel("Work threads")
        threads_title.setObjectName("panelTitle")
        threads_layout.addWidget(threads_title)

        controls_row = QHBoxLayout()
        controls_row.setSpacing(10)

        self.thread_name_input = QLineEdit()
        self.thread_name_input.setPlaceholderText("New work thread name...")
        self.thread_name_input.returnPressed.connect(self._create_work_thread)
        controls_row.addWidget(self.thread_name_input, 2)

        self.create_thread_btn = QPushButton("Create thread")
        self.create_thread_btn.clicked.connect(self._create_work_thread)
        controls_row.addWidget(self.create_thread_btn)

        self.thread_dropdown = QComboBox()
        self.thread_dropdown.setMinimumWidth(200)
        controls_row.addWidget(self.thread_dropdown, 2)

        self.associate_btn = QPushButton("Associate latest context")
        self.associate_btn.clicked.connect(self._associate_latest_context)
        controls_row.addWidget(self.associate_btn)

        threads_layout.addLayout(controls_row)

        self.thread_feedback = QLabel()
        self.thread_feedback.setObjectName("caption")
        threads_layout.addWidget(self.thread_feedback)

        self.threads_table = QTableWidget(0, 3)
        self.threads_table.setHorizontalHeaderLabels(("ID", "Thread name", "Created"))
        self.threads_table.verticalHeader().setVisible(False)
        self.threads_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.threads_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.threads_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.threads_table.horizontalHeader().setStretchLastSection(True)
        self.threads_table.setColumnWidth(0, 60)
        self.threads_table.setColumnWidth(1, 240)
        self.threads_table.setMaximumHeight(110)
        self.threads_table.itemSelectionChanged.connect(self._on_thread_table_selection_changed)
        threads_layout.addWidget(self.threads_table)

        layout.addWidget(threads_panel)

        timeline_panel = QFrame()
        timeline_panel.setObjectName("panel")
        timeline_layout = QVBoxLayout(timeline_panel)
        timeline_layout.setContentsMargins(18, 16, 18, 14)
        timeline_title = QLabel("Recent activity")
        timeline_title.setObjectName("panelTitle")
        timeline_layout.addWidget(timeline_title)
        self.timeline = QTableWidget(0, 4)
        self.timeline.setHorizontalHeaderLabels(("Time", "Application", "Window title", "Duration"))
        self.timeline.verticalHeader().setVisible(False)
        self.timeline.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.timeline.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.timeline.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.timeline.horizontalHeader().setStretchLastSection(True)
        self.timeline.setColumnWidth(0, 82)
        self.timeline.setColumnWidth(1, 150)
        self.timeline.setColumnWidth(2, 420)
        timeline_layout.addWidget(self.timeline)
        layout.addWidget(timeline_panel, 1)

        self.footer = QLabel()
        self.footer.setObjectName("footer")
        layout.addWidget(self.footer)
        return page

    def _create_work_thread(self) -> None:
        name = self.thread_name_input.text().strip()
        if not name:
            self.thread_feedback.setText("Please enter a non-empty name for the work thread.")
            return
        try:
            created = self.work_thread_store.create_work_thread(
                name, created_at=datetime.now(timezone.utc)
            )
            self.thread_name_input.clear()
            self.thread_feedback.setText(f"Created work thread '{created.name}' (#{created.id})")
            self.refresh()
            index = self.thread_dropdown.findData(created.id)
            if index >= 0:
                self.thread_dropdown.setCurrentIndex(index)
        except Exception as e:
            self.thread_feedback.setText(f"Error creating work thread: {e}")

    def _associate_latest_context(self) -> None:
        selected_thread_id = self.thread_dropdown.currentData()
        if selected_thread_id is None:
            self.thread_feedback.setText("Please select a work thread first.")
            return
        selected_thread = self.work_thread_store.get_work_thread(selected_thread_id)
        thread_name = selected_thread.name if selected_thread else f"#{selected_thread_id}"

        recent_obs = None
        if self.controller.context_observation_store is not None:
            recent = self.controller.context_observation_store.list_recent_observations(limit=1)
            if recent:
                recent_obs = recent[0]

        if recent_obs is None:
            self.thread_feedback.setText("No context observation available to associate.")
            return

        try:
            self.work_thread_store.associate_observation(
                selected_thread_id,
                recent_obs.id,
                associated_at=datetime.now(timezone.utc),
            )
            self.thread_feedback.setText(
                f"Associated observation #{recent_obs.id} ({recent_obs.predicted_label}) with thread '{thread_name}'."
            )
            self.refresh()
        except Exception as e:
            self.thread_feedback.setText(f"Error associating context: {e}")

    def _on_thread_table_selection_changed(self) -> None:
        selected_items = self.threads_table.selectedItems()
        if not selected_items:
            return
        row = selected_items[0].row()
        id_item = self.threads_table.item(row, 0)
        if id_item:
            try:
                thread_id = int(id_item.text())
                index = self.thread_dropdown.findData(thread_id)
                if index >= 0:
                    self.thread_dropdown.setCurrentIndex(index)
            except ValueError:
                pass

    def toggle_monitoring(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.toggle_button.setEnabled(False)
            return
        self.worker = MonitoringWorker(self.controller.service)
        self.worker.started_monitoring.connect(self._monitoring_started)
        self.worker.stopped_monitoring.connect(self._monitoring_stopped)
        self.worker.monitoring_error.connect(self._monitoring_error)
        self.worker.start()

    def _monitoring_started(self) -> None:
        self.toggle_button.setText("Stop monitoring")
        self.toggle_button.setEnabled(True)
        self.refresh()

    def _monitoring_stopped(self) -> None:
        self.toggle_button.setText("Start monitoring")
        self.toggle_button.setEnabled(True)
        self.refresh()

    def _monitoring_error(self, error: str) -> None:
        self.status.setText(f"Monitoring error: {error}")

    def _run_context_inference(self) -> None:
        """Start one background inference pass unless one is already running (overlap guard)."""
        if self.context_coordinator is None:
            return
        if self.inference_worker is not None and self.inference_worker.isRunning():
            return
        self.inference_worker = ContextInferenceWorker(self.context_coordinator)
        self.inference_worker.coordinator_error.connect(self._context_inference_error)
        self.inference_worker.start()

    def _context_inference_error(self, error: str) -> None:
        self.status.setText(f"Context inference error: {error}")

    def refresh(self) -> None:
        running = self.worker is not None and self.worker.isRunning()
        snapshot = self.controller.snapshot(running)
        self.status.setText("●  Monitoring" if snapshot.monitoring_running else "●  Stopped")
        self.status.setProperty("running", snapshot.monitoring_running)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)
        self._set_card_values(self.current_values, (snapshot.current_application, snapshot.current_process, snapshot.current_title, snapshot.current_duration))
        self._set_card_values(self.today_values, (snapshot.total_today, str(snapshot.segment_count), str(snapshot.session_count)))
        self._set_card_values(self.session_values, (snapshot.session_label, snapshot.session_duration, str(snapshot.session_activity_count)))
        self._set_card_values(self.context_values, (snapshot.latest_context_label, snapshot.latest_context_session, snapshot.latest_context_observed_at))
        self.timeline.setRowCount(len(snapshot.timeline))
        for row, item in enumerate(snapshot.timeline):
            for column, value in enumerate((item.timestamp, item.application, item.window_title, item.duration)):
                table_item = QTableWidgetItem(value)
                table_item.setToolTip(value)
                self.timeline.setItem(row, column, table_item)

        # Update work threads table and dropdown
        current_selected_id = self.thread_dropdown.currentData()
        self.thread_dropdown.blockSignals(True)
        self.thread_dropdown.clear()

        threads = snapshot.work_threads
        self.threads_table.setRowCount(len(threads))
        for row, thread in enumerate(threads):
            formatted_time = thread.created_at.astimezone().strftime("%Y-%m-%d %H:%M")
            for col, val in enumerate((str(thread.id), thread.name, formatted_time)):
                table_item = QTableWidgetItem(val)
                table_item.setToolTip(val)
                self.threads_table.setItem(row, col, table_item)

            self.thread_dropdown.addItem(f"{thread.name} (#{thread.id})", thread.id)

        if current_selected_id is not None:
            idx = self.thread_dropdown.findData(current_selected_id)
            if idx >= 0:
                self.thread_dropdown.setCurrentIndex(idx)

        self.thread_dropdown.blockSignals(False)
        self.associate_btn.setEnabled(len(threads) > 0 and snapshot.latest_context_observation_id is not None)

        self.footer.setText(f"{snapshot.polling_interval}  ·  Database {snapshot.database_status}")

    @staticmethod
    def _set_card_values(labels: tuple[QLabel, ...], values: tuple[str, ...]) -> None:
        for label, value in zip(labels, values, strict=True):
            label.setText(value)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2_000)
        if self.inference_worker is not None and self.inference_worker.isRunning():
            self.inference_worker.wait(2_000)
        event.accept()


def _card(title: str, labels: tuple[str, ...]) -> tuple[QFrame, tuple[QLabel, ...]]:
    card = QFrame()
    card.setObjectName("card")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(18, 15, 18, 16)
    layout.setSpacing(5)
    heading = QLabel(title)
    heading.setObjectName("cardTitle")
    layout.addWidget(heading)
    value_labels: list[QLabel] = []
    for text in labels:
        caption = QLabel(text.upper())
        caption.setObjectName("caption")
        layout.addWidget(caption)
        value = QLabel("—")
        value.setObjectName("cardValue")
        value.setWordWrap(True)
        layout.addWidget(value)
        value_labels.append(value)
    return card, tuple(value_labels)


_DARK_STYLESHEET = """
QWidget#root { background: #11151b; color: #e7edf5; font-family: 'Segoe UI'; }
QFrame#sidebar { background: #0b0e13; border-right: 1px solid #222a35; }
QLabel#brand { color: #e8efff; font-size: 16px; font-weight: 700; letter-spacing: 1.5px; }
QLabel#navActive { background: #18283d; color: #9bc8ff; border-radius: 8px; padding: 10px 12px; font-weight: 600; }
QLabel#navMuted { color: #667080; padding: 10px 12px; }
QLabel#privacy { color: #586473; font-size: 10px; letter-spacing: 1px; }
QLabel#pageTitle { font-size: 28px; font-weight: 700; }
QLabel#subtitle, QLabel#footer { color: #8a96a6; }
QLabel#status { color: #8893a2; padding: 7px 12px; }
QLabel#status[running="true"] { color: #72dea0; }
QPushButton { background: #3e82ca; color: white; border: 0; border-radius: 7px; padding: 9px 16px; font-weight: 600; }
QPushButton:hover { background: #5597df; } QPushButton:disabled { background: #47515e; color: #a4acb8; }
QFrame#card, QFrame#panel { background: #171d26; border: 1px solid #27313e; border-radius: 10px; }
QLabel#cardTitle, QLabel#panelTitle { font-size: 14px; font-weight: 650; color: #dce6f3; }
QLabel#caption { color: #718094; font-size: 9px; letter-spacing: 0.8px; margin-top: 7px; }
QLabel#cardValue { color: #f0f4f9; font-size: 14px; }
QTableWidget { background: transparent; border: 0; gridline-color: #27313e; color: #dce5ef; }
QHeaderView::section { background: #121821; color: #78869a; border: 0; border-bottom: 1px solid #27313e; padding: 9px; font-size: 10px; }
QTableWidget::item { padding: 8px; border-bottom: 1px solid #202936; }
QLineEdit { background: #121821; color: #e7edf5; border: 1px solid #27313e; border-radius: 6px; padding: 7px 10px; font-size: 13px; }
QLineEdit:focus { border: 1px solid #3e82ca; }
QComboBox { background: #121821; color: #e7edf5; border: 1px solid #27313e; border-radius: 6px; padding: 6px 10px; font-size: 13px; }
QComboBox::drop-down { border: 0; }
QComboBox QAbstractItemView { background: #171d26; color: #e7edf5; selection-background-color: #264366; }
"""
