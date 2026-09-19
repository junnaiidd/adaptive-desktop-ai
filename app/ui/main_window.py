"""PySide6 Activity dashboard for the local Adaptive Desktop AI service."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.ui.dashboard_controller import DashboardController
from app.ui.monitoring_worker import MonitoringWorker


class MainWindow(QMainWindow):
    """The first native Activity interface, backed by the local monitoring service."""

    def __init__(self, controller: DashboardController) -> None:
        super().__init__()
        self.controller = controller
        self.worker: MonitoringWorker | None = None
        self.setWindowTitle("Adaptive Desktop AI")
        self.resize(1180, 760)
        self.setMinimumSize(960, 640)
        self._build_ui()
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self.refresh)
        self._refresh_timer.start(1_000)
        self.refresh()

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
        cards.addWidget(self.current_card, 2)
        cards.addWidget(self.today_card, 1)
        cards.addWidget(self.session_card, 1)
        layout.addLayout(cards)

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
        self.timeline.setRowCount(len(snapshot.timeline))
        for row, item in enumerate(snapshot.timeline):
            for column, value in enumerate((item.timestamp, item.application, item.window_title, item.duration)):
                table_item = QTableWidgetItem(value)
                table_item.setToolTip(value)
                self.timeline.setItem(row, column, table_item)
        self.footer.setText(f"{snapshot.polling_interval}  ·  Database {snapshot.database_status}")

    @staticmethod
    def _set_card_values(labels: tuple[QLabel, ...], values: tuple[str, ...]) -> None:
        for label, value in zip(labels, values, strict=True):
            label.setText(value)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2_000)
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
"""
