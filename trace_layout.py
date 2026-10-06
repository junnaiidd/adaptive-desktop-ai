"""Print the live two-panel Activity layout geometry. Run from the repo root:  python trace_layout.py [W H]
Uses a temporary database, so it never touches your real data."""
import os, sys, tempfile, pathlib
if sys.platform != "win32": os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from types import SimpleNamespace
from datetime import datetime, timezone
from PySide6.QtWidgets import QApplication, QScrollArea
from app.database.activity_repository import ActivityRepository
from app.ml.work_thread_store import WorkThreadStore
from app.ui.dashboard_controller import DashboardController
from app.ui.main_window import MainWindow
W, H = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (1180, 760)
app = QApplication([]); d = pathlib.Path(tempfile.mkdtemp())
svc = SimpleNamespace(poll_interval_seconds=5.0, current_activity=None, current_duration_seconds=None,
                      session_manager=SimpleNamespace(current_session=None))
WorkThreadStore(d / "a.db").create_work_thread("Trace", created_at=datetime.now(timezone.utc))
w = MainWindow(DashboardController(ActivityRepository(d / "a.db"), svc))
w.resize(W, H); w.show(); w.refresh(); app.processEvents(); app.processEvents()
sa = w.findChild(QScrollArea, "pageScroll")
print("scroll area present:", sa is not None, "(must be True)")
work, rec = sa.parentWidget(), w.timeline.parentWidget()
pos = lambda x: x.mapTo(w, x.rect().topLeft())
print(f"window {w.width()}x{w.height()}  dpr={w.devicePixelRatio()}")
print(f"work   panel x={pos(work).x()} y={pos(work).y()} {work.width()}x{work.height()}")
print(f"recent panel x={pos(rec).x()} y={pos(rec).y()} {rec.width()}x{rec.height()}")
print(f"work share = {100*work.width()/(work.width()+rec.width()):.0f}%  (expect ~55%)")
print(f"work scroll: content={sa.widget().height()} viewport={sa.viewport().height()} range={sa.verticalScrollBar().maximum()}")
print("tables:", {n: getattr(w, n).height() for n in ("threads_table", "tasks_table", "unfinished_table", "timeline")})
