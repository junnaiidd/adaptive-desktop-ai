import os, sys, tempfile, pathlib
if sys.platform!="win32": os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
from types import SimpleNamespace
from PySide6.QtWidgets import QApplication, QLayout, QWidget, QScrollArea
from app.database.activity_repository import ActivityRepository
from app.ui.dashboard_controller import DashboardController
from app.ui.main_window import MainWindow
app=QApplication([]); d=pathlib.Path(tempfile.mkdtemp())
svc=SimpleNamespace(poll_interval_seconds=5.0,current_activity=None,current_duration_seconds=None,session_manager=SimpleNamespace(current_session=None))
w=MainWindow(DashboardController(ActivityRepository(d/"a.db"),svc))
print("style:",app.style().objectName(),"| font:",app.font().family(),app.font().pointSizeF())
page=w.centralWidget().layout().itemAt(1).widget()
def lay(l,depth,label):
    # report each item of a layout with its minimum width
    pad="  "*depth
    print(f"{pad}{label}: layout min w={l.totalMinimumSize().width()} spacing={l.spacing()} margins=({l.contentsMargins().left()},{l.contentsMargins().right()})")
    for i in range(l.count()):
        it=l.itemAt(i); wd=it.widget(); sub=it.layout()
        if wd is not None:
            nm=f"{type(wd).__name__}#{wd.objectName() or (wd.text()[:24] if hasattr(wd,'text') else '-')}"
            print(f"{pad}  [{i}] {nm} minW={it.minimumSize().width()} hintW={wd.sizeHint().width()} minSz={wd.minimumSize().width()}")
            if wd.layout() is not None and not isinstance(wd,QScrollArea) and depth<MAXD: lay(wd.layout(),depth+2,"  ↳ "+nm)
        elif sub is not None:
            print(f"{pad}  [{i}] <sub-layout> minW={it.minimumSize().width()}")
            if depth<MAXD: lay(sub,depth+2,"  ↳ sublayout")
        else: print(f"{pad}  [{i}] spacer minW={it.minimumSize().width()}")
MAXD=int(sys.argv[1]) if len(sys.argv)>1 else 8
print("PAGE minimumSizeHint width =",page.minimumSizeHint().width(),"| sidebar =",w.centralWidget().layout().itemAt(0).widget().width())
lay(page.layout(),0,"page")
