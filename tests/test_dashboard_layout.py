"""Regression: lower Activity panels must never be squeezed into thin, clipped strips."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QScrollArea

from app.database.activity_repository import ActivityRepository
from app.ui.dashboard_controller import DashboardController
from app.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _window(tmp_path) -> MainWindow:
    service = SimpleNamespace(
        poll_interval_seconds=5.0,
        current_activity=None,
        current_duration_seconds=None,
        session_manager=SimpleNamespace(current_session=None),
    )
    return MainWindow(DashboardController(ActivityRepository(tmp_path / "activity.db"), service))


@pytest.mark.parametrize("size", [(960, 640), (1180, 760)])
def test_lower_tables_keep_readable_heights_at_supported_window_sizes(tmp_path, qapp, size) -> None:
    window = _window(tmp_path)
    window.resize(*size)
    window.show()
    qapp.processEvents()

    for table in (window.threads_table, window.tasks_table, window.unfinished_table):
        assert table.height() >= 100
    # Side-by-side layout: Recent activity gets the full lower-area height but no longer a stacked 200px floor.
    assert window.timeline.height() >= 110
    window.close()


def test_lower_panels_scroll_instead_of_collapsing(tmp_path, qapp) -> None:
    window = _window(tmp_path)
    scroll = window.findChild(QScrollArea, "pageScroll")

    assert scroll is not None and scroll.widgetResizable()
    # The pinned part of the page (header, cards, scroll viewport, footer) must fit the minimum window height.
    page = window.centralWidget().layout().itemAt(1).widget()
    assert page.minimumSizeHint().height() <= window.minimumSize().height()
    window.close()


@pytest.mark.parametrize("size", [(960, 640), (1180, 760)])
def test_no_lower_panel_widget_is_laid_out_below_its_own_minimum(tmp_path, qapp, size) -> None:
    """Mechanism-independent check: a squeezed layout shows up as a widget shorter than its minimum."""
    from datetime import datetime, timezone

    from PySide6.QtWidgets import (
        QComboBox, QFrame, QHeaderView, QLabel, QLineEdit, QPushButton, QTableWidget, QWidget,
    )

    from app.ml.work_thread_store import WorkThreadStore

    WorkThreadStore(tmp_path / "activity.db").create_work_thread("Report", created_at=datetime.now(timezone.utc))
    window = _window(tmp_path)
    window.resize(*size)
    window.show()
    window.refresh()
    qapp.processEvents()

    content = window.findChild(QWidget, "pageScrollContent")
    assert content is not None
    assert content.height() >= content.minimumSizeHint().height()
    recent_panel = window.timeline.parentWidget()
    checked = (QFrame, QLabel, QPushButton, QLineEdit, QComboBox, QTableWidget)
    squeezed = [
        f"{type(w).__name__}#{w.objectName() or w.text() if hasattr(w, 'text') else w.objectName()} "
        f"h={w.height()} < min={max(w.minimumSize().height(), w.minimumSizeHint().height())}"
        for w in (*content.findChildren(QWidget), recent_panel, *recent_panel.findChildren(QWidget))
        if isinstance(w, checked)
        and not isinstance(w, QHeaderView)  # a table's own header bar is internal, not dashboard layout
        and w.isVisibleTo(window)
        and w.height() < max(w.minimumSize().height(), w.minimumSizeHint().height())
    ]
    assert squeezed == []
    window.close()


# ---------------------------------------------------------------------------
# Two-column lower area: Work / Resume (left) | Recent activity (right)
# ---------------------------------------------------------------------------


def _shown(tmp_path, qapp, size):
    from datetime import datetime, timezone

    from app.ml.task_store import TaskStore
    from app.ml.work_thread_store import WorkThreadStore

    now = datetime.now(timezone.utc)
    thread = WorkThreadStore(tmp_path / "activity.db").create_work_thread("Report", created_at=now)
    TaskStore(tmp_path / "activity.db").create_task(thread.id, "Draft intro", created_at=now)
    window = _window(tmp_path)
    window.resize(*size)
    window.show()
    window.refresh()
    qapp.processEvents()
    return window


def _panels(window):
    scroll = window.findChild(QScrollArea, "pageScroll")
    work_panel = scroll.parentWidget()
    recent_panel = window.timeline.parentWidget()
    return scroll, work_panel, recent_panel


def _top_left(widget, window):
    return widget.mapTo(window, widget.rect().topLeft())


@pytest.mark.parametrize("size", [(960, 640), (1180, 760)])
def test_work_and_recent_activity_panels_sit_side_by_side(tmp_path, qapp, size) -> None:
    window = _shown(tmp_path, qapp, size)
    _scroll, work, recent = _panels(window)
    work_pos, recent_pos = _top_left(work, window), _top_left(recent, window)

    assert work_pos.y() == recent_pos.y() and work.height() == recent.height()  # one row, same height
    assert work_pos.x() + work.width() <= recent_pos.x()  # left | right, not overlapping
    share = work.width() / (work.width() + recent.width())
    assert 0.50 <= share <= 0.60  # ~55% / ~45%
    assert work.width() >= 340 and recent.width() >= 280  # both usable even at the 960px minimum
    window.close()


def test_work_panel_contains_all_existing_work_resume_controls(tmp_path, qapp) -> None:
    from PySide6.QtWidgets import QLabel

    window = _shown(tmp_path, qapp, (1180, 760))
    scroll, work, recent = _panels(window)
    content = scroll.widget()

    for name in (
        "thread_name_input", "create_thread_btn", "thread_dropdown", "associate_btn", "threads_table",
        "task_input", "add_task_btn", "toggle_task_btn", "delete_task_btn", "tasks_table",
        "capture_workspace_btn", "restore_workspace_btn", "workspace_snapshot_label",
        "mark_done_btn", "goto_thread_btn", "unfinished_table",
    ):
        assert content.isAncestorOf(getattr(window, name)), name
    headings = {label.text() for label in content.findChildren(QLabel) if label.objectName() == "panelTitle"}
    assert {"Work threads", "Tasks", "Resume work", "Unfinished work"} <= headings
    assert not work.isAncestorOf(window.timeline) and recent.isAncestorOf(window.timeline)
    window.close()


@pytest.mark.parametrize("size", [(960, 640), (1180, 760)])
def test_each_panel_scrolls_itself_and_the_page_does_not(tmp_path, qapp, size) -> None:
    from PySide6.QtCore import Qt as _Qt
    from PySide6.QtWidgets import QTableWidgetItem

    window = _shown(tmp_path, qapp, size)
    scroll, work, recent = _panels(window)
    work_height_before = work.height()

    # Work panel owns the only scroll area; nothing scrolls the page as a whole.
    assert work.isAncestorOf(scroll) and window.findChildren(QScrollArea) == [scroll]
    assert scroll.verticalScrollBar().maximum() > 0  # more content than fits -> scrolls inside the panel

    # Recent activity: hundreds of rows scroll inside its own table without growing either panel.
    window.timeline.setRowCount(300)
    for row in range(300):
        for column in range(4):
            window.timeline.setItem(row, column, QTableWidgetItem(f"r{row}c{column}"))
    qapp.processEvents()
    assert window.timeline.verticalScrollBarPolicy() != _Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert window.timeline.verticalScrollBar().maximum() > 0
    assert work.height() == work_height_before and recent.height() == work_height_before

    # No clipped-looking horizontal scrollbars inside the panels at this size.
    assert not window.timeline.horizontalScrollBar().isVisible()
    for table in (window.threads_table, window.tasks_table, window.unfinished_table):
        assert not table.horizontalScrollBar().isVisible()
    window.close()


def test_page_minimum_size_fits_the_minimum_window(tmp_path, qapp) -> None:
    window = _window(tmp_path)
    page = window.centralWidget().layout().itemAt(1).widget()
    hint = page.minimumSizeHint()
    sidebar_width = window.centralWidget().layout().itemAt(0).widget().width()

    assert hint.height() <= window.minimumSize().height()
    assert hint.width() + sidebar_width <= window.minimumSize().width()
    window.close()


@pytest.mark.parametrize("font_px", [12, 16, 22, 26])
def test_page_minimum_width_stays_within_the_window_minimum_for_larger_fonts(tmp_path, qapp, font_px) -> None:
    """Regression: an unwrappable header subtitle made the page demand ~1100px on Windows font metrics."""
    from PySide6.QtGui import QFont

    original = qapp.font()
    larger = QFont(original)
    larger.setPixelSize(font_px)
    qapp.setFont(larger)
    try:
        window = _window(tmp_path)
        page = window.centralWidget().layout().itemAt(1).widget()
        sidebar_width = window.centralWidget().layout().itemAt(0).widget().width()
        assert page.minimumSizeHint().width() + sidebar_width <= window.minimumSize().width()
        window.close()
    finally:
        qapp.setFont(original)


@pytest.mark.parametrize("size", [(960, 640), (1180, 760)])
def test_header_subtitle_keeps_its_full_single_line_when_there_is_room(tmp_path, qapp, size) -> None:
    """The minimum-width fix must not change the normal look: no wrapping or clipping at default fonts."""
    from PySide6.QtWidgets import QLabel

    window = _window(tmp_path)
    window.resize(*size)
    window.show()
    qapp.processEvents()
    subtitle = window.findChild(QLabel, "subtitle")

    # Contract: fully visible on one line whenever it fits the page; if the text is wider than the whole page
    # (extreme fonts) it is clipped to the page width, but never widens the page (see the minimum-width tests).
    page_width = window.context_card.mapTo(window, window.context_card.rect().topRight()).x() - window.current_card.mapTo(
        window, window.current_card.rect().topLeft()
    ).x()
    assert subtitle.width() >= min(subtitle.fontMetrics().horizontalAdvance(subtitle.text()), page_width)
    assert subtitle.height() <= subtitle.fontMetrics().lineSpacing() + 4  # one line, not wrapped
    window.close()


# ---------------------------------------------------------------------------
# Minimum-width contract: no text, font or glyph width may widen the page past the window minimum
# ---------------------------------------------------------------------------

_PAGE_BUDGET = 960 - 204  # MainWindow minimum width minus the fixed sidebar


def _page_min_width(window) -> int:
    page = window.centralWidget().layout().itemAt(1).widget()
    page.layout().invalidate()
    return page.minimumSizeHint().width()


@pytest.mark.parametrize("stretch", [100, 170, 250])
@pytest.mark.parametrize("font_px", [12, 26])
def test_page_minimum_width_is_independent_of_glyph_width_and_font_size(tmp_path, qapp, font_px, stretch) -> None:
    """Wide fonts (Windows metrics are ~1.7x the Linux ones) must clip, never widen the page."""
    from PySide6.QtGui import QFont

    original = qapp.font()
    stressed = QFont(original)
    stressed.setPixelSize(font_px)
    stressed.setStretch(stretch)
    qapp.setFont(stressed)
    try:
        window = _window(tmp_path)
        assert _page_min_width(window) <= _PAGE_BUDGET
        window.close()
    finally:
        qapp.setFont(original)


def test_no_header_or_footer_text_length_can_widen_the_page(tmp_path, qapp) -> None:
    """Header/footer sizing must not carry an intrinsic-width constraint."""
    window = _window(tmp_path)
    baseline = _page_min_width(window)
    assert baseline <= _PAGE_BUDGET

    from PySide6.QtWidgets import QLabel

    long_text = "wide text " * 60
    window.findChild(QLabel, "subtitle").setText(long_text)
    window.status.setText(long_text)
    window.footer.setText(long_text)
    window.toggle_button.setText(long_text)

    assert _page_min_width(window) == baseline
    window.close()


@pytest.mark.parametrize("size", [(960, 640), (1180, 760)])
def test_header_geometry_has_no_overlap_and_subtitle_spans_the_page_width(tmp_path, qapp, size) -> None:
    """Subtitle contract: it owns a full-width row below the title row, so title/status/button never share it."""
    from PySide6.QtWidgets import QLabel

    window = _shown(tmp_path, qapp, size)
    title = window.findChild(QLabel, "pageTitle")
    subtitle = window.findChild(QLabel, "subtitle")

    def rect(widget):
        top_left = widget.mapTo(window, widget.rect().topLeft())
        return top_left.x(), top_left.y(), top_left.x() + widget.width(), top_left.y() + widget.height()

    row = [rect(title), rect(window.status), rect(window.toggle_button)]
    for i, a in enumerate(row):
        for b in row[i + 1:]:
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]  # no overlap
    sub = rect(subtitle)
    assert sub[1] >= max(r[3] for r in (rect(title),))  # subtitle sits below the title row
    cards_left = rect(window.current_card)[0]
    cards_right = rect(window.context_card)[2]
    assert (sub[0], sub[2]) == (cards_left, cards_right)  # spans the same width as the cards row
    window.close()


@pytest.mark.parametrize("size", [(960, 640), (1180, 760)])
def test_card_titles_are_never_clipped_where_the_declared_minimum_covers_the_text(tmp_path, qapp, size) -> None:
    """Explicit minimums must not undercut text that fits (normal fonts: all four cards).

    On machines whose text is so wide that four cards cannot fit in the page, clipping is the defined, bounded
    behaviour (see the stress tests); this check then has nothing to assert for those cards.
    """
    from PySide6.QtWidgets import QLabel

    window = _shown(tmp_path, qapp, size)
    for card in (window.current_card, window.today_card, window.session_card, window.context_card):
        natural_min = card.layout().totalMinimumSize().width()
        if natural_min <= card.minimumWidth():
            heading = card.findChild(QLabel, "cardTitle")
            assert heading.width() >= heading.fontMetrics().horizontalAdvance(heading.text()), heading.text()
    window.close()


def test_work_panel_controls_stay_reachable_with_extreme_fonts(tmp_path, qapp) -> None:
    """If the Work controls cannot fit, the panel scrolls horizontally instead of hiding them."""
    from PySide6.QtGui import QFont

    original = qapp.font()
    stressed = QFont(original)
    stressed.setPixelSize(26)
    stressed.setStretch(250)
    qapp.setFont(stressed)
    try:
        window = _shown(tmp_path, qapp, (960, 640))
        scroll, _work, _recent = _panels(window)
        content = scroll.widget()
        for button in (window.create_thread_btn, window.associate_btn, window.capture_workspace_btn):
            assert button.mapTo(content, button.rect().topRight()).x() <= content.width()
        assert content.width() >= content.minimumSizeHint().width()
        window.close()
    finally:
        qapp.setFont(original)
