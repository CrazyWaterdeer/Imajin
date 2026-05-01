"""Dock-scoped Qt theme: dark background with coral accent.

Adapted from SCAT (Spot Classification and Analysis Tool, sibling project).
Applied per-dock via setStyleSheet so napari's window chrome stays untouched.
"""
from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QWheelEvent
from qtpy.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox, QWidget


class Theme:
    PRIMARY = "#DA4E42"
    PRIMARY_DARK = "#C44539"
    PRIMARY_LIGHT = "#E8695E"

    SECONDARY = "#636867"
    SECONDARY_DARK = "#525756"
    SECONDARY_LIGHT = "#7A7F7E"

    BG_DARKEST = "#0A0A0A"
    BG_DARK = "#121212"
    BG_MEDIUM = "#101010"
    BG_LIGHT = "#242424"
    BG_LIGHTER = "#2E2E2E"

    TEXT_PRIMARY = "#FFFFFF"
    TEXT_SECONDARY = "#9A9A9A"
    TEXT_MUTED = "#5A5A5A"

    BORDER = "#2A2A2A"

    @classmethod
    def get_dock_stylesheet(cls) -> str:
        return f"""
            QWidget {{
                background-color: {cls.BG_DARKEST};
                color: {cls.TEXT_PRIMARY};
            }}

            QGroupBox {{
                background-color: {cls.BG_DARK};
                border: 1px solid {cls.BORDER};
                border-radius: 8px;
                margin-top: 20px;
                padding: 20px 12px 12px 12px;
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 14px;
                top: 6px;
                padding: 2px 10px;
                color: {cls.PRIMARY};
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(10, 10, 10, 255),
                    stop:0.55 rgba(10, 10, 10, 255),
                    stop:0.65 rgba(18, 18, 18, 255),
                    stop:1 rgba(18, 18, 18, 255));
                font-size: 13px;
            }}

            QPushButton {{
                background-color: {cls.BG_LIGHT};
                color: {cls.TEXT_PRIMARY};
                border: 1px solid {cls.BORDER};
                padding: 8px 16px;
                border-radius: 5px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {cls.SECONDARY};
                border-color: {cls.SECONDARY};
            }}
            QPushButton:pressed {{
                background-color: {cls.SECONDARY_DARK};
            }}
            QPushButton:disabled {{
                background-color: #1E1E1E;
                color: #404040;
            }}

            QLineEdit, QSpinBox, QDoubleSpinBox {{
                background-color: {cls.BG_MEDIUM};
                border: 1px solid {cls.BORDER};
                border-radius: 5px;
                padding: 8px 10px;
                color: {cls.TEXT_PRIMARY};
                min-height: 20px;
                selection-background-color: {cls.SECONDARY};
            }}
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
                border-color: {cls.PRIMARY};
            }}

            QComboBox {{
                background-color: {cls.BG_MEDIUM};
                border: 1px solid {cls.BORDER};
                border-radius: 5px;
                padding: 8px 10px;
                color: {cls.TEXT_PRIMARY};
                min-height: 20px;
                selection-background-color: {cls.SECONDARY};
            }}
            QComboBox:focus {{ border-color: {cls.PRIMARY}; }}
            QComboBox::drop-down {{ border: none; padding-right: 12px; }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid {cls.TEXT_SECONDARY};
            }}
            QComboBox QAbstractItemView {{
                background-color: {cls.BG_DARK};
                border: 1px solid {cls.BORDER};
                selection-background-color: {cls.SECONDARY};
                padding: 4px;
                outline: none;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 6px 10px;
                min-height: 24px;
            }}

            QLabel {{
                color: {cls.TEXT_PRIMARY};
                background-color: transparent;
                padding: 0px;
                font-weight: bold;
            }}

            QTableView, QTableWidget {{
                background-color: {cls.BG_DARK};
                gridline-color: {cls.BORDER};
                border: 1px solid {cls.BORDER};
                border-radius: 5px;
            }}
            QTableView::item, QTableWidget::item {{
                padding: 8px;
            }}
            QTableView::item:selected, QTableWidget::item:selected {{
                background-color: {cls.SECONDARY};
            }}
            QHeaderView::section {{
                background-color: {cls.BG_DARK};
                color: {cls.TEXT_PRIMARY};
                padding: 10px 8px;
                border: none;
                border-bottom: 1px solid {cls.BORDER};
                font-weight: bold;
            }}
            QTableCornerButton::section {{
                background-color: {cls.BG_DARK};
                border: none;
                border-bottom: 1px solid {cls.BORDER};
            }}

            QTextEdit, QPlainTextEdit {{
                background-color: {cls.BG_DARK};
                border: 1px solid {cls.BORDER};
                border-radius: 5px;
                padding: 10px;
                color: {cls.TEXT_PRIMARY};
            }}

            /* VS Code-style composer */
            QFrame#composer {{
                background-color: {cls.BG_DARK};
                border: 1px solid {cls.BORDER};
                border-radius: 8px;
            }}
            QFrame#composer QPlainTextEdit {{
                background-color: transparent;
                border: none;
                padding: 4px 6px;
                color: {cls.TEXT_PRIMARY};
                font-size: 12px;
            }}

            /* Model picker pill — compact button with menu */
            QPushButton#modelBtn {{
                background-color: transparent;
                border: 1px solid {cls.BORDER};
                color: {cls.TEXT_SECONDARY};
                padding: 3px 10px;
                border-radius: 4px;
                font-weight: normal;
                font-size: 11px;
                text-align: left;
                min-height: 0;
            }}
            QPushButton#modelBtn:hover {{
                background-color: {cls.BG_LIGHT};
                color: {cls.TEXT_PRIMARY};
            }}
            QPushButton#modelBtn::menu-indicator {{
                image: none;
                width: 0;
            }}

            /* Subtle toolbar buttons (Clear etc.) — visible only on hover */
            QPushButton#composerTool {{
                background-color: transparent;
                border: 1px solid transparent;
                color: {cls.TEXT_SECONDARY};
                padding: 3px 10px;
                border-radius: 4px;
                font-weight: normal;
                font-size: 11px;
                min-height: 0;
            }}
            QPushButton#composerTool:hover {{
                background-color: {cls.BG_LIGHT};
                color: {cls.TEXT_PRIMARY};
            }}
            QPushButton#composerTool:disabled {{
                color: {cls.TEXT_MUTED};
            }}

            /* Primary send action — compact coral */
            QPushButton#sendBtn {{
                background-color: {cls.PRIMARY};
                color: white;
                border: 1px solid {cls.PRIMARY};
                padding: 3px 14px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
                min-height: 0;
            }}
            QPushButton#sendBtn:hover {{
                background-color: {cls.PRIMARY_LIGHT};
                border-color: {cls.PRIMARY_LIGHT};
            }}
            QPushButton#sendBtn:pressed {{
                background-color: {cls.PRIMARY_DARK};
            }}
            QPushButton#sendBtn:disabled {{
                background-color: {cls.BORDER};
                color: {cls.TEXT_MUTED};
                border-color: {cls.BORDER};
            }}

            /* Stop button — replaces Send while streaming */
            QPushButton#stopBtn {{
                background-color: {cls.BG_LIGHT};
                color: {cls.TEXT_PRIMARY};
                border: 1px solid {cls.BORDER};
                padding: 3px 14px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
                min-height: 0;
            }}
            QPushButton#stopBtn:hover {{
                background-color: {cls.PRIMARY};
                border-color: {cls.PRIMARY};
                color: white;
            }}

            QScrollBar:vertical {{
                background-color: {cls.BG_DARK};
                width: 10px;
                border-radius: 5px;
                margin: 2px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {cls.BG_LIGHTER};
                border-radius: 5px;
                min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {cls.SECONDARY};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar:horizontal {{
                background-color: {cls.BG_DARK};
                height: 10px;
                border-radius: 5px;
                margin: 2px;
            }}
            QScrollBar::handle:horizontal {{
                background-color: {cls.BG_LIGHTER};
                border-radius: 5px;
                min-width: 30px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background-color: {cls.SECONDARY};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}

            QCheckBox {{
                spacing: 10px;
                color: {cls.TEXT_PRIMARY};
                padding: 6px 0px;
                min-height: 26px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid {cls.BORDER};
                background-color: {cls.BG_DARK};
            }}
            QCheckBox::indicator:checked {{
                background-color: {cls.PRIMARY};
                border-color: {cls.PRIMARY};
            }}
            QCheckBox::indicator:hover {{
                border-color: {cls.PRIMARY_LIGHT};
            }}

            QToolTip {{
                background-color: {cls.BG_MEDIUM};
                color: {cls.TEXT_PRIMARY};
                border: 1px solid {cls.BORDER};
                padding: 8px;
                border-radius: 4px;
            }}
        """


def apply_dock_theme(widget: QWidget) -> None:
    widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    widget.setStyleSheet(Theme.get_dock_stylesheet())


class NoScrollSpinBox(QSpinBox):
    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()


class NoScrollComboBox(QComboBox):
    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()
