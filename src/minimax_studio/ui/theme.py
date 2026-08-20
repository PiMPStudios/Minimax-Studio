"""Dark studio palette. Fusion + stylesheet, no custom widget kit."""

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1c1c1e;
    color: #e8e8ea;
    font-size: 13px;
}
QListWidget {
    background-color: #161618;
    border: none;
    padding: 8px 4px;
    outline: none;
}
QListWidget::item {
    padding: 8px 12px;
    border-radius: 6px;
    margin: 2px 6px;
}
QListWidget::item:selected {
    background-color: #3a3a3c;
    color: #ffffff;
}
QDockWidget {
    titlebar-close-icon: none;
}
QDockWidget::title {
    background: #161618;
    padding: 8px;
    border-bottom: 1px solid #2c2c2e;
}
QGroupBox {
    border: 1px solid #2c2c2e;
    border-radius: 8px;
    margin-top: 12px;
    padding: 12px 8px 8px 8px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {
    background-color: #2c2c2e;
    border: 1px solid #3a3a3c;
    border-radius: 6px;
    padding: 4px 8px;
    selection-background-color: #0a84ff;
}
QPushButton {
    background-color: #2c2c2e;
    border: 1px solid #3a3a3c;
    border-radius: 6px;
    padding: 6px 14px;
}
QPushButton:hover {
    background-color: #3a3a3c;
}
QPushButton#primary {
    background-color: #0a84ff;
    border: none;
    color: white;
    font-weight: 600;
}
QPushButton#primary:hover {
    background-color: #409cff;
}
QStatusBar {
    background: #161618;
    border-top: 1px solid #2c2c2e;
}
QLabel#pageTitle {
    font-size: 22px;
    font-weight: 700;
}
QLabel#pageSubtitle {
    color: #8e8e93;
}
QLabel#brand {
    color: #8e8e93;
    font-size: 11px;
}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    bg = QColor("#1c1c1e")
    text = QColor("#e8e8ea")
    palette.setColor(QPalette.ColorRole.Window, bg)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, QColor("#161618"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#2c2c2e"))
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.Button, QColor("#2c2c2e"))
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#0a84ff"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#2c2c2e"))
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)
