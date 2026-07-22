"""Entry point for the OpenRouter API Tester desktop app.

Usage:
    export OPENROUTER_API_KEY="sk-or-..."   # optional; can also be typed in UI
    python main.py
"""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from openrouter_tester.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("OpenRouter API Tester")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
