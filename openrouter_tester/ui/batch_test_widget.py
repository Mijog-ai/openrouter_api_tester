"""Batch test tab: probe many models with one prompt and tabulate results."""

from __future__ import annotations

from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..catalog import Model, flatten
from ..client import OpenRouterClient
from ..workers import BatchTestWorker

_ALL = "All categories"
_GREEN = QColor("#1f6f3d")
_RED = QColor("#7a2620")


class BatchTestWidget(QWidget):
    """Run a small prompt against every model in a category and show results."""

    def __init__(self, client: OpenRouterClient, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._client = client
        self._catalog: dict[str, list[Model]] = {}
        self._current: list[Model] = []
        self._worker: BatchTestWorker | None = None
        self._build_ui()

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Category:"))
        self._category = QComboBox()
        self._category.addItem(_ALL)
        self._category.currentTextChanged.connect(self._update_count_hint)
        controls.addWidget(self._category)

        controls.addWidget(QLabel("Max models:"))
        self._limit = QSpinBox()
        self._limit.setRange(1, 1000)
        self._limit.setValue(20)
        self._limit.setToolTip("Cap how many models to probe (protects your credits).")
        controls.addWidget(self._limit)

        controls.addWidget(QLabel("Prompt:"))
        self._prompt = QLineEdit("Reply with the single word: pong")
        controls.addWidget(self._prompt, stretch=1)

        self._run_btn = QPushButton("Run test")
        self._run_btn.clicked.connect(self._on_run)
        controls.addWidget(self._run_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        controls.addWidget(self._stop_btn)
        layout.addLayout(controls)

        self._progress = QProgressBar()
        self._progress.setTextVisible(True)
        layout.addWidget(self._progress)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["Model", "Category", "Status", "Latency", "Detail"]
        )
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._table, stretch=1)

        self._summary = QLabel("Load models, choose a category, then Run test.")
        layout.addWidget(self._summary)

    # ------------------------------------------------------------------ #
    def set_catalog(self, catalog: dict[str, list[Model]]) -> None:
        self._catalog = catalog
        current = self._category.currentText()
        self._category.blockSignals(True)
        self._category.clear()
        self._category.addItem(_ALL)
        for category in catalog:
            self._category.addItem(category)
        idx = self._category.findText(current)
        self._category.setCurrentIndex(idx if idx >= 0 else 0)
        self._category.blockSignals(False)
        self._update_count_hint()

    def _selected_models(self) -> list[Model]:
        category = self._category.currentText()
        if category == _ALL:
            # Distinct models across all categories (no double-testing).
            return flatten(self._catalog)
        return list(self._catalog.get(category, []))

    def _update_count_hint(self, *_args) -> None:
        available = len(self._selected_models())
        self._limit.setMaximum(max(1, available))
        self._summary.setText(
            f"{available} models available in this selection. "
            f"'Run test' probes up to 'Max models'."
        )

    # ------------------------------------------------------------------ #
    def _on_run(self) -> None:
        if self._worker is not None:
            return
        if not self._client.api_key:
            QMessageBox.information(
                self, "API key required",
                "Enter your OpenRouter API key (top of the window) before testing.",
            )
            return
        models = self._selected_models()[: self._limit.value()]
        if not models:
            return
        prompt = self._prompt.text().strip() or "Reply with the single word: pong"

        self._current = models
        self._table.setRowCount(len(models))
        for row, model in enumerate(models):
            self._set_cell(row, 0, model.name)
            self._set_cell(row, 1, model.primary_category)
            self._set_cell(row, 2, "queued")
            self._set_cell(row, 3, "")
            self._set_cell(row, 4, "")
        self._progress.setMaximum(len(models))
        self._progress.setValue(0)

        self._worker = BatchTestWorker(self._client, models, prompt)
        self._worker.result.connect(self._on_result)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_all.connect(self._on_finished)
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._category.setEnabled(False)
        self._worker.start()

    def _on_stop(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()
            self._stop_btn.setEnabled(False)

    def _on_result(
        self, index: int, model_id: str, ok: bool, latency: float, detail: str
    ) -> None:
        status = "✓ pass" if ok else "✗ fail"
        self._set_cell(index, 2, status)
        self._set_cell(index, 3, f"{latency:.2f}s")
        self._set_cell(index, 4, detail)
        color = _GREEN if ok else _RED
        for col in range(5):
            item = self._table.item(index, col)
            if item is not None:
                item.setBackground(QBrush(color))

    def _on_progress(self, done: int, total: int) -> None:
        self._progress.setValue(done)
        self._progress.setFormat(f"{done}/{total}")

    def _on_finished(self) -> None:
        passed = sum(
            1
            for row in range(self._table.rowCount())
            if (it := self._table.item(row, 2)) and it.text().startswith("✓")
        )
        tested = self._progress.value()
        self._summary.setText(
            f"Done. {passed}/{tested} passed"
            + (" (stopped early)" if tested < len(self._current) else "")
            + "."
        )
        if self._worker is not None:
            self._worker.wait(50)
            self._worker = None
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._category.setEnabled(True)

    # ------------------------------------------------------------------ #
    def _set_cell(self, row: int, col: int, text: str) -> None:
        item = QTableWidgetItem(text)
        item.setToolTip(text)
        self._table.setItem(row, col, item)
