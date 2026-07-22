"""Main application window: model browser (left) + chat playground (right)."""

from __future__ import annotations

import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..catalog import CATEGORY_DESCRIPTIONS, Model, build_catalog
from ..client import OpenRouterClient
from ..workers import ModelFetchWorker
from .chat_widget import ChatWidget

_MODEL_ROLE = Qt.ItemDataRole.UserRole


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OpenRouter API Tester")
        self.resize(1150, 720)

        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self._client = OpenRouterClient(api_key=api_key)
        self._catalog: dict[str, list[Model]] = {}
        self._fetch_worker: ModelFetchWorker | None = None

        self._build_ui()
        if api_key:
            self._key_input.setText(api_key)
        # Kick off an initial load (models endpoint works without a key).
        self.refresh_models()

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)

        # --- top bar: API key + refresh ------------------------------- #
        top = QHBoxLayout()
        top.addWidget(QLabel("OpenRouter API key:"))
        self._key_input = QLineEdit()
        self._key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._key_input.setPlaceholderText("sk-or-…  (or set OPENROUTER_API_KEY)")
        self._key_input.textChanged.connect(self._on_key_changed)
        top.addWidget(self._key_input, stretch=1)

        self._refresh_btn = QPushButton("Refresh models")
        self._refresh_btn.clicked.connect(self.refresh_models)
        top.addWidget(self._refresh_btn)
        outer.addLayout(top)

        # --- main splitter -------------------------------------------- #
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: search + tree.
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        filter_row = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText("Filter models…")
        self._search.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self._search, stretch=1)
        self._free_only = QCheckBox("Free only")
        self._free_only.stateChanged.connect(self._apply_filter)
        filter_row.addWidget(self._free_only)
        left_layout.addLayout(filter_row)

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self._tree, stretch=1)

        splitter.addWidget(left)

        # Right: chat playground.
        self._chat = ChatWidget(self._client)
        splitter.addWidget(self._chat)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 810])
        outer.addWidget(splitter, stretch=1)

        self.statusBar().showMessage("Ready.")

    # ------------------------------------------------------------------ #
    # API key handling
    # ------------------------------------------------------------------ #
    def _on_key_changed(self, text: str) -> None:
        self._client.api_key = text.strip()

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #
    def refresh_models(self) -> None:
        if self._fetch_worker is not None:
            return
        self._refresh_btn.setEnabled(False)
        self.statusBar().showMessage("Loading models from OpenRouter…")
        self._fetch_worker = ModelFetchWorker(self._client)
        self._fetch_worker.finished_ok.connect(self._on_models_loaded)
        self._fetch_worker.failed.connect(self._on_models_failed)
        self._fetch_worker.start()

    def _on_models_loaded(self, raw_models: list) -> None:
        self._catalog = build_catalog(raw_models)
        total = sum(len(v) for v in self._catalog.values())
        self._populate_tree()
        self.statusBar().showMessage(
            f"Loaded {total} models across {len(self._catalog)} categories "
            f"(audio models excluded)."
        )
        self._cleanup_fetch_worker()

    def _on_models_failed(self, message: str) -> None:
        self.statusBar().showMessage(f"Failed to load models: {message}")
        self._cleanup_fetch_worker()

    def _cleanup_fetch_worker(self) -> None:
        if self._fetch_worker is not None:
            self._fetch_worker.wait(50)
            self._fetch_worker = None
        self._refresh_btn.setEnabled(True)

    # ------------------------------------------------------------------ #
    # Tree population + filtering
    # ------------------------------------------------------------------ #
    def _populate_tree(self) -> None:
        self._tree.clear()
        for category, models in self._catalog.items():
            cat_item = QTreeWidgetItem([f"{category}  ({len(models)})"])
            cat_item.setToolTip(0, CATEGORY_DESCRIPTIONS.get(category, ""))
            cat_item.setFlags(cat_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            font = cat_item.font(0)
            font.setBold(True)
            cat_item.setFont(0, font)
            for model in models:
                child = QTreeWidgetItem([model.name])
                child.setData(0, _MODEL_ROLE, model)
                child.setToolTip(0, self._model_tooltip(model))
                cat_item.addChild(child)
            self._tree.addTopLevelItem(cat_item)
            cat_item.setExpanded(True)
        self._apply_filter()

    @staticmethod
    def _model_tooltip(model: Model) -> str:
        ctx = f"{model.context_length:,}" if model.context_length else "unknown"
        desc = model.description[:280] + ("…" if len(model.description) > 280 else "")
        return (
            f"{model.id}\n"
            f"Context: {ctx} tokens\n"
            f"Pricing: {model.price_summary()}\n\n"
            f"{desc}"
        )

    def _apply_filter(self, *_args) -> None:
        query = self._search.text().strip().lower()
        free_only = self._free_only.isChecked()
        for i in range(self._tree.topLevelItemCount()):
            cat_item = self._tree.topLevelItem(i)
            visible_children = 0
            for j in range(cat_item.childCount()):
                child = cat_item.child(j)
                model: Model = child.data(0, _MODEL_ROLE)
                matches = (
                    not query
                    or query in model.name.lower()
                    or query in model.id.lower()
                )
                if free_only and not model.is_free:
                    matches = False
                child.setHidden(not matches)
                if matches:
                    visible_children += 1
            cat_item.setHidden(visible_children == 0)

    def _on_selection_changed(self) -> None:
        items = self._tree.selectedItems()
        if not items:
            return
        model: Model | None = items[0].data(0, _MODEL_ROLE)
        if model is None:
            return
        self._chat.set_model(model)
        self.statusBar().showMessage(
            f"Selected {model.name}  ·  {model.price_summary()}  ·  "
            f"{model.context_length:,} ctx"
        )

    # ------------------------------------------------------------------ #
    # Qt lifecycle
    # ------------------------------------------------------------------ #
    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        # Make sure background threads are stopped before we exit.
        if self._fetch_worker is not None:
            self._fetch_worker.wait(200)
        super().closeEvent(event)
