"""Dead-simple image-generation studio: pick model → prompt → big image → save."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..catalog import CAT_IMAGE_GEN, Model
from ..client import OpenRouterClient
from ..workers import ImageGenWorker


class ImageStudioWidget(QWidget):
    """Focused view for the image-generation models only."""

    def __init__(self, client: OpenRouterClient, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._client = client
        self._models: list[Model] = []
        self._worker: ImageGenWorker | None = None
        self._images: list[QImage] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        row = QHBoxLayout()
        row.addWidget(QLabel("Image model:"))
        self._model_combo = QComboBox()
        self._model_combo.setMinimumWidth(320)
        row.addWidget(self._model_combo)
        row.addStretch(1)
        self._save_btn = QPushButton("Save image…")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save)
        row.addWidget(self._save_btn)
        layout.addLayout(row)

        self._hint = QLabel(
            "Enter a prompt and click Generate. The result appears below."
        )
        self._hint.setStyleSheet("color: #888;")
        layout.addWidget(self._hint)

        # Big image area.
        self._image_label = QLabel("No image yet.")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet(
            "background:#111; color:#888; border:1px solid #333;"
        )
        self._image_label.setMinimumHeight(360)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._image_label)
        layout.addWidget(scroll, stretch=1)

        # Any text the model returned alongside (or instead of) an image.
        self._text_out = QLabel("")
        self._text_out.setWordWrap(True)
        self._text_out.setStyleSheet("color: #ccc;")
        layout.addWidget(self._text_out)

        # Prompt + buttons.
        prompt_row = QHBoxLayout()
        self._prompt = QPlainTextEdit()
        self._prompt.setPlaceholderText("Describe the image to generate…")
        self._prompt.setFixedHeight(80)
        prompt_row.addWidget(self._prompt, stretch=1)

        btn_col = QVBoxLayout()
        self._generate_btn = QPushButton("Generate")
        self._generate_btn.clicked.connect(self._on_generate)
        btn_col.addWidget(self._generate_btn)
        self._status = QLabel("")
        self._status.setStyleSheet("color: #888;")
        btn_col.addWidget(self._status)
        btn_col.addStretch(1)
        prompt_row.addLayout(btn_col)
        layout.addLayout(prompt_row)

    # ------------------------------------------------------------------ #
    def set_catalog(self, catalog: dict[str, list[Model]]) -> None:
        self._models = list(catalog.get(CAT_IMAGE_GEN, []))
        current = self._model_combo.currentData()
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        for m in self._models:
            self._model_combo.addItem(m.name, m.id)
        if current is not None:
            idx = self._model_combo.findData(current)
            if idx >= 0:
                self._model_combo.setCurrentIndex(idx)
        self._model_combo.blockSignals(False)
        if not self._models:
            self._status.setText("No image-generation models available.")

    # ------------------------------------------------------------------ #
    def _current_model_id(self) -> str | None:
        return self._model_combo.currentData()

    def _on_generate(self) -> None:
        if self._worker is not None:
            return
        model_id = self._current_model_id()
        if not model_id:
            return
        prompt = self._prompt.toPlainText().strip()
        if not prompt:
            return
        if not self._client.api_key:
            QMessageBox.information(
                self, "API key required",
                "Enter your OpenRouter API key (top of the window) first.",
            )
            return

        self._text_out.setText("")
        self._image_label.setText("Generating…")
        self._status.setText("Generating…")
        self._generate_btn.setEnabled(False)

        messages = [{"role": "user", "content": prompt}]
        self._worker = ImageGenWorker(self._client, model_id, messages)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_done(self, images: list, text: str) -> None:
        self._images = []
        for data in images:
            img = QImage()
            if img.loadFromData(data):
                self._images.append(img)

        if self._images:
            self._show_image(self._images[0])
            self._save_btn.setEnabled(True)
            extra = f"  (+{len(self._images) - 1} more)" if len(self._images) > 1 else ""
            self._status.setText(f"Done: {len(self._images)} image(s).{extra}")
        else:
            self._image_label.setText("No image returned.")
            self._save_btn.setEnabled(False)
            self._status.setText("Model returned no image.")

        if text:
            self._text_out.setText(text)
        elif not self._images:
            self._text_out.setText(
                "The model replied with neither an image nor text. Try another "
                "image model (e.g. google/gemini-2.5-flash-image)."
            )
        self._cleanup()

    def _on_failed(self, message: str) -> None:
        self._image_label.setText("Generation failed.")
        self._text_out.setText(f"⚠️ {message}")
        self._status.setText("Failed.")
        self._save_btn.setEnabled(False)
        self._cleanup()

    def _cleanup(self) -> None:
        if self._worker is not None:
            self._worker.wait(50)
            self._worker = None
        self._generate_btn.setEnabled(True)

    def _show_image(self, img: QImage) -> None:
        pix = QPixmap.fromImage(img)
        self._image_label.setPixmap(
            pix.scaled(
                self._image_label.width() or 640,
                self._image_label.height() or 360,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _on_save(self) -> None:
        if not self._images:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save image", "generated.png", "PNG image (*.png)"
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        if not self._images[0].save(path, "PNG"):
            QMessageBox.warning(self, "Save failed", "Could not write the image.")

    # Re-fit the shown image when the panel resizes.
    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        if self._images:
            self._show_image(self._images[0])
