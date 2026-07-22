"""Video studio: play a local video and send it to a video-input model.

OpenRouter has **no video-generation** models (nothing outputs video), so this
view is about video *understanding*: load/play a clip locally and ask a
video-capable model about it.

Playback uses QtMultimedia, which ships with PyQt6 but needs platform media
plugins. If it can't be imported we degrade gracefully: the player is replaced
by a note, and you can still send the video to a model.
"""

from __future__ import annotations

import base64
import mimetypes

from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..catalog import CAT_VIDEO, Model
from ..client import OpenRouterClient
from ..workers import CompletionWorker

try:  # QtMultimedia is optional at runtime.
    from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PyQt6.QtMultimediaWidgets import QVideoWidget

    _MULTIMEDIA_OK = True
    _MULTIMEDIA_ERR = ""
except Exception as exc:  # pragma: no cover - depends on platform libs
    _MULTIMEDIA_OK = False
    _MULTIMEDIA_ERR = str(exc)


class VideoStudioWidget(QWidget):
    def __init__(self, client: OpenRouterClient, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._client = client
        self._models: list[Model] = []
        self._worker: CompletionWorker | None = None
        self._video_path: str | None = None
        self._player = None
        self._audio = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        note = QLabel(
            "OpenRouter has no video-generation models — this tests video "
            "understanding. Load a clip, play it, and ask a video-capable "
            "model about it."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #888;")
        layout.addWidget(note)

        row = QHBoxLayout()
        row.addWidget(QLabel("Video model:"))
        self._model_combo = QComboBox()
        self._model_combo.setMinimumWidth(300)
        row.addWidget(self._model_combo)
        row.addStretch(1)
        self._open_btn = QPushButton("Open video…")
        self._open_btn.clicked.connect(self._on_open)
        row.addWidget(self._open_btn)
        layout.addLayout(row)

        # Player area (or a fallback note).
        if _MULTIMEDIA_OK:
            self._video_widget = QVideoWidget()
            self._video_widget.setMinimumHeight(320)
            self._video_widget.setStyleSheet("background:#000;")
            layout.addWidget(self._video_widget, stretch=1)

            self._player = QMediaPlayer()
            self._audio = QAudioOutput()
            self._player.setAudioOutput(self._audio)
            self._player.setVideoOutput(self._video_widget)

            controls = QHBoxLayout()
            self._play_btn = QPushButton("Play")
            self._play_btn.clicked.connect(self._on_play)
            self._pause_btn = QPushButton("Pause")
            self._pause_btn.clicked.connect(lambda: self._player.pause())
            controls.addWidget(self._play_btn)
            controls.addWidget(self._pause_btn)
            controls.addStretch(1)
            layout.addLayout(controls)
        else:
            fallback = QLabel(
                "Video playback unavailable (QtMultimedia could not load: "
                f"{_MULTIMEDIA_ERR}).\nYou can still open a file and send it to "
                "a model. On Linux, installing GStreamer plugins usually fixes "
                "playback."
            )
            fallback.setWordWrap(True)
            fallback.setStyleSheet("color:#d29922; border:1px solid #333; padding:8px;")
            fallback.setMinimumHeight(200)
            layout.addWidget(fallback, stretch=1)

        self._file_label = QLabel("No video loaded.")
        self._file_label.setStyleSheet("color:#888;")
        layout.addWidget(self._file_label)

        # Prompt + send.
        prompt_row = QHBoxLayout()
        self._prompt = QPlainTextEdit()
        self._prompt.setPlaceholderText("Ask about the video (e.g. 'Describe what happens')…")
        self._prompt.setFixedHeight(70)
        prompt_row.addWidget(self._prompt, stretch=1)
        self._send_btn = QPushButton("Send to model")
        self._send_btn.clicked.connect(self._on_send)
        prompt_row.addWidget(self._send_btn)
        layout.addLayout(prompt_row)

        self._response = QTextEdit()
        self._response.setReadOnly(True)
        self._response.setPlaceholderText("Model response appears here.")
        self._response.setFixedHeight(120)
        layout.addWidget(self._response)

    # ------------------------------------------------------------------ #
    def set_catalog(self, catalog: dict[str, list[Model]]) -> None:
        self._models = list(catalog.get(CAT_VIDEO, []))
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

    # ------------------------------------------------------------------ #
    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open video", "", "Videos (*.mp4 *.mov *.webm *.mkv *.avi)"
        )
        if not path:
            return
        self._video_path = path
        self._file_label.setText(f"Loaded: {path.rsplit('/', 1)[-1]}")
        if _MULTIMEDIA_OK and self._player is not None:
            self._player.setSource(QUrl.fromLocalFile(path))
            self._player.play()

    def _on_play(self) -> None:
        if _MULTIMEDIA_OK and self._player is not None:
            self._player.play()

    def _on_send(self) -> None:
        if self._worker is not None:
            return
        model_id = self._model_combo.currentData()
        if not model_id:
            QMessageBox.information(self, "No model", "No video-capable model selected.")
            return
        if not self._video_path:
            QMessageBox.information(self, "No video", "Open a video file first.")
            return
        if not self._client.api_key:
            QMessageBox.information(
                self, "API key required",
                "Enter your OpenRouter API key (top of the window) first.",
            )
            return
        prompt = self._prompt.toPlainText().strip() or "Describe this video."

        try:
            with open(self._video_path, "rb") as fh:
                encoded = base64.b64encode(fh.read()).decode("ascii")
        except OSError as exc:
            QMessageBox.warning(self, "Read failed", str(exc))
            return
        mime, _ = mimetypes.guess_type(self._video_path)
        mime = mime or "video/mp4"
        data_url = f"data:{mime};base64,{encoded}"

        content = [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": data_url}},
        ]
        messages = [{"role": "user", "content": content}]

        self._response.setPlainText("Sending…")
        self._send_btn.setEnabled(False)
        self._worker = CompletionWorker(
            self._client, model_id, messages, temperature=0.5
        )
        self._worker.finished_ok.connect(self._on_response)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_response(self, message: dict) -> None:
        content = message.get("content")
        text = content if isinstance(content, str) else ""
        self._response.setPlainText(text or "(no text returned)")
        self._cleanup()

    def _on_failed(self, message: str) -> None:
        self._response.setPlainText(f"⚠️ {message}")
        self._cleanup()

    def _cleanup(self) -> None:
        if self._worker is not None:
            self._worker.wait(50)
            self._worker = None
        self._send_btn.setEnabled(True)
