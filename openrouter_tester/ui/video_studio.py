"""Video generation studio.

OpenRouter generates video through async jobs on the dedicated
``/api/v1/videos`` endpoint (separate from the chat ``/models`` API). This tab
lists the real video-generation models, submits a text-to-video (or
image-to-video) job, polls it to completion, then plays the resulting mp4
in-window and lets you save it.

Playback uses QtMultimedia, which ships with PyQt6 but needs platform media
plugins. If it can't load we degrade gracefully: no in-window player, but you
can still generate and save the mp4.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import tempfile

from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..catalog import VideoModel, build_video_models
from ..client import OpenRouterClient
from ..workers import VideoGenWorker, VideoModelFetchWorker

try:  # QtMultimedia is optional at runtime.
    from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PyQt6.QtMultimediaWidgets import QVideoWidget

    _MM_OK = True
    _MM_ERR = ""
except Exception as exc:  # pragma: no cover - depends on platform libs
    _MM_OK = False
    _MM_ERR = str(exc)

_ANY = "(default)"


class VideoStudioWidget(QWidget):
    def __init__(self, client: OpenRouterClient, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._client = client
        self._models: list[VideoModel] = []
        self._fetch_worker: VideoModelFetchWorker | None = None
        self._gen_worker: VideoGenWorker | None = None
        self._player = None
        self._audio = None
        self._video_widget = None
        self._mp4_bytes: bytes | None = None
        self._tmp_path: str | None = None
        self._first_frame_data_url: str | None = None
        self._last_body: dict | None = None
        self._build_ui()

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Model + options row.
        row = QHBoxLayout()
        row.addWidget(QLabel("Video model:"))
        self._model_combo = QComboBox()
        self._model_combo.setMinimumWidth(240)
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        row.addWidget(self._model_combo)

        row.addWidget(QLabel("Duration"))
        self._duration = QComboBox()
        row.addWidget(self._duration)
        row.addWidget(QLabel("Resolution"))
        self._resolution = QComboBox()
        row.addWidget(self._resolution)
        row.addWidget(QLabel("Aspect"))
        self._aspect = QComboBox()
        row.addWidget(self._aspect)
        self._audio_check = QCheckBox("Audio")
        row.addWidget(self._audio_check)
        row.addStretch(1)
        layout.addLayout(row)

        # First-frame (image-to-video) row.
        frame_row = QHBoxLayout()
        self._frame_btn = QPushButton("First frame image…")
        self._frame_btn.clicked.connect(self._on_pick_frame)
        frame_row.addWidget(self._frame_btn)
        self._frame_label = QLabel("")
        self._frame_label.setStyleSheet("color:#888;")
        frame_row.addWidget(self._frame_label, stretch=1)
        layout.addLayout(frame_row)

        # Player area (or fallback note).
        if _MM_OK:
            self._video_widget = QVideoWidget()
            self._video_widget.setMinimumHeight(320)
            self._video_widget.setStyleSheet("background:#000;")
            layout.addWidget(self._video_widget, stretch=1)
            self._player = QMediaPlayer()
            self._audio = QAudioOutput()
            self._player.setAudioOutput(self._audio)
            self._player.setVideoOutput(self._video_widget)

            controls = QHBoxLayout()
            play = QPushButton("Play")
            play.clicked.connect(lambda: self._player and self._player.play())
            pause = QPushButton("Pause")
            pause.clicked.connect(lambda: self._player and self._player.pause())
            controls.addWidget(play)
            controls.addWidget(pause)
            controls.addStretch(1)
            self._save_btn = QPushButton("Save video…")
            self._save_btn.setEnabled(False)
            self._save_btn.clicked.connect(self._on_save)
            controls.addWidget(self._save_btn)
            layout.addLayout(controls)
        else:
            note = QLabel(
                "In-window playback unavailable (QtMultimedia could not load: "
                f"{_MM_ERR}).\nYou can still generate and save the mp4."
            )
            note.setWordWrap(True)
            note.setStyleSheet("color:#d29922; border:1px solid #333; padding:8px;")
            note.setMinimumHeight(180)
            layout.addWidget(note, stretch=1)
            self._save_btn = QPushButton("Save video…")
            self._save_btn.setEnabled(False)
            self._save_btn.clicked.connect(self._on_save)
            layout.addWidget(self._save_btn)

        # Prompt + generate.
        prompt_row = QHBoxLayout()
        self._prompt = QPlainTextEdit()
        self._prompt.setPlaceholderText("Describe the video to generate…")
        self._prompt.setFixedHeight(70)
        prompt_row.addWidget(self._prompt, stretch=1)

        btn_col = QVBoxLayout()
        self._generate_btn = QPushButton("Generate")
        self._generate_btn.clicked.connect(self._on_generate)
        btn_col.addWidget(self._generate_btn)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_col.addWidget(self._cancel_btn)
        prompt_row.addLayout(btn_col)
        layout.addLayout(prompt_row)

        self._status = QLabel("Loading video models…")
        self._status.setStyleSheet("color:#888;")
        layout.addWidget(self._status)

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #
    def refresh_models(self) -> None:
        if self._fetch_worker is not None:
            return
        self._status.setText("Loading video models…")
        self._fetch_worker = VideoModelFetchWorker(self._client)
        self._fetch_worker.finished_ok.connect(self._on_models)
        self._fetch_worker.failed.connect(self._on_models_failed)
        self._fetch_worker.start()

    def _on_models(self, raw: list) -> None:
        self._models = build_video_models(raw)
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        for m in self._models:
            self._model_combo.addItem(m.name, m.id)
        self._model_combo.blockSignals(False)
        self._on_model_changed()
        self._status.setText(f"{len(self._models)} video-generation models available.")
        if self._fetch_worker is not None:
            self._fetch_worker.wait(50)
            self._fetch_worker = None

    def _on_models_failed(self, message: str) -> None:
        self._status.setText(f"Failed to load video models: {message}")
        if self._fetch_worker is not None:
            self._fetch_worker.wait(50)
            self._fetch_worker = None

    def _current_model(self) -> VideoModel | None:
        idx = self._model_combo.currentIndex()
        if 0 <= idx < len(self._models):
            return self._models[idx]
        return None

    def _on_model_changed(self, *_args) -> None:
        m = self._current_model()
        self._duration.clear()
        self._resolution.clear()
        self._aspect.clear()
        self._duration.addItem(_ANY, None)
        self._resolution.addItem(_ANY, None)
        self._aspect.addItem(_ANY, None)
        if m is None:
            return
        for d in m.supported_durations:
            self._duration.addItem(f"{d}s", d)
        for r in m.supported_resolutions:
            self._resolution.addItem(r, r)
        for a in m.supported_aspect_ratios:
            self._aspect.addItem(a, a)
        # Preselect the first real supported value (index 1) rather than the
        # "(default)" placeholder, so a valid value is always sent — several
        # models reject a job that omits duration/resolution/aspect.
        if self._duration.count() > 1:
            self._duration.setCurrentIndex(1)
        if self._resolution.count() > 1:
            self._resolution.setCurrentIndex(1)
        if self._aspect.count() > 1:
            self._aspect.setCurrentIndex(1)
        self._audio_check.setChecked(m.generate_audio)
        self._audio_check.setEnabled(m.generate_audio)
        self._frame_btn.setVisible(m.supports_image_to_video)
        if not m.supports_image_to_video:
            self._first_frame_data_url = None
            self._frame_label.setText("")

    # ------------------------------------------------------------------ #
    def _on_pick_frame(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "First frame image", "", "Images (*.png *.jpg *.jpeg *.webp)"
        )
        if not path:
            return
        mime, _ = mimetypes.guess_type(path)
        mime = mime or "image/png"
        try:
            with open(path, "rb") as fh:
                enc = base64.b64encode(fh.read()).decode("ascii")
        except OSError as exc:
            QMessageBox.warning(self, "Read failed", str(exc))
            return
        self._first_frame_data_url = f"data:{mime};base64,{enc}"
        self._frame_label.setText(f"First frame: {path.rsplit('/', 1)[-1]}")

    # ------------------------------------------------------------------ #
    def _on_generate(self) -> None:
        if self._gen_worker is not None:
            return
        model = self._current_model()
        if model is None:
            return
        prompt = self._prompt.toPlainText().strip()
        if not prompt:
            QMessageBox.information(self, "No prompt", "Enter a prompt first.")
            return
        if not self._client.api_key:
            QMessageBox.information(
                self, "API key required",
                "Enter your OpenRouter API key (top of the window) first.",
            )
            return

        duration = self._duration.currentData()
        resolution = self._resolution.currentData()
        aspect = self._aspect.currentData()
        audio = self._audio_check.isChecked() if self._audio_check.isEnabled() else None

        frame_images = None
        if self._first_frame_data_url and model.supports_image_to_video:
            # Per the API schema each frame image needs all three fields:
            # type == "image_url", the image_url object, and frame_type.
            frame_images = [
                {
                    "type": "image_url",
                    "frame_type": "first_frame",
                    "image_url": {"url": self._first_frame_data_url},
                }
            ]

        def make_body(**extra) -> dict:
            b: dict = {"model": model.id, "prompt": prompt}
            if audio is not None:
                b["generate_audio"] = audio
            if frame_images is not None:
                b["frame_images"] = frame_images
            b.update({k: v for k, v in extra.items() if v is not None})
            return b

        # Fallback ladder: try resolution+aspect first (documented shape), then
        # `size` (which some providers like Wan expect), then minimal.
        bodies = [make_body(duration=duration, resolution=resolution, aspect_ratio=aspect)]
        size = self._best_size(model, resolution, aspect)
        if size:
            bodies.append(make_body(duration=duration, size=size))
        bodies.append(make_body(duration=duration))
        bodies.append(make_body())  # last resort: model + prompt only

        # De-duplicate while preserving order.
        seen, ladder = [], []
        for b in bodies:
            key = tuple(sorted((k, str(v)) for k, v in b.items() if k != "frame_images"))
            if key not in seen:
                seen.append(key)
                ladder.append(b)

        self._last_body = ladder[0]
        self._set_busy(True)
        self._status.setText("Submitting job…")
        self._gen_worker = VideoGenWorker(self._client, ladder)
        self._gen_worker.status.connect(self._status.setText)
        self._gen_worker.finished_ok.connect(self._on_generated)
        self._gen_worker.failed.connect(self._on_failed)
        self._gen_worker.start()

    @staticmethod
    def _best_size(model: VideoModel, resolution, aspect) -> str | None:
        """Pick a supported ``WxH`` size matching resolution+aspect if possible."""
        sizes = model.raw.get("supported_sizes") or []
        if not sizes:
            return None
        # Height implied by the resolution label (e.g. 720p -> 720).
        want_h = None
        if isinstance(resolution, str) and resolution.endswith("p"):
            try:
                want_h = int(resolution[:-1])
            except ValueError:
                want_h = None
        # Aspect landscape vs portrait.
        want_landscape = None
        if isinstance(aspect, str) and ":" in aspect:
            try:
                w, h = (int(x) for x in aspect.split(":", 1))
                want_landscape = w >= h
            except ValueError:
                want_landscape = None

        def parse(s):
            try:
                w, h = (int(x) for x in s.lower().split("x", 1))
                return w, h
            except ValueError:
                return None

        best = None
        for s in sizes:
            wh = parse(s)
            if not wh:
                continue
            w, h = wh
            if want_h is not None and min(w, h) != want_h:
                continue
            if want_landscape is not None and (w >= h) != want_landscape:
                continue
            best = s
            break
        return best or (sizes[0] if sizes else None)

    def _on_cancel(self) -> None:
        if self._gen_worker is not None:
            self._gen_worker.request_stop()
            self._status.setText("Cancelling…")

    def _on_generated(self, data: bytes) -> None:
        self._mp4_bytes = data
        self._status.setText(f"Done: received {len(data):,} bytes of video.")
        self._save_btn.setEnabled(True)
        # Write to a temp file and play it.
        try:
            fd, path = tempfile.mkstemp(suffix=".mp4")
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            self._tmp_path = path
        except OSError as exc:
            self._status.setText(f"Saved-to-temp failed: {exc}")
            self._cleanup_gen()
            return
        if _MM_OK and self._player is not None:
            self._player.setSource(QUrl.fromLocalFile(self._tmp_path))
            self._player.play()
        self._cleanup_gen()

    def _on_failed(self, message: str) -> None:
        self._status.setText("⚠️ Generation failed — see dialog for details.")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Video generation failed")
        box.setText("The video job failed. Full server response:")
        import json

        detail = message
        if self._last_body is not None:
            # Redact the (huge) first-frame data URL for readability.
            shown = dict(self._last_body)
            if "frame_images" in shown:
                shown["frame_images"] = "[first-frame image attached]"
            detail += "\n\nRequest body sent:\n" + json.dumps(shown, indent=2)
        box.setDetailedText(detail)  # expandable + selectable/copyable
        box.exec()
        self._cleanup_gen()

    def _cleanup_gen(self) -> None:
        if self._gen_worker is not None:
            self._gen_worker.wait(50)
            self._gen_worker = None
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        self._generate_btn.setEnabled(not busy)
        self._cancel_btn.setEnabled(busy)

    def _on_save(self) -> None:
        if not self._mp4_bytes:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save video", "generated.mp4", "MP4 video (*.mp4)"
        )
        if not path:
            return
        if not path.lower().endswith(".mp4"):
            path += ".mp4"
        try:
            with open(path, "wb") as fh:
                fh.write(self._mp4_bytes)
        except OSError as exc:
            QMessageBox.warning(self, "Save failed", str(exc))

    # Video models come from /videos/models via refresh_models(); the chat
    # catalog is irrelevant here, so set_catalog is intentionally a no-op.
    def set_catalog(self, _catalog) -> None:
        return
