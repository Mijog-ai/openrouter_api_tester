"""The adaptive chat playground panel.

The panel reshapes itself around the selected model:

* **Text** models        -> plain streaming chat.
* **Vision/Multimodal**  -> chat + "Attach image" (sent inline as a data URL).
* **Document/File**      -> chat + "Attach file" (PDF etc., sent as a file part).
* **Image Generation**   -> a non-streaming request asking for image output;
                            returned images are rendered inline in the transcript.
"""

from __future__ import annotations

import base64
import mimetypes
from typing import Any

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..catalog import Model
from ..client import OpenRouterClient
from ..workers import ChatWorker, CompletionWorker


class TranscriptView(QTextEdit):
    """Read-only rich-text transcript that can render inline images.

    Qt's rich text engine won't decode ``data:`` image URLs directly, so we
    register decoded images under short resource keys and serve them via an
    overridden :meth:`loadResource`.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self._images: dict[str, QImage] = {}

    def register_image(self, key: str, image: QImage) -> None:
        self._images[key] = image

    def clear_images(self) -> None:
        self._images.clear()

    def loadResource(self, resource_type: int, url: QUrl):  # noqa: N802
        key = url.toString()
        if key in self._images:
            return self._images[key]
        return super().loadResource(resource_type, url)


class ChatWidget(QWidget):
    def __init__(self, client: OpenRouterClient, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._client = client
        self._model: Model | None = None
        # Internal transcript: each turn is a dict with keys:
        #   role: "user"|"assistant"
        #   text: str
        #   parts: list[dict]   (extra API content parts for user turns)
        #   img_keys: list[str] (resource keys for inline display)
        self._turns: list[dict[str, Any]] = []
        self._pending_parts: list[dict[str, Any]] = []  # attachments for next msg
        self._pending_labels: list[str] = []
        self._pending_img_keys: list[str] = []
        self._worker: ChatWorker | CompletionWorker | None = None
        self._assistant_buffer = ""
        self._img_counter = 0

        self._build_ui()
        self._set_model(None)

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QHBoxLayout()
        self._model_label = QLabel("No model selected")
        self._model_label.setStyleSheet("font-weight: 600;")
        self._model_label.setWordWrap(True)
        header.addWidget(self._model_label, stretch=1)

        header.addWidget(QLabel("Temp"))
        self._temp_spin = QDoubleSpinBox()
        self._temp_spin.setRange(0.0, 2.0)
        self._temp_spin.setSingleStep(0.1)
        self._temp_spin.setValue(0.7)
        self._temp_spin.setFixedWidth(70)
        header.addWidget(self._temp_spin)

        header.addWidget(QLabel("Max tokens"))
        self._max_tokens_spin = QSpinBox()
        self._max_tokens_spin.setRange(0, 200_000)
        self._max_tokens_spin.setSingleStep(128)
        self._max_tokens_spin.setValue(1024)
        self._max_tokens_spin.setSpecialValueText("auto")
        self._max_tokens_spin.setFixedWidth(90)
        header.addWidget(self._max_tokens_spin)
        layout.addLayout(header)

        self._transcript = TranscriptView()
        self._transcript.setPlaceholderText(
            "Select a model on the left, then start chatting to test it."
        )
        layout.addWidget(self._transcript, stretch=1)

        self._attach_label = QLabel("")
        self._attach_label.setStyleSheet("color: #888;")
        self._attach_label.setWordWrap(True)
        layout.addWidget(self._attach_label)

        input_row = QHBoxLayout()
        self._input = QPlainTextEdit()
        self._input.setPlaceholderText("Type a message…  (Ctrl+Enter to send)")
        self._input.setFixedHeight(80)
        input_row.addWidget(self._input, stretch=1)

        btn_col = QVBoxLayout()
        self._attach_img_btn = QPushButton("Attach image")
        self._attach_img_btn.clicked.connect(self._on_attach_image)
        btn_col.addWidget(self._attach_img_btn)

        self._attach_file_btn = QPushButton("Attach file")
        self._attach_file_btn.clicked.connect(self._on_attach_file)
        btn_col.addWidget(self._attach_file_btn)

        self._send_btn = QPushButton("Send")
        self._send_btn.setDefault(True)
        self._send_btn.clicked.connect(self._on_send)
        btn_col.addWidget(self._send_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.clicked.connect(self._on_stop)
        self._stop_btn.setEnabled(False)
        btn_col.addWidget(self._stop_btn)

        self._clear_btn = QPushButton("Clear chat")
        self._clear_btn.clicked.connect(self.clear_chat)
        btn_col.addWidget(self._clear_btn)

        input_row.addLayout(btn_col)
        layout.addLayout(input_row)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def set_model(self, model: Model) -> None:
        self._set_model(model)

    def clear_chat(self) -> None:
        self._turns.clear()
        self._assistant_buffer = ""
        self._transcript.clear()
        self._transcript.clear_images()
        self._clear_attachments()

    # ------------------------------------------------------------------ #
    # Model selection / adaptive UI
    # ------------------------------------------------------------------ #
    def _set_model(self, model: Model | None) -> None:
        self._model = model
        if model is None:
            self._model_label.setText("No model selected")
            self._send_btn.setEnabled(False)
            self._attach_img_btn.setVisible(False)
            self._attach_file_btn.setVisible(False)
            return

        caps = []
        if model.supports_images:
            caps.append("image input")
        if model.supports_video:
            caps.append("video input")
        if model.supports_files:
            caps.append("file input")
        if model.generates_images:
            caps.append("image output")
        cap_text = f"  ·  {', '.join(caps)}" if caps else ""
        self._model_label.setText(f"{model.name}\n{model.id}{cap_text}")

        self._send_btn.setEnabled(True)
        # Adapt the input controls to the model's capabilities.
        self._attach_img_btn.setVisible(model.supports_images)
        self._attach_file_btn.setVisible(model.supports_files)
        if model.generates_images:
            self._send_btn.setText("Generate")
            self._input.setPlaceholderText(
                "Describe the image to generate…  (Ctrl+Enter to send)"
            )
        else:
            self._send_btn.setText("Send")
            self._input.setPlaceholderText("Type a message…  (Ctrl+Enter to send)")

        # Drop attachments the new model can't use.
        if not (model.supports_images or model.supports_files):
            self._clear_attachments()

    # ------------------------------------------------------------------ #
    # Attachments
    # ------------------------------------------------------------------ #
    def _on_attach_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select an image", "",
            "Images (*.png *.jpg *.jpeg *.webp *.gif)",
        )
        if not path:
            return
        data_url, name = self._encode_file(path, "image/png")
        if data_url is None:
            return
        self._pending_parts.append(
            {"type": "image_url", "image_url": {"url": data_url}}
        )
        # Also decode for inline display of the user's attachment.
        key = self._store_data_url_image(data_url)
        if key:
            self._pending_img_keys.append(key)
        self._pending_labels.append(f"🖼 {name}")
        self._refresh_attach_label()

    def _on_attach_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a file", "",
            "Documents (*.pdf *.txt *.md *.csv *.json);;All files (*)",
        )
        if not path:
            return
        data_url, name = self._encode_file(path, "application/octet-stream")
        if data_url is None:
            return
        self._pending_parts.append(
            {"type": "file", "file": {"filename": name, "file_data": data_url}}
        )
        self._pending_labels.append(f"📄 {name}")
        self._refresh_attach_label()

    def _encode_file(self, path: str, default_mime: str) -> tuple[str | None, str]:
        mime, _ = mimetypes.guess_type(path)
        mime = mime or default_mime
        name = path.rsplit("/", 1)[-1]
        try:
            with open(path, "rb") as fh:
                encoded = base64.b64encode(fh.read()).decode("ascii")
        except OSError as exc:
            QMessageBox.warning(self, "Attach failed", str(exc))
            return None, name
        return f"data:{mime};base64,{encoded}", name

    def _refresh_attach_label(self) -> None:
        if self._pending_labels:
            self._attach_label.setText(
                "Attached (sent with next message): "
                + ",  ".join(self._pending_labels)
            )
        else:
            self._attach_label.setText("")

    def _clear_attachments(self) -> None:
        self._pending_parts.clear()
        self._pending_labels.clear()
        self._pending_img_keys.clear()
        self._refresh_attach_label()

    # ------------------------------------------------------------------ #
    # Sending
    # ------------------------------------------------------------------ #
    def _on_send(self) -> None:
        if self._model is None or self._worker is not None:
            return
        text = self._input.toPlainText().strip()
        if not text and not self._pending_parts:
            return
        if not self._client.api_key:
            QMessageBox.information(
                self, "API key required",
                "Enter your OpenRouter API key (top of the window) before chatting.",
            )
            return

        self._turns.append(
            {
                "role": "user",
                "text": text,
                "parts": list(self._pending_parts),
                "img_keys": list(self._pending_img_keys),
            }
        )
        self._clear_attachments()
        self._input.clear()
        self._assistant_buffer = ""
        self._render_all()

        messages = self._build_api_messages()
        max_tokens = self._max_tokens_spin.value() or None

        if self._model.generates_images:
            self._start_image_generation(messages, max_tokens)
        else:
            self._start_streaming_chat(messages, max_tokens)

    def _start_streaming_chat(self, messages, max_tokens) -> None:
        worker = ChatWorker(
            self._client, self._model.id, messages,
            temperature=self._temp_spin.value(), max_tokens=max_tokens,
        )
        worker.chunk.connect(self._on_chunk)
        worker.finished_ok.connect(self._on_stream_finished)
        worker.failed.connect(self._on_failed)
        self._worker = worker
        self._set_busy(True)
        worker.start()

    def _start_image_generation(self, messages, max_tokens) -> None:
        self._assistant_buffer = "Generating image…"
        self._render_all()
        worker = CompletionWorker(
            self._client, self._model.id, messages,
            temperature=self._temp_spin.value(), max_tokens=max_tokens,
            modalities=["image", "text"],
        )
        worker.finished_ok.connect(self._on_image_message)
        worker.failed.connect(self._on_failed)
        self._worker = worker
        self._set_busy(True)
        worker.start()

    def _build_api_messages(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for turn in self._turns:
            if turn["role"] == "user":
                if turn["parts"]:
                    content: Any = [{"type": "text", "text": turn["text"]}]
                    content.extend(turn["parts"])
                else:
                    content = turn["text"]
                messages.append({"role": "user", "content": content})
            else:
                # Assistant turns are replayed as text only.
                messages.append({"role": "assistant", "content": turn["text"]})
        return messages

    # ------------------------------------------------------------------ #
    # Worker callbacks
    # ------------------------------------------------------------------ #
    def _on_stop(self) -> None:
        if isinstance(self._worker, ChatWorker):
            self._worker.request_stop()
            self._stop_btn.setEnabled(False)

    def _on_chunk(self, text: str) -> None:
        self._assistant_buffer += text
        self._render_all()

    def _on_stream_finished(self) -> None:
        text = self._assistant_buffer or "(no content returned)"
        self._turns.append({"role": "assistant", "text": text, "img_keys": []})
        self._assistant_buffer = ""
        self._render_all()
        self._cleanup_worker()

    def _on_image_message(self, message: dict) -> None:
        images = message.get("images") or []
        content = message.get("content")
        text = content if isinstance(content, str) else ""
        img_keys: list[str] = []
        for img in images:
            url = ""
            if isinstance(img, dict):
                url = (img.get("image_url") or {}).get("url", "") or img.get("url", "")
            if url:
                key = self._store_data_url_image(url)
                if key:
                    img_keys.append(key)
        if not img_keys and not text:
            text = "(model returned no image or text)"
        self._turns.append({"role": "assistant", "text": text, "img_keys": img_keys})
        self._assistant_buffer = ""
        self._render_all()
        self._cleanup_worker()

    def _on_failed(self, message: str) -> None:
        self._assistant_buffer = ""
        self._turns.append(
            {"role": "assistant", "text": f"⚠️ {message}", "img_keys": []}
        )
        self._render_all()
        self._cleanup_worker()

    def _cleanup_worker(self) -> None:
        if self._worker is not None:
            self._worker.wait(50)
            self._worker = None
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        self._send_btn.setEnabled(not busy and self._model is not None)
        # Only streaming chats can be stopped mid-flight.
        self._stop_btn.setEnabled(busy and isinstance(self._worker, ChatWorker))
        self._input.setEnabled(not busy)

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _store_data_url_image(self, data_url: str) -> str | None:
        """Decode a ``data:image/...;base64,`` URL into a registered QImage."""
        if "base64," not in data_url:
            return None
        try:
            raw = base64.b64decode(data_url.split("base64,", 1)[1])
        except (ValueError, TypeError):
            return None
        image = QImage()
        if not image.loadFromData(raw):
            return None
        key = f"orimg://{self._img_counter}"
        self._img_counter += 1
        self._transcript.register_image(key, image)
        return key

    def _render_all(self) -> None:
        parts: list[str] = []
        for turn in self._turns:
            speaker = "You" if turn["role"] == "user" else (
                self._model.name if self._model else "assistant"
            )
            parts.append(self._turn_html(speaker, turn["text"], turn.get("img_keys")))
        if self._worker is not None or self._assistant_buffer:
            name = self._model.name if self._model else "assistant"
            parts.append(self._turn_html(name, self._assistant_buffer, None))
        self._transcript.setHtml("".join(parts))
        bar = self._transcript.verticalScrollBar()
        bar.setValue(bar.maximum())

    @staticmethod
    def _turn_html(speaker: str, text: str, img_keys: list[str] | None) -> str:
        color = "#4a9eff" if speaker == "You" else "#3fb950"
        body = _html_escape(text) if text else ""
        images_html = ""
        for key in img_keys or []:
            images_html += (
                f'<br><img src="{key}" style="max-width:420px;" width="420"><br>'
            )
        return (
            f'<p style="margin:6px 0;"><b style="color:{color};">{speaker}:</b> '
            f'<span>{body}</span>{images_html}</p>'
        )

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if (
            event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self._on_send()
            return
        super().keyPressEvent(event)


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )
