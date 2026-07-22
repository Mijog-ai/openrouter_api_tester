"""The chat playground panel."""

from __future__ import annotations

import base64
import mimetypes
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
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
from ..workers import ChatWorker


class ChatWidget(QWidget):
    """A minimal but functional chat playground.

    Keeps its own conversation history and, for vision-capable models, lets the
    user attach an image that is sent inline with the next message.
    """

    def __init__(self, client: OpenRouterClient, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._client = client
        self._model: Model | None = None
        self._history: list[dict[str, Any]] = []
        self._pending_image: dict[str, str] | None = None  # {data_url, name}
        self._worker: ChatWorker | None = None
        self._assistant_buffer = ""

        self._build_ui()
        self._set_model(None)

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Header: current model + sampling controls.
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

        # Transcript.
        self._transcript = QTextEdit()
        self._transcript.setReadOnly(True)
        self._transcript.setPlaceholderText(
            "Select a model on the left, then start chatting to test it."
        )
        layout.addWidget(self._transcript, stretch=1)

        # Attachment status line.
        self._attach_label = QLabel("")
        self._attach_label.setStyleSheet("color: #888;")
        layout.addWidget(self._attach_label)

        # Input row.
        input_row = QHBoxLayout()
        self._input = QPlainTextEdit()
        self._input.setPlaceholderText("Type a message…  (Ctrl+Enter to send)")
        self._input.setFixedHeight(80)
        input_row.addWidget(self._input, stretch=1)

        btn_col = QVBoxLayout()
        self._attach_btn = QPushButton("Attach image")
        self._attach_btn.clicked.connect(self._on_attach)
        btn_col.addWidget(self._attach_btn)

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
        """Switch the active model. Existing history is preserved."""
        self._set_model(model)

    def clear_chat(self) -> None:
        self._history.clear()
        self._transcript.clear()
        self._clear_attachment()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _set_model(self, model: Model | None) -> None:
        self._model = model
        if model is None:
            self._model_label.setText("No model selected")
            self._send_btn.setEnabled(False)
            self._attach_btn.setEnabled(False)
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
        self._attach_btn.setEnabled(model.supports_images)
        if not model.supports_images:
            self._clear_attachment()

    def _on_attach(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select an image",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.gif)",
        )
        if not path:
            return
        mime, _ = mimetypes.guess_type(path)
        if not mime:
            mime = "image/png"
        try:
            with open(path, "rb") as fh:
                encoded = base64.b64encode(fh.read()).decode("ascii")
        except OSError as exc:
            QMessageBox.warning(self, "Attach failed", str(exc))
            return
        name = path.rsplit("/", 1)[-1]
        self._pending_image = {
            "data_url": f"data:{mime};base64,{encoded}",
            "name": name,
        }
        self._attach_label.setText(f"📎 Attached: {name}  (sent with next message)")

    def _clear_attachment(self) -> None:
        self._pending_image = None
        self._attach_label.setText("")

    def _build_user_content(self, text: str) -> Any:
        """Return message content, using the multimodal list form if needed."""
        if not self._pending_image:
            return text
        parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": self._pending_image["data_url"]},
            }
        )
        return parts

    def _on_send(self) -> None:
        if self._model is None or self._worker is not None:
            return
        text = self._input.toPlainText().strip()
        if not text and not self._pending_image:
            return
        if not self._client.api_key:
            QMessageBox.information(
                self,
                "API key required",
                "Enter your OpenRouter API key (top of the window) before chatting.",
            )
            return

        content = self._build_user_content(text)
        self._history.append({"role": "user", "content": content})
        self._clear_attachment()
        self._input.clear()

        # Prepare an empty assistant buffer and render the new state.
        self._assistant_buffer = ""
        self._render_all()

        max_tokens = self._max_tokens_spin.value() or None
        self._worker = ChatWorker(
            self._client,
            self._model.id,
            list(self._history),
            temperature=self._temp_spin.value(),
            max_tokens=max_tokens,
        )
        self._worker.chunk.connect(self._on_chunk)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._set_busy(True)
        self._worker.start()

    def _on_stop(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()
            self._stop_btn.setEnabled(False)

    def _on_chunk(self, text: str) -> None:
        self._assistant_buffer += text
        self._replace_last_assistant_text(self._assistant_buffer)

    def _on_finished(self) -> None:
        if self._assistant_buffer:
            self._history.append(
                {"role": "assistant", "content": self._assistant_buffer}
            )
        else:
            self._replace_last_assistant_text("(no content returned)")
        self._cleanup_worker()

    def _on_failed(self, message: str) -> None:
        self._replace_last_assistant_text(f"⚠️ {message}")
        # Drop the user turn we optimistically added so retrying is clean.
        if self._history and self._history[-1]["role"] == "user":
            self._history.pop()
        self._cleanup_worker()

    def _cleanup_worker(self) -> None:
        if self._worker is not None:
            self._worker.wait(50)
            self._worker = None
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        self._send_btn.setEnabled(not busy and self._model is not None)
        self._stop_btn.setEnabled(busy)
        self._input.setEnabled(not busy)

    # ------------------------------------------------------------------ #
    # Transcript rendering
    # ------------------------------------------------------------------ #
    def _replace_last_assistant_text(self, text: str) -> None:
        """Rewrite the in-progress assistant paragraph as streaming grows.

        Rather than fiddle with a QTextCursor mid-document, we re-render the
        whole transcript from history plus the live buffer. History drives
        correctness; the transcript is purely presentational.
        """
        self._render_all()

    def _render_all(self) -> None:
        html_parts = []
        for msg in self._history:
            speaker = "You" if msg["role"] == "user" else (
                self._model.name if self._model else "assistant"
            )
            html_parts.append(self._turn_html(speaker, _content_to_text(msg["content"])))
        # Live (in-progress) assistant turn.
        if self._worker is not None or self._assistant_buffer:
            name = self._model.name if self._model else "assistant"
            html_parts.append(self._turn_html(name, self._assistant_buffer))
        self._transcript.setHtml("".join(html_parts))
        self._scroll_to_end()

    @staticmethod
    def _turn_html(speaker: str, text: str) -> str:
        color = "#4a9eff" if speaker == "You" else "#3fb950"
        safe = _html_escape(text)
        return (
            f'<p style="margin:6px 0;"><b style="color:{color};">{speaker}:</b> '
            f'<span>{safe}</span></p>'
        )

    def _scroll_to_end(self) -> None:
        bar = self._transcript.verticalScrollBar()
        bar.setValue(bar.maximum())

    # Keyboard: Ctrl+Enter to send.
    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if (
            event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self._on_send()
            return
        super().keyPressEvent(event)


def _content_to_text(content: Any) -> str:
    """Flatten message content (string or multimodal parts) to display text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                chunks.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                chunks.append("[📎 image]")
        return " ".join(c for c in chunks if c)
    return str(content)


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )
