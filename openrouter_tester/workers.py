"""Background QThread workers so network I/O never blocks the UI thread."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal

from .client import OpenRouterClient, OpenRouterError


class ModelFetchWorker(QThread):
    """Fetch the raw model list off the UI thread."""

    finished_ok = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, client: OpenRouterClient) -> None:
        super().__init__()
        self._client = client

    def run(self) -> None:  # noqa: D401 - QThread entry point
        try:
            models = self._client.list_models()
        except OpenRouterError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            self.failed.emit(f"Unexpected error: {exc}")
        else:
            self.finished_ok.emit(models)


class ChatWorker(QThread):
    """Stream a chat completion, emitting text chunks as they arrive."""

    chunk = pyqtSignal(str)
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(
        self,
        client: OpenRouterClient,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self._client = client
        self._model = model
        self._messages = messages
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            stream = self._client.chat_completion_stream(
                self._model,
                self._messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                should_stop=lambda: self._stop_requested,
            )
            for piece in stream:
                if self._stop_requested:
                    break
                self.chunk.emit(piece)
        except OpenRouterError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            self.failed.emit(f"Unexpected error: {exc}")
        else:
            self.finished_ok.emit()
