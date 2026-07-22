"""Background QThread workers so network I/O never blocks the UI thread."""

from __future__ import annotations

import time
from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal

import base64

from .catalog import Model
from .client import (
    OpenRouterClient,
    OpenRouterError,
    extract_image_urls,
    message_text,
)


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


class CompletionWorker(QThread):
    """Run a single non-streaming completion (used for image generation)."""

    finished_ok = pyqtSignal(dict)  # the assistant ``message`` object
    failed = pyqtSignal(str)

    def __init__(
        self,
        client: OpenRouterClient,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        modalities: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._client = client
        self._model = model
        self._messages = messages
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._modalities = modalities

    def run(self) -> None:
        try:
            message = self._client.chat_completion(
                self._model,
                self._messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                modalities=self._modalities,
            )
        except OpenRouterError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            self.failed.emit(f"Unexpected error: {exc}")
        else:
            self.finished_ok.emit(message)


class ImageGenWorker(QThread):
    """Generate images: run the completion, then resolve every image to bytes.

    Emits ``finished_ok(images_bytes, text)`` where ``images_bytes`` is a list
    of raw image byte-strings (data URLs decoded, remote URLs downloaded) and
    ``text`` is any accompanying text the model returned.
    """

    finished_ok = pyqtSignal(list, str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        client: OpenRouterClient,
        model: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self._client = client
        self._model = model
        self._messages = messages
        self._max_tokens = max_tokens

    def run(self) -> None:
        try:
            message = self._client.chat_completion(
                self._model,
                self._messages,
                temperature=None,  # image models commonly reject sampling params
                max_tokens=self._max_tokens,
                modalities=["image", "text"],
            )
        except OpenRouterError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # pragma: no cover - defensive
            self.failed.emit(f"Unexpected error: {exc}")
            return

        urls = extract_image_urls(message)
        text = message_text(message)
        images: list[bytes] = []
        for url in urls:
            data = self._resolve(url)
            if data:
                images.append(data)
        self.finished_ok.emit(images, text)

    def _resolve(self, url: str) -> bytes | None:
        if url.startswith("data:") and "base64," in url:
            try:
                return base64.b64decode(url.split("base64,", 1)[1])
            except (ValueError, TypeError):
                return None
        if url.startswith("http://") or url.startswith("https://"):
            try:
                return self._client.fetch_bytes(url)
            except OpenRouterError:
                return None
        return None


class BatchTestWorker(QThread):
    """Probe a list of models sequentially with a small prompt.

    Emits one ``result`` per model as it completes so the UI can fill a table
    live. Runs sequentially (not in parallel) to stay friendly to rate limits.
    """

    # (index, model_id, ok, latency_seconds, detail)
    result = pyqtSignal(int, str, bool, float, str)
    progress = pyqtSignal(int, int)  # done, total
    finished_all = pyqtSignal()

    def __init__(
        self,
        client: OpenRouterClient,
        models: list[Model],
        prompt: str,
        *,
        max_tokens: int = 32,
    ) -> None:
        super().__init__()
        self._client = client
        self._models = models
        self._prompt = prompt
        self._max_tokens = max_tokens
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        total = len(self._models)
        for index, model in enumerate(self._models):
            if self._stop_requested:
                break
            ok, latency, detail = self._probe(model)
            self.result.emit(index, model.id, ok, latency, detail)
            self.progress.emit(index + 1, total)
        self.finished_all.emit()

    def _probe(self, model: Model) -> tuple[bool, float, str]:
        messages = [{"role": "user", "content": self._prompt}]
        modalities = ["image", "text"] if model.generates_images else None
        # Image models often reject sampling params; omit temperature for them.
        temperature = None if model.generates_images else 0.2
        start = time.perf_counter()
        try:
            message = self._client.chat_completion(
                model.id,
                messages,
                temperature=temperature,
                max_tokens=self._max_tokens,
                modalities=modalities,
            )
        except OpenRouterError as exc:
            return False, time.perf_counter() - start, str(exc)
        except Exception as exc:  # pragma: no cover - defensive
            return False, time.perf_counter() - start, f"Unexpected error: {exc}"

        latency = time.perf_counter() - start
        images = message.get("images") or []
        content = message.get("content")
        text = content if isinstance(content, str) else _flatten_content(content)
        if model.generates_images:
            if images:
                return True, latency, f"{len(images)} image(s) returned"
            # Some image models also answer in text; treat text as partial pass.
            if text.strip():
                return True, latency, "text only (no image)"
            return False, latency, "empty response"
        snippet = text.strip().replace("\n", " ")
        if snippet:
            return True, latency, snippet[:80]
        return False, latency, "empty response"


def _flatten_content(content: Any) -> str:
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return " ".join(parts)
    return "" if content is None else str(content)
