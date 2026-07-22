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


class VideoModelFetchWorker(QThread):
    """Fetch the list of video-generation models off the UI thread."""

    finished_ok = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, client: OpenRouterClient) -> None:
        super().__init__()
        self._client = client

    def run(self) -> None:
        try:
            models = self._client.list_video_models()
        except OpenRouterError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            self.failed.emit(f"Unexpected error: {exc}")
        else:
            self.finished_ok.emit(models)


class VideoGenWorker(QThread):
    """Run an async video-generation job: create → poll → download the mp4."""

    status = pyqtSignal(str)  # human-readable progress updates
    finished_ok = pyqtSignal(bytes)  # the finished mp4 bytes
    failed = pyqtSignal(str)

    _TERMINAL_OK = {"completed", "succeeded", "success"}
    _TERMINAL_FAIL = {"failed", "error", "cancelled", "canceled"}

    def __init__(
        self,
        client: OpenRouterClient,
        bodies: list[dict[str, Any]] | dict[str, Any],
        *,
        poll_seconds: float = 4.0,
        timeout_seconds: float = 600.0,
    ) -> None:
        super().__init__()
        self._client = client
        # Accept either a single body or a fallback ladder of candidate bodies.
        self._bodies = [bodies] if isinstance(bodies, dict) else list(bodies)
        self._poll = poll_seconds
        self._timeout = timeout_seconds
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def _create_with_fallback(self) -> dict[str, Any]:
        """Try each candidate body; on a 400 move to the next. Raise otherwise."""
        last_error: OpenRouterError | None = None
        for i, body in enumerate(self._bodies):
            if self._stop_requested:
                raise OpenRouterError("Cancelled.")
            if i > 0:
                self.status.emit(f"Retrying with an alternate request ({i + 1})…")
            try:
                return self._client.create_video_job(body)
            except OpenRouterError as exc:
                # Only a 400 (bad params, no job created) is worth retrying.
                if "(400" in str(exc):
                    last_error = exc
                    continue
                raise
        raise last_error or OpenRouterError("Video job creation failed.")

    def run(self) -> None:
        try:
            self.status.emit("Submitting job…")
            created = self._create_with_fallback()
        except OpenRouterError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # pragma: no cover - defensive
            self.failed.emit(f"Unexpected error: {exc}")
            return

        job_id = created.get("id")
        if not job_id:
            self.failed.emit(f"No job id in response: {created}")
            return

        waited = 0.0
        while True:
            if self._stop_requested:
                self.failed.emit("Cancelled.")
                return
            if waited >= self._timeout:
                self.failed.emit("Timed out waiting for the video.")
                return
            try:
                job = self._client.get_video_job(job_id)
            except OpenRouterError as exc:
                self.failed.emit(str(exc))
                return

            state = str(job.get("status", "")).lower()
            if state in self._TERMINAL_OK:
                break
            if state in self._TERMINAL_FAIL:
                detail = job.get("error") or job.get("failure_reason") or state
                self.failed.emit(f"Generation {state}: {detail}")
                return
            self.status.emit(f"Status: {state or 'working'}… ({int(waited)}s)")
            self._sleep(self._poll)
            waited += self._poll

        self.status.emit("Downloading video…")
        try:
            data = self._client.download_video_content(job_id, index=0)
        except OpenRouterError as exc:
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(data)

    def _sleep(self, seconds: float) -> None:
        # Sleep in small slices so a stop request is honoured quickly.
        end = time.perf_counter() + seconds
        while time.perf_counter() < end and not self._stop_requested:
            time.sleep(0.1)


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
