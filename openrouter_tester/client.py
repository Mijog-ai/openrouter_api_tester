"""Thin wrapper around the OpenRouter REST API.

Only two endpoints are needed for this app:

* ``GET  /api/v1/models``            -> list every available model
* ``POST /api/v1/chat/completions``  -> run a (streaming) chat request

The client is intentionally free of any Qt imports so it can be unit tested
and reused outside of the GUI.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable, Iterator

import requests

BASE_URL = "https://openrouter.ai/api/v1"

# Sent so requests show up nicely on the OpenRouter dashboard. Both headers
# are optional per the OpenRouter docs but recommended.
DEFAULT_HEADERS = {
    "HTTP-Referer": "https://github.com/mijog-ai/openrouter_api_tester",
    "X-Title": "OpenRouter API Tester",
}


class OpenRouterError(RuntimeError):
    """Raised when the API returns a non-success response."""


class OpenRouterClient:
    def __init__(self, api_key: str = "", timeout: int = 60) -> None:
        self.api_key = api_key.strip()
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = dict(DEFAULT_HEADERS)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if extra:
            headers.update(extra)
        return headers

    # ------------------------------------------------------------------ #
    # Models
    # ------------------------------------------------------------------ #
    def list_models(self) -> list[dict[str, Any]]:
        """Return the raw list of model objects from OpenRouter.

        The models endpoint does not strictly require an API key, so this can
        be called before the user has entered one.
        """
        url = f"{BASE_URL}/models"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=self.timeout)
        except requests.RequestException as exc:  # network layer failure
            raise OpenRouterError(f"Failed to reach OpenRouter: {exc}") from exc

        if resp.status_code != 200:
            raise OpenRouterError(
                f"Model list request failed ({resp.status_code}): {resp.text[:300]}"
            )

        payload = resp.json()
        data = payload.get("data", payload)
        if not isinstance(data, list):
            raise OpenRouterError("Unexpected response shape from /models")
        return data

    def fetch_bytes(self, url: str) -> bytes:
        """Download raw bytes from an http(s) URL (for remote image results)."""
        try:
            resp = requests.get(url, timeout=self.timeout)
        except requests.RequestException as exc:
            raise OpenRouterError(f"Failed to download image: {exc}") from exc
        if resp.status_code != 200:
            raise OpenRouterError(f"Image download failed ({resp.status_code}).")
        return resp.content

    # ------------------------------------------------------------------ #
    # Chat completions
    # ------------------------------------------------------------------ #
    def chat_completion_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = 0.7,
        max_tokens: int | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[str]:
        """Yield text chunks for a streaming chat completion.

        ``messages`` follows the OpenAI/OpenRouter schema. Each element is a
        dict with ``role`` and ``content`` where content may be a plain string
        or a list of content parts (for multimodal / vision requests).

        ``temperature`` of ``None`` omits the parameter entirely. If a model
        rejects an optional parameter (e.g. some models don't accept
        ``temperature``), the request is retried once without it.

        ``should_stop`` is polled between chunks so a GUI thread can cancel an
        in-flight generation.
        """
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens:
            body["max_tokens"] = max_tokens

        resp = self._post_chat(body, stream=True)
        yield from self._iter_sse(resp, should_stop)

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = 0.7,
        max_tokens: int | None = None,
        modalities: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run a single non-streaming chat completion.

        Returns the assistant ``message`` object from ``choices[0]`` (which may
        contain ``content`` text and/or an ``images`` list for image-generation
        models), augmented with a top-level ``usage`` key when present.

        ``temperature`` of ``None`` omits the parameter. Parameters a model
        rejects as unsupported are stripped and the request retried.
        ``modalities`` lets callers request image output, e.g.
        ``["image", "text"]`` for image-generation models.
        """
        body: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens:
            body["max_tokens"] = max_tokens
        if modalities:
            body["modalities"] = modalities

        resp = self._post_chat(body, stream=False)

        payload = resp.json()
        choices = payload.get("choices") or []
        if not choices:
            # Some errors come back 200 with an ``error`` field.
            err = payload.get("error")
            if err:
                raise OpenRouterError(str(err.get("message") or err))
            raise OpenRouterError("No choices returned by the model.")
        message = dict(choices[0].get("message") or {})
        if "usage" in payload:
            message["usage"] = payload["usage"]
        return message

    # Optional parameters we're willing to drop and retry without if a model
    # reports them as unsupported. Never strip model/messages.
    _RETRYABLE_PARAMS = ("temperature", "max_tokens", "top_p", "modalities")

    def _post_chat(self, body: dict[str, Any], *, stream: bool) -> requests.Response:
        """POST to /chat/completions, retrying without unsupported parameters.

        Some providers reject optional sampling parameters (e.g. GPT-5 image
        rejects ``temperature``). On a 400 that names an unsupported parameter,
        we remove it and retry, up to a few times.
        """
        if not self.api_key:
            raise OpenRouterError("An OpenRouter API key is required to run chats.")

        url = f"{BASE_URL}/chat/completions"
        body = dict(body)
        for _ in range(len(self._RETRYABLE_PARAMS) + 1):
            try:
                resp = requests.post(
                    url,
                    headers=self._headers({"Content-Type": "application/json"}),
                    data=json.dumps(body),
                    stream=stream,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                raise OpenRouterError(f"Request failed: {exc}") from exc

            if resp.status_code == 200:
                return resp

            detail = resp.text[:800]
            resp.close()
            dropped = self._strip_unsupported_param(body, detail)
            if not dropped:
                raise OpenRouterError(
                    f"Chat request failed ({resp.status_code}): {detail}"
                )
            # else: loop and retry without the offending parameter.

        raise OpenRouterError("Chat request failed after stripping parameters.")

    @classmethod
    def _strip_unsupported_param(cls, body: dict[str, Any], error_text: str) -> bool:
        """Remove a parameter the error names as unsupported. Return True if one
        was removed."""
        lowered = error_text.lower()
        if "unsupported" not in lowered and "not supported" not in lowered:
            return False
        for param in cls._RETRYABLE_PARAMS:
            if param in body and f"'{param}'" in error_text:
                body.pop(param, None)
                return True
        # Fall back: if the message clearly flags a parameter we set but didn't
        # match by quotes, drop the first retryable param present.
        for param in cls._RETRYABLE_PARAMS:
            if param in body and param in lowered:
                body.pop(param, None)
                return True
        return False

    @staticmethod
    def _iter_sse(
        resp: requests.Response,
        should_stop: Callable[[], bool] | None,
    ) -> Iterator[str]:
        try:
            for raw_line in resp.iter_lines(decode_unicode=True):
                if should_stop and should_stop():
                    break
                if not raw_line:
                    continue
                if raw_line.startswith(":"):
                    # SSE comment / keep-alive ping.
                    continue
                if not raw_line.startswith("data:"):
                    continue
                data = raw_line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for piece in _extract_delta_text(chunk):
                    yield piece
        finally:
            resp.close()


def _extract_delta_text(chunk: dict[str, Any]) -> Iterable[str]:
    """Pull assistant text out of a streaming chunk, tolerating shape drift."""
    choices = chunk.get("choices")
    if not choices:
        return
    for choice in choices:
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str) and content:
            yield content
        elif isinstance(content, list):
            # Some providers stream structured content parts.
            for part in content:
                text = part.get("text") if isinstance(part, dict) else None
                if text:
                    yield text


def extract_image_urls(message: dict[str, Any]) -> list[str]:
    """Collect image URLs from an assistant message, checking every known shape.

    Different providers place generated images in different spots, so we look
    in all of them:

    * ``message["images"]``: ``[{"image_url": {"url": ...}}]`` or ``[{"url": ...}]``
    * ``message["content"]`` when it's a list of parts with ``image_url``
    * any ``data:image`` or ``http(s)`` image URL embedded in a content string

    Returns a de-duplicated list of URLs (data: or http(s)).
    """
    urls: list[str] = []

    def add(url: Any) -> None:
        if isinstance(url, str) and url and url not in urls:
            urls.append(url)

    # 1) Dedicated images array.
    for img in message.get("images") or []:
        if isinstance(img, dict):
            add((img.get("image_url") or {}).get("url"))
            add(img.get("url"))
        elif isinstance(img, str):
            add(img)

    # 2) Structured content parts.
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                add((part.get("image_url") or {}).get("url"))
                add(part.get("url"))

    # 3) Image URLs embedded in a plain-text content string (markdown etc.).
    if isinstance(content, str) and content:
        import re

        for m in re.findall(r"data:image/[^\s\)\"']+", content):
            add(m)
        for m in re.findall(r"https?://[^\s\)\"']+\.(?:png|jpe?g|webp|gif)", content):
            add(m)

    return urls


def message_text(message: dict[str, Any]) -> str:
    """Return the plain-text portion of an assistant message, if any."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        return " ".join(t for t in parts if t)
    return ""
