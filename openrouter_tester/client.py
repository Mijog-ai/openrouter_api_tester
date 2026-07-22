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

    # ------------------------------------------------------------------ #
    # Chat completions
    # ------------------------------------------------------------------ #
    def chat_completion_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[str]:
        """Yield text chunks for a streaming chat completion.

        ``messages`` follows the OpenAI/OpenRouter schema. Each element is a
        dict with ``role`` and ``content`` where content may be a plain string
        or a list of content parts (for multimodal / vision requests).

        ``should_stop`` is polled between chunks so a GUI thread can cancel an
        in-flight generation.
        """
        if not self.api_key:
            raise OpenRouterError("An OpenRouter API key is required to run chats.")

        url = f"{BASE_URL}/chat/completions"
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            body["max_tokens"] = max_tokens

        try:
            resp = requests.post(
                url,
                headers=self._headers({"Content-Type": "application/json"}),
                data=json.dumps(body),
                stream=True,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise OpenRouterError(f"Request failed: {exc}") from exc

        if resp.status_code != 200:
            # Read the (small) error body eagerly for a useful message.
            detail = resp.text[:500]
            resp.close()
            raise OpenRouterError(f"Chat request failed ({resp.status_code}): {detail}")

        yield from self._iter_sse(resp, should_stop)

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        modalities: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run a single non-streaming chat completion.

        Returns the assistant ``message`` object from ``choices[0]`` (which may
        contain ``content`` text and/or an ``images`` list for image-generation
        models), augmented with a top-level ``usage`` key when present.

        ``modalities`` lets callers request image output, e.g.
        ``["image", "text"]`` for image-generation models.
        """
        if not self.api_key:
            raise OpenRouterError("An OpenRouter API key is required to run chats.")

        url = f"{BASE_URL}/chat/completions"
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            body["max_tokens"] = max_tokens
        if modalities:
            body["modalities"] = modalities

        try:
            resp = requests.post(
                url,
                headers=self._headers({"Content-Type": "application/json"}),
                data=json.dumps(body),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise OpenRouterError(f"Request failed: {exc}") from exc

        if resp.status_code != 200:
            raise OpenRouterError(
                f"Chat request failed ({resp.status_code}): {resp.text[:500]}"
            )

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
