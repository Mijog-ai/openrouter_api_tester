"""Model normalisation and categorisation.

OpenRouter describes each model with an ``architecture`` block containing
``input_modalities`` and ``output_modalities``. We use those to bucket models
into human-friendly categories and to drop anything audio related, per the
project requirement to test every model type *except* audio.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Bundled preload of the model list so the app has data instantly on launch,
# even offline. Regenerate with ``python scripts/fetch_models.py``.
CACHE_PATH = Path(__file__).resolve().parent / "data" / "models.json"

# Category identifiers (ordering here drives display order in the sidebar).
#
# Categories are CAPABILITY FILTERS and are intentionally OVERLAPPING, mirroring
# how OpenRouter's own model filters work: a single model can appear under
# several categories (e.g. a model that accepts text, image and video shows up
# under Text, Image and Video). This is why per-category counts add up to more
# than the number of distinct models.
CAT_TEXT = "Text"
CAT_IMAGE = "Image (vision)"
CAT_VIDEO = "Video"
CAT_DOCUMENT = "Document / File"
CAT_IMAGE_GEN = "Image Generation"

CATEGORY_ORDER = [CAT_TEXT, CAT_IMAGE, CAT_VIDEO, CAT_DOCUMENT, CAT_IMAGE_GEN]

CATEGORY_DESCRIPTIONS = {
    CAT_TEXT: "Accept text input (chat / completion).",
    CAT_IMAGE: "Accept image input (vision / multimodal).",
    CAT_VIDEO: "Accept video input.",
    CAT_DOCUMENT: "Accept file/document input (e.g. PDFs).",
    CAT_IMAGE_GEN: "Produce images as output.",
}


@dataclass
class Model:
    """A normalised view over a raw OpenRouter model object."""

    id: str
    name: str
    description: str
    context_length: int
    input_modalities: list[str]
    output_modalities: list[str]
    modality: str
    pricing: dict[str, str] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    # --- capability helpers ------------------------------------------- #
    @property
    def supports_images(self) -> bool:
        return "image" in self.input_modalities

    @property
    def supports_video(self) -> bool:
        return "video" in self.input_modalities

    @property
    def supports_files(self) -> bool:
        return "file" in self.input_modalities

    @property
    def generates_images(self) -> bool:
        return "image" in self.output_modalities

    @property
    def accepts_audio(self) -> bool:
        return "audio" in self.input_modalities

    @property
    def generates_audio(self) -> bool:
        return "audio" in self.output_modalities

    @property
    def is_audio(self) -> bool:
        """True only for models whose OUTPUT is audio.

        These are the true "audio models" (TTS / music) whose output we cannot
        render, so they are excluded per the project requirement. Models that
        merely *accept* audio input but reply in text/image (e.g. Gemini) are
        kept — they are fully usable via their text/image/video capabilities.
        """
        return self.generates_audio

    @property
    def categories(self) -> list[str]:
        """Every capability category this model belongs to (overlapping)."""
        cats: list[str] = []
        if "text" in self.input_modalities:
            cats.append(CAT_TEXT)
        if self.supports_images:
            cats.append(CAT_IMAGE)
        if self.supports_video:
            cats.append(CAT_VIDEO)
        if self.supports_files:
            cats.append(CAT_DOCUMENT)
        if self.generates_images:
            cats.append(CAT_IMAGE_GEN)
        if not cats:
            # Anything with no recognised non-audio capability still lists as
            # Text so it remains reachable.
            cats.append(CAT_TEXT)
        return cats

    @property
    def primary_category(self) -> str:
        """A single representative category, for compact display."""
        if self.generates_images:
            return CAT_IMAGE_GEN
        if self.supports_video:
            return CAT_VIDEO
        if self.supports_images:
            return CAT_IMAGE
        if self.supports_files:
            return CAT_DOCUMENT
        return CAT_TEXT

    @property
    def is_free(self) -> bool:
        prompt = _to_float(self.pricing.get("prompt"))
        completion = _to_float(self.pricing.get("completion"))
        return prompt == 0.0 and completion == 0.0

    def price_summary(self) -> str:
        prompt = _to_float(self.pricing.get("prompt"))
        completion = _to_float(self.pricing.get("completion"))
        if prompt == 0.0 and completion == 0.0:
            return "Free"
        # Prices are per-token; show per-million tokens which is how people
        # usually reason about them.
        return (
            f"${prompt * 1_000_000:.2f} in / "
            f"${completion * 1_000_000:.2f} out (per 1M tokens)"
        )


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def model_from_raw(raw: dict[str, Any]) -> Model:
    arch = raw.get("architecture") or {}
    input_mods = arch.get("input_modalities") or []
    output_mods = arch.get("output_modalities") or []

    # Fall back to parsing the "modality" string if the explicit lists are
    # missing (older / sparse model entries).
    modality = arch.get("modality") or ""
    if not input_mods and "->" in modality:
        left = modality.split("->", 1)[0]
        input_mods = [m.strip() for m in left.split("+") if m.strip()]
    if not output_mods and "->" in modality:
        right = modality.split("->", 1)[1]
        output_mods = [m.strip() for m in right.split("+") if m.strip()]
    if not input_mods:
        input_mods = ["text"]
    if not output_mods:
        output_mods = ["text"]

    return Model(
        id=raw.get("id", ""),
        name=raw.get("name") or raw.get("id", ""),
        description=raw.get("description", "") or "",
        context_length=int(raw.get("context_length") or 0),
        input_modalities=[m.lower() for m in input_mods],
        output_modalities=[m.lower() for m in output_mods],
        modality=modality,
        pricing=raw.get("pricing") or {},
        raw=raw,
    )


def build_catalog(raw_models: list[dict[str, Any]]) -> dict[str, list[Model]]:
    """Return ``{category: [Model, ...]}`` using overlapping capability filters.

    A model appears under **every** category it qualifies for (so the same
    model can be listed under Text, Image and Video at once), matching how
    OpenRouter's own filters behave. Only true audio-output models are excluded
    (see :attr:`Model.is_audio`). Categories are returned in
    :data:`CATEGORY_ORDER`; empty ones are omitted, and each list is sorted by
    name.
    """
    buckets: dict[str, list[Model]] = {cat: [] for cat in CATEGORY_ORDER}

    for raw in raw_models:
        model = model_from_raw(raw)
        if not model.id:
            continue
        if model.is_audio:  # exclude audio-OUTPUT models (TTS / music)
            continue
        for category in model.categories:
            buckets.setdefault(category, []).append(model)

    for models in buckets.values():
        models.sort(key=lambda m: m.name.lower())

    # Drop empty categories while preserving order.
    return {cat: buckets[cat] for cat in CATEGORY_ORDER if buckets.get(cat)}


def flatten(catalog: dict[str, list[Model]]) -> list[Model]:
    """Return the DISTINCT models across all categories (each once, by id)."""
    seen: set[str] = set()
    result: list[Model] = []
    for models in catalog.values():
        for model in models:
            if model.id not in seen:
                seen.add(model.id)
                result.append(model)
    return result


def load_cached_raw() -> tuple[list[dict[str, Any]], str]:
    """Load the bundled model preload.

    Returns ``(raw_models, generated_at)``. If the cache is missing or
    unreadable, returns ``([], "")`` so the caller can fall back to the network.
    """
    try:
        with CACHE_PATH.open(encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return [], ""
    data = payload.get("data", [])
    if not isinstance(data, list):
        return [], ""
    return data, payload.get("generated_at", "")


def save_cache(raw_models: list[dict[str, Any]], generated_at: str) -> None:
    """Persist the raw model list as the bundled preload."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": generated_at, "data": raw_models}
    with CACHE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
