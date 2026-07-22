"""Model normalisation and categorisation.

OpenRouter describes each model with an ``architecture`` block containing
``input_modalities`` and ``output_modalities``. We use those to bucket models
into human-friendly categories and to drop anything audio related, per the
project requirement to test every model type *except* audio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Category identifiers (ordering here drives display order in the sidebar).
CAT_IMAGE_GEN = "Image Generation"
CAT_VISION = "Vision / Multimodal"
CAT_DOCUMENT = "Document / File"
CAT_TEXT = "Text"

CATEGORY_ORDER = [CAT_TEXT, CAT_VISION, CAT_IMAGE_GEN, CAT_DOCUMENT]

CATEGORY_DESCRIPTIONS = {
    CAT_TEXT: "Text-in / text-out chat and completion models.",
    CAT_VISION: "Accept images or video alongside text (multimodal input).",
    CAT_IMAGE_GEN: "Produce images as output.",
    CAT_DOCUMENT: "Accept file/document input (e.g. PDFs) alongside text.",
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
    def is_audio(self) -> bool:
        return "audio" in self.input_modalities or "audio" in self.output_modalities

    @property
    def category(self) -> str:
        # Priority: image generation > vision/multimodal > document > text.
        if self.generates_images:
            return CAT_IMAGE_GEN
        if self.supports_images or self.supports_video:
            return CAT_VISION
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
    """Return ``{category: [Model, ...]}`` with audio models excluded.

    Categories are returned in :data:`CATEGORY_ORDER`; empty categories are
    omitted. Models within a category are sorted by name.
    """
    buckets: dict[str, list[Model]] = {cat: [] for cat in CATEGORY_ORDER}

    for raw in raw_models:
        model = model_from_raw(raw)
        if not model.id:
            continue
        if model.is_audio:  # requirement: skip audio models entirely
            continue
        buckets.setdefault(model.category, []).append(model)

    for models in buckets.values():
        models.sort(key=lambda m: m.name.lower())

    # Drop empty categories while preserving order.
    return {cat: buckets[cat] for cat in CATEGORY_ORDER if buckets.get(cat)}
