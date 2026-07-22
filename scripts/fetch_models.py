"""Regenerate the bundled model preload (openrouter_tester/data/models.json).

Run whenever you want to refresh the offline model list:

    python scripts/fetch_models.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running as a plain script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openrouter_tester.catalog import build_catalog, flatten, save_cache  # noqa: E402
from openrouter_tester.client import OpenRouterClient  # noqa: E402


def main() -> int:
    client = OpenRouterClient()
    print("Fetching models from OpenRouter…")
    raw = client.list_models()
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_cache(raw, generated_at)

    catalog = build_catalog(raw)
    distinct = len(flatten(catalog))
    print(
        f"Saved {len(raw)} raw models "
        f"({distinct} distinct after excluding audio-output models)."
    )
    print("Per-category counts (overlapping — a model can appear in several):")
    for category, models in catalog.items():
        print(f"  {category:22} {len(models)}")
    print(f"generated_at = {generated_at}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
