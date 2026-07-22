"""Minimal ``.env`` loading with no third-party dependency.

Python does not read ``.env`` files automatically, so a key placed there is
never visible via ``os.environ``. :func:`load_dotenv` fills that gap: it walks
up from the current directory (and the project root) looking for a ``.env``
file and loads any ``KEY=VALUE`` pairs into the environment.

Existing environment variables always win, so an explicitly exported key is
never overridden by the file.
"""

from __future__ import annotations

import os
from pathlib import Path


def _parse_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):  # tolerate `export KEY=VALUE`
        line = line[len("export "):].lstrip()
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    # Strip matching surrounding quotes.
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    if not key:
        return None
    return key, value


def find_dotenv(start: Path | None = None) -> Path | None:
    """Return the nearest ``.env`` file searching upward from ``start``.

    Also checks the package's project root so the app finds a ``.env`` sitting
    next to ``main.py`` regardless of the working directory it was launched in.
    """
    candidates: list[Path] = []
    start = (start or Path.cwd()).resolve()
    for directory in [start, *start.parents]:
        candidates.append(directory / ".env")
    # Project root (two levels up from this file: openrouter_tester/ -> repo).
    project_root = Path(__file__).resolve().parent.parent
    candidates.append(project_root / ".env")

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def load_dotenv(path: str | os.PathLike | None = None, *, override: bool = False) -> str | None:
    """Load a ``.env`` file into ``os.environ``.

    Returns the path that was loaded (as a string) or ``None`` if no file was
    found. When ``override`` is False (default), variables already present in
    the environment are left untouched.
    """
    env_path = Path(path) if path else find_dotenv()
    if env_path is None or not env_path.is_file():
        return None

    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return None

    for raw_line in text.splitlines():
        parsed = _parse_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        if override or key not in os.environ:
            os.environ[key] = value
    return str(env_path)
