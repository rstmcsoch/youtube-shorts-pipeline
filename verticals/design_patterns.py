"""Design-pattern registry (design-patterns/index.json): resolve $DYPOL_PATTERN to a ready pattern."""

import json
import os
from pathlib import Path

from .log import log

DEFAULT_PATTERN = "elements"
INDEX_PATH = Path(__file__).resolve().parent.parent / "design-patterns" / "index.json"


def load_patterns(path: Path = INDEX_PATH) -> list:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return [p for p in data if isinstance(p, dict) and p.get("id")]
    except (OSError, ValueError, TypeError):
        return [{"id": DEFAULT_PATTERN, "label": "Elements & animation", "status": "ready"}]


def resolve_design_pattern(value: str | None = None, path: Path = INDEX_PATH) -> str:
    """The pattern id to render: `value` (else $DYPOL_PATTERN) when it is a ready pattern;
    unknown or coming_soon patterns log a line and fall back to `elements`."""
    requested = str(value if value is not None else os.environ.get("DYPOL_PATTERN", "")).strip().lower()
    if not requested:
        return DEFAULT_PATTERN
    match = next((p for p in load_patterns(path) if str(p["id"]).lower() == requested), None)
    if match is None:
        log(f"DYPOL design pattern {requested[:60]!r} is unknown; falling back to {DEFAULT_PATTERN}")
        return DEFAULT_PATTERN
    if match.get("status") != "ready":
        log(f"DYPOL design pattern {requested!r} is {match.get('status', 'not ready')}; "
            f"falling back to {DEFAULT_PATTERN}")
        return DEFAULT_PATTERN
    return requested
