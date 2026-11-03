"""DYPOL style guide loader — the one place that reads style/style.guide.yaml."""

from functools import lru_cache
from pathlib import Path

import yaml

STYLE_DIR = Path(__file__).resolve().parent.parent / "style"
STYLE_FILE = STYLE_DIR / "style.guide.yaml"


@lru_cache(maxsize=1)
def load_style() -> dict:
    """Parsed style guide (cached)."""
    with STYLE_FILE.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def block_kinds() -> tuple[str, ...]:
    return tuple(load_style()["blocks"]["kinds"])


def max_same_run() -> int:
    return int(load_style()["variety"]["max_same_kind_in_a_row"])


def hook_rules() -> dict:
    return dict(load_style()["hook"])


def block_limits(kind: str) -> dict:
    return dict(load_style()["blocks"].get("limits", {}).get(kind, {}))


def hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def color(name: str) -> tuple[int, int, int]:
    """Palette colour as an RGB tuple ('background', 'lime', 'cyan', 'amber', ...)."""
    return hex_rgb(load_style()["look"]["palette"][name])


def font_path(role: str) -> Path:
    """Absolute path of a vendored font ('display_bold', 'display_medium', 'mono', 'mono_bold')."""
    return STYLE_DIR / load_style()["look"]["fonts"][role]


def screen() -> tuple[int, int, int]:
    s = load_style()["screen"]
    return int(s["width"]), int(s["height"]), int(s["fps"])


def voice_settings() -> dict:
    """The brand narration voice block (empty dict when the style file has none)."""
    return dict(load_style().get("voice") or {})
