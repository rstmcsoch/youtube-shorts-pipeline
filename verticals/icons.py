"""Simple line icons drawn with Pillow (original DYPOL drawings, no third-party icon set).

Each icon is drawn in a unit box and scaled to `size`; stroke colour and width are
parameters so a block can fade or highlight it. pick_icon() maps topic words to an
icon name. Licence: original work for this repository (no external assets).
"""

import math
import re
from functools import lru_cache

from PIL import Image, ImageDraw

ICON_WORDS = {
    "phone": "phone smartphone mobile iphone android device screen app unlock unlocks handset",
    "card": "card cards credit debit payment payments transaction transactions purchase purchases bank banking "
            "money swipe checkout wallet pay",
    "lock": "lock locked security secure password passcode encrypt encryption privacy pin key keys authentication",
    "face": "face faces facial selfie biometric biometrics identity person user customer customers depth",
    "wave": "audio speech voice sound wave waves spectrogram frequency microphone listen signal signals",
    "chip": "ai model models neural network networks chip gpu compute machine learning algorithm algorithms "
            "transformer training trained",
    "shield": "fraud protect protection defence defense guard safe safety prevent prevention prevented",
    "alert": "risk alert flag flags warning block blocked suspicious anomaly anomalies unusual criminal criminals "
             "attack attacker",
    "server": "data database server servers cloud storage history records pattern patterns",
    "check": "verify verified match matches approved approve pass accept confirmed accuracy accurate",
    "eye": "scan scans scanner detect detects detection camera infrared vision see sensor sensors dot dots",
}
_LOOKUP = {w: name for name, words in ICON_WORDS.items() for w in words.split()}
ICONS = tuple(ICON_WORDS)


def pick_icon(*texts, default: str = "chip") -> str:
    """First icon whose words appear in the texts (in order)."""
    for text in texts:
        items = text if isinstance(text, (list, tuple)) else [text]
        for item in items:
            for tok in re.findall(r"[a-z]+", str(item).lower()):
                if tok in _LOOKUP:
                    return _LOOKUP[tok]
    return default


def _draw(d: ImageDraw.ImageDraw, name: str, s: float, col, w: int) -> None:
    def P(x, y):
        return (x * s, y * s)

    def box(x0, y0, x1, y1, r=0.08):
        d.rounded_rectangle([P(x0, y0), P(x1, y1)], radius=r * s, outline=col, width=w)

    if name == "phone":
        box(0.28, 0.06, 0.72, 0.94, 0.09)
        d.line([P(0.43, 0.15), P(0.57, 0.15)], fill=col, width=w)
        d.ellipse([P(0.47, 0.82), P(0.53, 0.88)], outline=col, width=w)
    elif name == "card":
        box(0.08, 0.22, 0.92, 0.78, 0.07)
        d.line([P(0.08, 0.38), P(0.92, 0.38)], fill=col, width=w)
        d.rounded_rectangle([P(0.18, 0.5), P(0.34, 0.62)], radius=0.02 * s, outline=col, width=w)
        d.line([P(0.5, 0.66), P(0.8, 0.66)], fill=col, width=w)
    elif name == "lock":
        box(0.2, 0.44, 0.8, 0.92, 0.07)
        d.arc([P(0.3, 0.1), P(0.7, 0.62)], 180, 360, fill=col, width=w)
        d.line([P(0.3, 0.36), P(0.3, 0.46)], fill=col, width=w)
        d.line([P(0.7, 0.36), P(0.7, 0.46)], fill=col, width=w)
        d.line([P(0.5, 0.6), P(0.5, 0.74)], fill=col, width=w)
    elif name == "face":
        for (a, b) in (((0.08, 0.08), (0.08, 0.28)), ((0.08, 0.08), (0.28, 0.08)), ((0.92, 0.08), (0.72, 0.08)),
                       ((0.92, 0.08), (0.92, 0.28)), ((0.08, 0.92), (0.08, 0.72)), ((0.08, 0.92), (0.28, 0.92)),
                       ((0.92, 0.92), (0.72, 0.92)), ((0.92, 0.92), (0.92, 0.72))):
            d.line([P(*a), P(*b)], fill=col, width=w)
        d.ellipse([P(0.34, 0.36), P(0.4, 0.42)], fill=col)
        d.ellipse([P(0.6, 0.36), P(0.66, 0.42)], fill=col)
        d.line([P(0.5, 0.42), P(0.5, 0.56), P(0.45, 0.58)], fill=col, width=w)
        d.arc([P(0.34, 0.5), P(0.66, 0.76)], 20, 160, fill=col, width=w)
    elif name == "wave":
        pts = [P(0.05 + 0.9 * i / 40, 0.5 - 0.32 * math.sin(i / 40 * 4 * math.pi) * math.sin(i / 40 * math.pi))
               for i in range(41)]
        d.line(pts, fill=col, width=w, joint="curve")
    elif name == "chip":
        box(0.26, 0.26, 0.74, 0.74, 0.06)
        box(0.4, 0.4, 0.6, 0.6, 0.03)
        for k in (0.36, 0.5, 0.64):
            d.line([P(k, 0.08), P(k, 0.26)], fill=col, width=w)
            d.line([P(k, 0.74), P(k, 0.92)], fill=col, width=w)
            d.line([P(0.08, k), P(0.26, k)], fill=col, width=w)
            d.line([P(0.74, k), P(0.92, k)], fill=col, width=w)
    elif name == "shield":
        d.line([P(0.5, 0.06), P(0.86, 0.2), P(0.82, 0.56), P(0.5, 0.94), P(0.18, 0.56), P(0.14, 0.2), P(0.5, 0.06)],
               fill=col, width=w, joint="curve")
        d.line([P(0.34, 0.5), P(0.46, 0.62), P(0.68, 0.38)], fill=col, width=w, joint="curve")
    elif name == "alert":
        d.line([P(0.5, 0.08), P(0.94, 0.88), P(0.06, 0.88), P(0.5, 0.08)], fill=col, width=w, joint="curve")
        d.line([P(0.5, 0.36), P(0.5, 0.62)], fill=col, width=w)
        d.ellipse([P(0.47, 0.71), P(0.53, 0.77)], fill=col)
    elif name == "server":
        for y in (0.1, 0.4, 0.7):
            box(0.12, y, 0.88, y + 0.22, 0.05)
            d.ellipse([P(0.22, y + 0.08), P(0.28, y + 0.14)], fill=col)
            d.line([P(0.5, y + 0.11), P(0.78, y + 0.11)], fill=col, width=w)
    elif name == "check":
        d.ellipse([P(0.08, 0.08), P(0.92, 0.92)], outline=col, width=w)
        d.line([P(0.3, 0.52), P(0.44, 0.66), P(0.7, 0.38)], fill=col, width=w, joint="curve")
    elif name == "eye":
        d.arc([P(0.04, 0.22), P(0.96, 1.1)], 200, 340, fill=col, width=w)
        d.arc([P(0.04, -0.1), P(0.96, 0.78)], 20, 160, fill=col, width=w)
        d.ellipse([P(0.36, 0.36), P(0.64, 0.64)], outline=col, width=w)
        d.ellipse([P(0.46, 0.46), P(0.54, 0.54)], fill=col)
    else:
        d.ellipse([P(0.2, 0.2), P(0.8, 0.8)], outline=col, width=w)


@lru_cache(maxsize=128)
def icon_sprite(name: str, size: int, colour: tuple, width: int = 0) -> Image.Image:
    """RGBA sprite of an icon (cached)."""
    size = max(8, int(size))
    width = width or max(3, size // 16)
    scale = 2  # draw at 2x and downsample for smooth lines
    big = Image.new("RGBA", (size * scale, size * scale), (0, 0, 0, 0))
    _draw(ImageDraw.Draw(big), name, size * scale, tuple(colour) + (255,), width * scale)
    return big.resize((size, size), Image.LANCZOS)
