"""Living backgrounds for Engine B motion blocks: never a flat, static slate.

Two kinds, both kept dark (mean luma well under 0.3) so text stays readable:

* footage: a topic clip or still for the beat's segment, downscaled, blurred,
  pulled ~72% toward the slate and moved with a slow Ken Burns push/pan.
* procedural: a slate field with two soft colour glows that drift, a faint grid
  that scrolls, and slow rising particles.

Backdrop.frame(t) returns a 1080x1920 RGB image for time t (seconds, continuous
across a shot's sub-shots). Frames are cheap: footage is decoded once at 180x320
and upscaled (it is blurred anyway); the procedural layers are pre-rendered and cropped.
"""

import math
import random
import subprocess
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from .style import color, screen

W, H, FPS = screen()
SMALL = (180, 320)
FOOTAGE_KEEP = 0.28      # share of the footage left after pulling it toward the slate
GRID_STEP = 64
PARTICLES = 34
MAX_FOOTAGE_LUMA = 0.2


def _mix(a, b, k):
    return tuple(int(round(a[i] + (b[i] - a[i]) * k)) for i in range(3))


@lru_cache(maxsize=4)
def _glow_field(seed: int) -> Image.Image:
    """Slate field (W+320 x H+320) with a cyan and a lime glow, built small and upscaled."""
    rng = random.Random(seed)
    sw, sh = (W + 320) // 8, (H + 320) // 8
    bg = color("background")
    img = Image.new("RGB", (sw, sh), bg)
    px = img.load()
    glows = [(color("cyan"), 0.16, rng.uniform(0.2, 0.45), rng.uniform(0.15, 0.4), 0.42),
             (color("lime"), 0.10, rng.uniform(0.55, 0.85), rng.uniform(0.55, 0.85), 0.38)]
    for y in range(sh):
        for x in range(sw):
            c = bg
            for col, strength, gx, gy, rad in glows:
                d = math.hypot(x / sw - gx, (y / sh - gy) * sh / sw) / rad
                k = strength * max(0.0, 1 - d) ** 2
                if k > 0:
                    c = _mix(c, col, k)
            px[x, y] = c
    img = img.resize((W + 320, H + 320), Image.BICUBIC)
    # the faint grid is baked into the field so it drifts with the glow (one crop per frame)
    grid = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(grid)
    line = color("body") + (16,)
    for x in range(0, img.width, GRID_STEP):
        d.line([(x, 0), (x, img.height)], fill=line, width=1)
    for y in range(0, img.height, GRID_STEP):
        d.line([(0, y), (img.width, y)], fill=line, width=1)
    img.paste(grid, (0, 0), grid)
    return img


@lru_cache(maxsize=1)
def _grid_layer() -> Image.Image:
    """RGBA grid (W+64 x H+64), lines at ~6% so it reads as texture, not a pattern."""
    layer = Image.new("RGBA", (W + GRID_STEP, H + GRID_STEP), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    line = color("body") + (16,)
    for x in range(0, W + GRID_STEP, GRID_STEP):
        d.line([(x, 0), (x, H + GRID_STEP)], fill=line, width=1)
    for y in range(0, H + GRID_STEP, GRID_STEP):
        d.line([(0, y), (W + GRID_STEP, y)], fill=line, width=1)
    return layer


class Backdrop:
    """A moving background. `source` is a clip (.mp4/.mov/.webm) or still image path, or None."""

    def __init__(self, source: str | Path | None = None, seed: int = 0, duration: float = 4.0):
        self.seed = int(seed)
        self.duration = max(0.5, float(duration))
        self.kind = "procedural"
        self.frames: list[Image.Image] = []
        self.still: Image.Image | None = None
        rng = random.Random(self.seed * 7919 + 1)
        self.particles = [(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(14, 38), rng.uniform(2, 5),
                           rng.uniform(0.15, 0.4), rng.choice(("cyan", "lime", "body"))) for _ in range(PARTICLES)]
        if source:
            try:
                self._load(Path(source))
            except Exception:  # a bad file just means the procedural background
                self.kind, self.frames, self.still = "procedural", [], None

    # ── footage ─────────────────────────────────────────
    def _load(self, path: Path) -> None:
        if not path.exists():
            return
        if path.suffix.lower() in (".mp4", ".mov", ".webm", ".mkv"):
            sw, sh = SMALL
            cmd = ["ffmpeg", "-loglevel", "error", "-i", str(path), "-t", f"{self.duration + 0.5:.2f}",
                   "-vf", f"fps=15,scale={sw}:{sh}:force_original_aspect_ratio=increase,crop={sw}:{sh},boxblur=4:1",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
            raw = subprocess.run(cmd, capture_output=True, timeout=120).stdout
            n = len(raw) // (sw * sh * 3)
            self.frames = [self._darken(Image.frombytes("RGB", (sw, sh), raw[i * sw * sh * 3:(i + 1) * sw * sh * 3]))
                           for i in range(n)]
            if self.frames:
                self.kind = "footage"
            return
        img = Image.open(path).convert("RGB")
        tw, th = SMALL[0] * 3 // 2, SMALL[1] * 3 // 2  # a little extra for the Ken Burns crop
        scale = max(tw / img.width, th / img.height)
        img = img.resize((max(tw, int(img.width * scale)), max(th, int(img.height * scale))), Image.BILINEAR)
        left, top = (img.width - tw) // 2, (img.height - th) // 2
        img = img.crop((left, top, left + tw, top + th)).filter(ImageFilter.GaussianBlur(5))
        self.still = self._darken(img)
        self.kind = "footage"

    @staticmethod
    def _darken(img: Image.Image) -> Image.Image:
        """Pull footage toward the slate; bright footage is pulled harder so the result stays
        at or under ~0.2 mean luma (a white still can never make a white background)."""
        slate = Image.new("RGB", img.size, color("background"))
        src_l, slate_l = mean_luma(img), mean_luma(slate)
        keep = FOOTAGE_KEEP
        if src_l > slate_l:
            keep = max(0.08, min(FOOTAGE_KEEP, (MAX_FOOTAGE_LUMA - slate_l) / (src_l - slate_l)))
        return Image.blend(slate, img, keep)

    def _footage(self, t: float) -> Image.Image:
        p = min(1.0, max(0.0, t / self.duration))
        zoom = 1.0 + 0.12 * p  # slow push-in
        if self.frames:
            src = self.frames[min(len(self.frames) - 1, int(t * 15)) % len(self.frames)]
        else:
            src = self.still
        sw, sh = src.size
        cw, ch = sw / zoom, sh / zoom
        drift = 0.5 + 0.35 * math.sin(self.seed + p * math.pi)  # slow lateral pan
        x0 = (sw - cw) * drift
        y0 = (sh - ch) * (0.5 - 0.3 * (p - 0.5))
        return src.resize((W, H), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch))

    # ── procedural ──────────────────────────────────────
    def _procedural(self, t: float) -> Image.Image:
        field = _glow_field(self.seed % 4)
        ox = 160 + 150 * math.sin(2 * math.pi * t / 17 + self.seed)
        oy = 160 + 150 * math.cos(2 * math.pi * t / 23 + self.seed * 0.7)
        return field.crop((int(ox), int(oy), int(ox) + W, int(oy) + H))

    def _particles(self, img: Image.Image, t: float, strength: float) -> None:
        d = ImageDraw.Draw(img)
        bg = color("background")
        for x0, y0, speed, r, alpha, col in self.particles:
            y = (y0 - t * speed) % (H + 40) - 20
            x = x0 + 18 * math.sin(t * 0.6 + y0)
            c = _mix(bg, color(col), alpha * strength)
            d.ellipse([x - r, y - r, x + r, y + r], fill=c)

    def frame(self, t: float) -> Image.Image:
        if self.kind == "footage":
            img = self._footage(t)
            self._particles(img, t, 0.6)
        else:
            img = self._procedural(t)
            self._particles(img, t, 1.0)
        return img


def mean_luma(img: Image.Image) -> float:
    """Mean luma 0-1 of an image (Rec. 601 weights via Pillow's L mode)."""
    small = img.convert("L").resize((54, 96), Image.BOX)
    data = list(small.getdata())
    return sum(data) / (255.0 * len(data))
