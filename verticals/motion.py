"""CPU motion blocks for Engine B: Pillow frames piped to ffmpeg (1080x1920, 30 fps).

Renderers: kinetic_text, count_up, code_panel, diagram. Each draws a frame for a
time t (render_frame) and render_block() streams every frame to ffmpeg with the
same encode settings as the still/clip shots, so the assembly concat stays a
stream copy. All motion is eased; something keeps moving for the whole block so
no frame sits static past the style guide's limit.

Layout keeps clear of the Shorts UI (bottom 22%, right 14%) and of the burned-in
caption line (~63% of height): content lives in x 72-930, y 170-1080.
"""

import math
import subprocess
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .style import color, font_path, screen

W, H, FPS = screen()
LEFT, RIGHT, TOP, BOTTOM = 72, 930, 170, 1080
CX = (LEFT + RIGHT) // 2
MOTION_KINDS = ("kinetic_text", "count_up", "code_panel", "diagram")


# ── easing ────────────────────────────────────────────
def clamp(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def ease_out_cubic(x: float) -> float:
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_in_out_cubic(x: float) -> float:
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_out_back(x: float, s: float = 1.6) -> float:
    x = clamp(x)
    return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2


def progress(t: float, start: float, length: float) -> float:
    return clamp((t - start) / length) if length > 0 else float(t >= start)


@lru_cache(maxsize=64)
def font(role: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(font_path(role)), size)


def mix(a, b, k: float):
    return tuple(int(round(a[i] + (b[i] - a[i]) * k)) for i in range(3))


def text_w(draw: ImageDraw.ImageDraw, text: str, f) -> int:
    box = draw.textbbox((0, 0), text, font=f)
    return box[2] - box[0]


def fit_size(text_list, role: str, start: int, max_w: int, min_size: int = 40) -> int:
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    size = start
    while size > min_size and max(text_w(probe, t, font(role, size)) for t in text_list) > max_w:
        size -= 4
    return size


def base_canvas(grid: bool = False) -> Image.Image:
    return _base_template(grid).copy()


@lru_cache(maxsize=2)
def _base_template(grid: bool) -> Image.Image:
    img = Image.new("RGB", (W, H), color("background"))
    if grid:  # very faint 64 px line grid (4%), diagram blocks only
        d = ImageDraw.Draw(img)
        line = mix(color("background"), color("body"), 0.04)
        for x in range(0, W, 64):
            d.line([(x, 0), (x, H)], fill=line, width=1)
        for y in range(0, H, 64):
            d.line([(0, y), (W, y)], fill=line, width=1)
    return img


def draw_tag(img: Image.Image, text: str, y: int, alpha: float = 1.0) -> None:
    """Left-edge 6 px lime bar + mono uppercase label."""
    if not text or alpha <= 0:
        return
    d = ImageDraw.Draw(img)
    bg = color("background")
    f = font("mono_bold", 32)
    d.rectangle([LEFT, y, LEFT + 5, y + 40], fill=mix(bg, color("lime"), alpha))
    d.text((LEFT + 24, y + 20), text.upper()[:28], font=f, fill=mix(bg, color("body"), alpha), anchor="lm")


def paste_scaled(img: Image.Image, sprite: Image.Image, cx: float, cy: float, scale: float, alpha: float,
                 anchor_left: float | None = None) -> None:
    if alpha <= 0.01 or scale <= 0.01:
        return
    w, h = max(1, int(sprite.width * scale)), max(1, int(sprite.height * scale))
    s = sprite.resize((w, h), Image.BICUBIC) if (w, h) != sprite.size else sprite.copy()
    if alpha < 1:
        a = s.getchannel("A").point(lambda v: int(v * alpha))
        s.putalpha(a)
    x = int(anchor_left if anchor_left is not None else cx - w / 2)
    img.paste(s, (x, int(cy - h / 2)), s)


def text_sprite(text: str, f, fill) -> Image.Image:
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    l, t, r, b = probe.textbbox((0, 0), text, font=f)
    pad = 8
    spr = Image.new("RGBA", (r - l + 2 * pad, b - t + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(spr).text((pad - l, pad - t), text, font=f, fill=fill)
    return spr


# The slow 3% push on text blocks is decoration: it is not a "change" a viewer sees.
# longest_static() switches it off so only real content changes count.
_DRIFT = 1.0


# ── kinetic_text ─────────────────────────────────────
def frame_kinetic_text(data: dict, t: float, duration: float) -> Image.Image:
    words = data["words"][:4]
    accent = data.get("accent") or words[-1]
    size = fit_size(words, "display_bold", 190 if len(words) <= 2 else 168, int((RIGHT - LEFT - 20) / 1.04))
    f = font("display_bold", size)
    line_h = int(size * 1.08)
    block_h = line_h * len(words)
    y0 = (TOP + BOTTOM) / 2 - block_h / 2 + line_h / 2
    img = base_canvas()
    drift = 1 + 0.035 * ease_in_out_cubic(t / max(duration, 0.1)) * _DRIFT  # slow push
    for i, word in enumerate(words):
        start = 0.10 + i * 0.26
        p = progress(t, start, 0.38)
        is_accent = word == accent
        spr = _sprite_cache(word, size, "lime" if is_accent else "body")
        cy = y0 + i * line_h
        if is_accent:  # pop: overshoot scale
            scale = (0.55 + 0.45 * ease_out_back(p)) * drift
            paste_scaled(img, spr, 0, cy, scale, ease_out_cubic(p * 1.6),
                         anchor_left=LEFT + (spr.width * (1 - scale / drift)) / 2)
        else:  # slide in from the left
            off = (1 - ease_out_cubic(p)) * -140
            paste_scaled(img, spr, 0, cy, drift, ease_out_cubic(p), anchor_left=LEFT + off)
    return img


@lru_cache(maxsize=32)
def _sprite_cache(word: str, size: int, colour: str) -> Image.Image:
    return text_sprite(word, font("display_bold", size), color(colour) + (255,))


# ── count_up ─────────────────────────────────────────
def _fmt_number(value: float, decimals: int) -> str:
    return f"{value:,.{decimals}f}"


def frame_count_up(data: dict, t: float, duration: float) -> Image.Image:
    img = base_canvas()
    d = ImageDraw.Draw(img)
    bg, body, lime = color("background"), color("body"), color("lime")
    target, dec = float(data["value"]), int(data.get("decimals", 0))
    pre, suf = data.get("prefix", ""), data.get("suffix", "")
    final_text = f"{pre}{_fmt_number(target, dec)}{suf}"
    size = fit_size([final_text], "display_bold", 260, int((RIGHT - LEFT - 10) / 1.035), 100)
    count_len = min(1.0, max(0.6, duration * 0.35))
    settled = bool(data.get("settled"))
    p = 1.0 if settled else progress(t, 0.15, count_len)
    k = ease_out_cubic(p)
    text = f"{pre}{_fmt_number(target * k, dec)}{suf}"
    done_flash = 1.0 if settled else progress(t, 0.15 + count_len, 0.25)
    num_color = mix(body, lime, done_flash)
    drift = 1 + 0.03 * ease_in_out_cubic(t / max(duration, 0.1)) * _DRIFT
    f = font("display_bold", int(size * drift))
    cy = 560
    draw_tag(img, "BY THE NUMBERS", TOP + 10, progress(t, 0.0, 0.25))
    d.text((LEFT, cy), text, font=f, fill=num_color, anchor="lm")
    # meter that fills with the count
    full_w = text_w(d, final_text, font("display_bold", size))
    my = cy + size * 0.62
    d.rounded_rectangle([LEFT, my, LEFT + full_w, my + 14], radius=7, fill=color("border"))
    if k > 0.01:
        d.rounded_rectangle([LEFT, my, LEFT + max(14, full_w * k), my + 14], radius=7, fill=lime)
    label = data.get("label", "")
    if label:
        lp = 1.0 if settled else ease_out_cubic(progress(t, 0.45, 0.35))
        lf = font("display_medium", fit_size([label], "display_medium", 64, RIGHT - LEFT, 36))
        d.text((LEFT + (1 - lp) * -60, my + 70), label, font=lf, fill=mix(bg, body, lp), anchor="la")
    note = data.get("note", "")
    if note:  # follow-up card: the narration's key words land under the settled number
        np_ = ease_out_back(progress(t, 0.1, 0.4), 1.3)
        nf = font("display_bold", fit_size([note], "display_bold", 84, RIGHT - LEFT, 44))
        d.text((LEFT + (1 - np_) * 90, my + 190), note, font=nf, fill=mix(bg, color("amber"), clamp(np_)), anchor="la")
    source = data.get("source", "")
    if source:
        sp = 1.0 if settled else ease_out_cubic(progress(t, 0.8, 0.35))
        sf = font("mono", fit_size([source], "mono", 30, RIGHT - LEFT, 22))
        d.text((LEFT, BOTTOM - 40), source, font=sf, fill=mix(bg, color("muted"), sp), anchor="la")
    return img


# ── code_panel ───────────────────────────────────────
def _token_color(ttype) -> tuple:
    from pygments import token as T
    if ttype in T.Comment:
        return color("muted")
    if ttype in T.Keyword or ttype in T.Name.Builtin or ttype in T.Name.Function:
        return color("cyan")
    if ttype in T.String:
        return color("lime")
    if ttype in T.Number or ttype in T.Literal:
        return color("amber")
    if ttype in T.Generic.Prompt:
        return color("lime")
    return color("body")


@lru_cache(maxsize=16)
def _highlight(code: str, language: str) -> tuple:
    """Per line: tuple of (text, rgb) runs."""
    from pygments.lexers import get_lexer_by_name
    from pygments.util import ClassNotFound
    try:
        lexer = get_lexer_by_name(language, stripnl=False)
    except ClassNotFound:
        lexer = get_lexer_by_name("text", stripnl=False)
    lines, cur = [], []
    for ttype, value in lexer.get_tokens(code):
        parts = value.split("\n")
        for j, part in enumerate(parts):
            if j > 0:
                lines.append(tuple(cur))
                cur = []
            if part:
                cur.append((part, _token_color(ttype)))
    if cur:
        lines.append(tuple(cur))
    return tuple(lines)


def frame_code_panel(data: dict, t: float, duration: float) -> Image.Image:
    lines = data["lines"]
    lang = data.get("language", "bash")
    shell = lang in {"bash", "sh", "shell", "console", "zsh"}
    shown = [(l[2:] if shell and l.startswith("$ ") else l) for l in lines]
    prompts = [shell and l.startswith("$ ") for l in lines]
    runs = _highlight("\n".join(shown), "bash" if shell else lang)
    runs = list(runs) + [()] * (len(lines) - len(runs))

    fsz = 38
    f = font("mono", fsz)
    cw = f.getlength("M")
    line_h = 60
    pad = 30
    bar_h = 66
    panel_w = RIGHT - LEFT - 10
    panel_h = bar_h + pad * 2 + line_h * len(lines)
    py = (TOP + BOTTOM) / 2 - panel_h / 2

    enter = ease_out_cubic(progress(t, 0.0, 0.3))
    py += (1 - enter) * 50
    img = base_canvas()
    d = ImageDraw.Draw(img)
    bg, panel, border = color("background"), color("panel"), color("border")
    x0, y0, x1, y1 = LEFT, py, LEFT + panel_w, py + panel_h
    d.rounded_rectangle([x0, y0, x1, y1], radius=14, fill=mix(bg, panel, enter), outline=mix(bg, border, enter))
    d.line([(x0 + 1, y0 + bar_h), (x1 - 1, y0 + bar_h)], fill=mix(bg, border, enter), width=1)
    for i, c in enumerate(("border", "border", "border")):
        d.ellipse([x0 + 26 + i * 30, y0 + bar_h / 2 - 8, x0 + 42 + i * 30, y0 + bar_h / 2 + 8],
                  fill=mix(bg, color("muted"), 0.35 * enter))
    d.text((x1 - 26, y0 + bar_h / 2), data.get("title", "terminal"), font=font("mono", 28),
           fill=mix(bg, color("muted"), enter), anchor="rm")

    # typing schedule: paced so typing ends by ~70% of the block (6-32 chars/s)
    prompt_w = 2 if any(prompts) else 0
    total = sum(len(s) for s in shown) or 1
    t0 = 0.3
    cps = _typing_cps(total, duration)
    typed = int(max(0.0, t - t0) * cps)
    done = typed >= total
    highlight_i = int(data.get("highlight", len(lines) - 1)) % len(lines)
    typed_end = t0 + total / cps
    if done and t > typed_end + 0.5 + PULSE_S:  # then the highlight walks the lines
        highlight_i = (highlight_i + 1 + int((t - typed_end - 0.5 - PULSE_S) / PULSE_S)) % len(lines)
    if done:  # highlight bar sweeps behind the key line
        hp = ease_out_cubic(progress(t, t0 + total / cps + 0.15, 0.35))
        hy = y0 + bar_h + pad + highlight_i * line_h
        if hp > 0:
            d.rectangle([x0 + 12, hy - 4, x0 + 12 + (panel_w - 24) * hp, hy + line_h - 8],
                        fill=mix(panel, color("lime"), 0.16))
    remaining = typed
    cursor = None
    for i, line in enumerate(runs):
        ly = y0 + bar_h + pad + i * line_h
        x = x0 + pad
        if remaining <= 0 and i > 0:
            break
        if prompts[i]:
            d.text((x, ly), "$", font=font("mono_bold", fsz), fill=color("lime"))
        x += prompt_w * cw
        n = min(remaining, len(shown[i]))
        left = n
        for text, col in line:
            if left <= 0:
                break
            piece = text[:left]
            d.text((x, ly), piece, font=f, fill=col)
            x += f.getlength(piece)
            left -= len(piece)
        cursor = (x, ly)
        remaining -= len(shown[i])
        if remaining < 0:
            break
    if cursor is not None and (not done or int(t * 2.2) % 2 == 0):
        cx, cy = cursor
        d.rectangle([cx + 3, cy + 4, cx + 3 + cw * 0.9, cy + fsz + 4], fill=color("lime"))
    return img


# ── diagram ──────────────────────────────────────────
PULSE_S = 2.2  # after the last node lands, the active highlight moves every 2.2 s


def diagram_step(n: int, duration: float) -> float:
    """Seconds between node reveals: spread over ~80% of the block (one node every <= 2.8 s),
    so a long diagram beat reveals progressively instead of landing in 2 s and holding."""
    return min(2.8, max(0.4, (duration * 0.8 - 0.2) / max(1, n)))


def frame_diagram(data: dict, t: float, duration: float) -> Image.Image:
    nodes = data["nodes"][:4]
    n = len(nodes)
    img = base_canvas(grid=True)
    d = ImageDraw.Draw(img)
    bg, panel, border, body = color("background"), color("panel"), color("border"), color("body")
    title = data.get("title", "")
    top = TOP + (80 if title else 20)
    box_h, gap = 132, 92
    total_h = n * box_h + (n - 1) * gap
    top = max(top, (TOP + BOTTOM) / 2 - total_h / 2 + (40 if title else 0))
    box_w = RIGHT - LEFT - 60
    bx0 = LEFT + 10
    draw_tag(img, title, TOP + 10, progress(t, 0.0, 0.25))

    step = diagram_step(n, duration)
    size = fit_size(nodes, "display_medium", 58, box_w - 80, 34)
    f = font("display_medium", size)
    active = min(n - 1, int(max(0.0, t - 0.15) / step))
    revealed = 0.15 + (n - 1) * step + 0.32
    if t > revealed + PULSE_S:  # all steps shown: the highlight walks the flow again
        active = int((t - revealed - PULSE_S) / PULSE_S) % n  # first pulse moves from the last node to node 1
    for i, label in enumerate(nodes):
        start = 0.15 + i * step
        p = ease_out_back(progress(t, start, 0.32), 1.2)
        a = ease_out_cubic(progress(t, start, 0.25))
        if a <= 0:
            continue
        y = top + i * (box_h + gap)
        # arrow from the previous node grows down over 0.3 s, just before this node lands
        if i > 0:
            ap = ease_in_out_cubic(progress(t, start - 0.28, 0.28))
            ax = bx0 + box_w / 2
            ay0 = y - gap + 10
            ay1 = ay0 + (gap - 20) * ap
            d.line([(ax, ay0), (ax, ay1)], fill=color("muted"), width=6)
            if ap > 0.85:
                d.polygon([(ax - 16, ay1 - 18), (ax + 16, ay1 - 18), (ax, ay1 + 2)], fill=color("muted"))
        is_last = i == n - 1
        if i == active:
            edge = color("lime") if is_last else color("cyan")
            width = 4
        else:
            edge, width = border, 2
        grow = 0.9 + 0.1 * p
        w2, h2 = box_w * grow / 2, box_h * grow / 2
        cx, cy = bx0 + box_w / 2, y + box_h / 2
        fill = mix(bg, panel, a)
        if i == active:
            fill = mix(fill, edge, 0.22 * a)
        if i == active:  # soft glow around the active step
            glow = mix(bg, edge, 0.18 * a)
            d.rounded_rectangle([cx - w2 - 10, cy - h2 - 10, cx + w2 + 10, cy + h2 + 10], radius=22, fill=glow)
        d.rounded_rectangle([cx - w2, cy - h2, cx + w2, cy + h2], radius=14, fill=fill,
                            outline=mix(bg, edge, a), width=width)
        d.text((cx, cy), label, font=f, fill=mix(bg, body, a), anchor="mm")
        # step number in mono
        d.text((cx - w2 + 24, cy - h2 + 16), f"0{i + 1}", font=font("mono", 26),
               fill=mix(bg, color("muted"), a))
    return img


def code_typing_window(data: dict, duration: float) -> tuple[float, float]:
    """(start, end) seconds of the code panel's typing, same schedule as frame_code_panel."""
    lines = data.get("lines") or []
    lang = data.get("language", "bash")
    shell = lang in {"bash", "sh", "shell", "console", "zsh"}
    shown = [(l[2:] if shell and l.startswith("$ ") else l) for l in lines]
    total = sum(len(s) for s in shown) or 1
    t0 = 0.3
    cps = _typing_cps(total, duration)
    return t0, min(duration, t0 + total / cps)


def _typing_cps(total_chars: int, duration: float) -> float:
    return min(32.0, max(6.0, total_chars / max(0.5, duration * 0.7 - 0.3)))


RENDERERS = {
    "kinetic_text": frame_kinetic_text,
    "count_up": frame_count_up,
    "code_panel": frame_code_panel,
    "diagram": frame_diagram,
}


def render_frame(kind: str, data: dict, t: float, duration: float) -> Image.Image:
    return RENDERERS[kind](data, t, duration)


def render_block(kind: str, data: dict, out_path: Path, duration: float, fps: int = FPS,
                 t_offset: float = 0.0, total: float | None = None) -> Path:
    """Render one motion block to an H.264 mp4 (same settings as the other shots).

    `t_offset`/`total` render a window [t_offset, t_offset + duration] of a block whose
    animation spans `total` seconds (a sub-shot of a continuous block)."""
    out_path = Path(out_path)
    total = duration if total is None else total
    n_frames = max(1, int(round(duration * fps)))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-frames:v", str(n_frames), "-r", str(fps),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for i in range(n_frames):
            proc.stdin.write(render_frame(kind, data, t_offset + i / fps, total).tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    err = proc.stderr.read().decode(errors="replace")
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed rendering {kind}: {err[-400:]}")
    return out_path


# ── sub-shot cards: nothing on screen holds past the 3.5 s static limit ──────
# A long motion beat (7-9 s in a 58 s Short with 8 beats) used to be rendered once and
# then re-cut with an alternating 0.92x pull-back, so a 2-3 word kinetic card (or a
# finished count/diagram) sat unchanged for 8-9 s. Now every sub-shot gets new content:
# kinetic_text follows the narration with a fresh card per sub-shot, count_up keeps the
# number and lands the narration's key words under it, diagram and code_panel animate
# continuously across the whole beat (progressive reveal / typing).
STOPWORDS = set("""a an the and or but so of to in on at for with by from as into onto than then that this these
those it its it's is are was were be been being do does did can could will would should may might must
not no yes your you you're we our they their them he she his her i me my mine about over under more most
less very just also only such which who whom what when where how why there here have has had if because
while each every any some all both either neither into out up down off again once same other own too
let's lets get gets got make makes made way ways thing things one anything something everything nothing
look looks like really actually even still much many lot lots well back going goes into whether""".split())


def key_words(words: list, limit: int = 3) -> list[str]:
    """An on-screen phrase from a narration window: the best run of consecutive content
    words ("traditional security measures"), so the card reads as a phrase, not a word salad.
    Runs with a number win, then longer runs, then longer words."""
    runs, cur = [], []
    for w in words:
        raw = str(w.get("word", "") if isinstance(w, dict) else w)
        tok = raw.strip(" .,!?;:\"'()[]—–-")
        content = tok and tok.lower() not in STOPWORDS and (len(tok) >= 3 or any(c.isdigit() for c in tok))
        if content:
            cur.append(tok)
        if not content or raw.rstrip()[-1:] in ".,!?;:—–":  # a stopword or a clause end breaks the run
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    best, best_key = [], None
    for run in runs:
        for a in range(len(run)):
            seg = run[a:a + limit]
            key = (any(ch.isdigit() for t in seg for ch in t), len(seg), sum(len(t) for t in seg))
            if best_key is None or key > best_key:
                best, best_key = seg, key
    return best


def _window(words: list | None, start: float, end: float) -> list:
    out = []
    for w in words or []:
        try:
            mid = (float(w["start"]) + float(w["end"])) / 2
        except (KeyError, TypeError, ValueError):
            continue
        if start <= mid < end:
            out.append(w)
    return out


def motion_cards(kind: str, data: dict, cuts: list[float], shot_start: float = 0.0,
                 words: list | None = None) -> list[dict]:
    """One render spec per sub-shot: {kind, data, duration, t_offset, total}.

    Continuous kinds (diagram, code_panel) are one block timed across the whole beat;
    kinetic_text and count_up get a new card per sub-shot built from the narration words
    spoken during it.
    """
    cuts = [float(c) for c in (cuts or [])] or [0.0]
    total = sum(cuts)
    if kind in ("diagram", "code_panel") or len(cuts) == 1:
        specs, t = [], 0.0
        for c in cuts:
            specs.append({"kind": kind, "data": data, "duration": c, "t_offset": t, "total": total})
            t += c
        return specs
    specs, t, prev = [], 0.0, None
    for j, c in enumerate(cuts):
        spoken = key_words(_window(words, shot_start + t, shot_start + t + c))
        if j == 0:
            card = dict(data)
        elif kind == "kinetic_text":
            words_ = spoken or [w for w in data.get("words", []) if w != (prev or {}).get("accent")] or data["words"]
            accent = next((w for w in words_ if any(ch.isdigit() for ch in w)), max(words_, key=len))
            card = {"words": words_[:4], "accent": accent}
            if prev and [w.lower() for w in card["words"]] == [w.lower() for w in prev["words"]]:
                card["words"] = list(reversed(card["words"]))
        else:  # count_up: the number stays, the narration's key words land under it
            note = " ".join(spoken[:3]) or " ".join(data.get("label", "").split()[:3])
            card = {**data, "settled": True, "note": note[:28]}
        specs.append({"kind": kind, "data": card, "duration": c, "t_offset": 0.0, "total": c})
        prev = card
        t += c
    return specs


def _thumb(img: Image.Image) -> list[int]:
    # 40x40 px area averages: a slow 3% push-in barely moves them, new words or a new box do
    return list(img.convert("L").resize((27, 48), Image.BOX).getdata())


STATIC_DIFF = 0.5  # % of 27x48 cells whose grey level moved >= PIXEL_DELTA = "the screen changed"
PIXEL_DELTA = 6


def longest_static(specs: list[dict], sample_fps: int = 10, threshold: float = STATIC_DIFF) -> float:
    """Longest stretch (s) in which the rendered picture stays within `threshold` of the
    frame that opened the stretch. Slow drifts count as static; a new card or a node
    reveal does not."""
    global _DRIFT
    anchor, run, best = None, 0.0, 0.0
    step = 1.0 / sample_fps
    _DRIFT, saved = 0.0, _DRIFT
    try:
        best = _longest_static(specs, sample_fps, threshold)
    finally:
        _DRIFT = saved
    return best


def _longest_static(specs: list[dict], sample_fps: int, threshold: float) -> float:
    anchor, run, best = None, 0.0, 0.0
    step = 1.0 / sample_fps
    for spec in specs:
        n = max(1, int(round(spec["duration"] * sample_fps)))
        for i in range(n):
            th = _thumb(render_frame(spec["kind"], spec["data"], spec["t_offset"] + i * step, spec["total"]))
            if anchor is not None and 100.0 * sum(abs(a - b) >= PIXEL_DELTA for a, b in zip(anchor, th)) / len(th) < threshold:
                run += step
            else:
                anchor, run = th, step
            best = max(best, run)
    return round(best, 2)
