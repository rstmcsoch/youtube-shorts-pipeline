"""Whisper word-level timestamps + ASS subtitle generation + Pillow fallback."""

from pathlib import Path

from .log import log


def _has_ass_filter() -> bool:
    """Check if ffmpeg has libass (for ASS subtitle burn-in)."""
    import subprocess
    try:
        r = subprocess.run(
            ["ffmpeg", "-filters"],
            capture_output=True, text=True, timeout=5,
        )
        return "ass" in r.stdout
    except Exception:
        return False


def _whisper_word_timestamps(audio_path: Path, lang: str = "en") -> list[dict]:
    """Get word-level timestamps from Whisper.

    Returns list of {"word": str, "start": float, "end": float}.
    """
    try:
        import whisper
    except ImportError:
        log("Whisper not installed — skipping word timestamps")
        return []

    log("Running Whisper for word-level timestamps...")
    model = whisper.load_model("base")
    result = model.transcribe(
        str(audio_path),
        language=lang[:2],
        word_timestamps=True,
    )

    words = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            words.append({
                "word": w["word"].strip(),
                "start": w["start"],
                "end": w["end"],
            })

    log(f"Got {len(words)} word timestamps.")
    return words


def align_words_to_script(words: list[dict], script: str) -> list[dict]:
    """Replace Whisper's word text with the script's spelling, keeping timings.

    Whisper mishears names and jargon ("Gemini 4 Carbon" -> "Gemini for carbon").
    Equal-length differing runs are swapped word-for-word; other runs keep
    Whisper's text but are re-spread over the script words when counts differ
    by little, so captions always show what the script actually says.
    """
    import difflib, re
    if not words or not script:
        return words
    script_words = script.split()
    norm = lambda t: re.sub(r"[^a-z0-9]", "", t.casefold())
    a = [norm(w["word"]) for w in words]
    b = [norm(w) for w in script_words]
    out = []
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            for k in range(i2 - i1):
                out.append({**words[i1 + k], "word": script_words[j1 + k]})
        elif tag == "replace" and i2 > i1 and j2 > j1:
            start, end = words[i1]["start"], words[i2 - 1]["end"]
            n = j2 - j1
            step = (end - start) / n
            for k in range(n):
                out.append({"word": script_words[j1 + k], "start": round(start + k * step, 3),
                            "end": round(start + (k + 1) * step, 3)})
        elif tag == "delete":
            out.extend(words[i1:i2])
        # "insert": script words Whisper never heard (not spoken) are skipped.
    return out


def _group_words(words: list[dict], group_size: int = 4) -> list[list[dict]]:
    groups = []
    for i in range(0, len(words), group_size):
        groups.append(words[i:i + group_size])
    return groups


def group_caption_lines(words: list[dict], max_words: int = 4, min_words: int = 2,
                        max_chars: int = 18, pause: float = 0.35) -> list[list[dict]]:
    """Readable caption lines of 2-5 words.

    Breaks after sentence/clause punctuation, on pauses, or when the line would
    exceed max_words/max_chars; single-word leftovers are merged into the
    neighbouring line when that keeps it within 5 words.
    """
    max_words = min(5, max(2, int(max_words or 4)))
    lines, cur = [], []
    for i, w in enumerate(words):
        if cur:
            text_len = len(" ".join(x["word"] for x in cur + [w]))
            gap = w["start"] - cur[-1]["end"]
            if len(cur) >= max_words or (text_len > max_chars and len(cur) >= min_words) or \
                    (gap > pause and len(cur) >= min_words):
                lines.append(cur)
                cur = []
        cur.append(w)
        if cur and len(cur) >= min_words and str(w["word"]).rstrip()[-1:] in ".!?;:,":
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    merged = []
    for line in lines:
        if len(line) == 1 and merged and len(merged[-1]) < 5 and \
                str(merged[-1][-1]["word"]).rstrip()[-1:] not in ".!?":
            merged[-1] = merged[-1] + line
        else:
            merged.append(line)
    # A leading single word joins the following line if room.
    if len(merged) >= 2 and len(merged[0]) == 1 and len(merged[1]) < 5:
        merged[1] = merged[0] + merged[1]
        merged.pop(0)
    return merged


CAPTION_FONT_CHAIN = ["Montserrat", "Inter", "DejaVu Sans"]


def _installed_font_families() -> set[str]:
    import subprocess
    try:
        r = subprocess.run(["fc-list", ":", "family"], capture_output=True, text=True, timeout=10)
        fams = set()
        for line in r.stdout.splitlines():
            for name in line.split(","):
                fams.add(name.strip().casefold())
        return fams
    except Exception:
        return set()


def resolve_caption_font(preferred: str = "", installed: set[str] | None = None) -> str:
    """Brand caption font: Montserrat → Inter → DejaVu Sans.

    A non-default niche font (e.g. a CJK font for ko/ja) is kept when installed.
    """
    installed = _installed_font_families() if installed is None else installed
    pref = str(preferred or "").strip()
    if pref and pref.casefold() not in {"arial", "montserrat", "inter", "dejavu sans"} and pref.casefold() in installed:
        return pref
    for name in CAPTION_FONT_CHAIN:
        if name.casefold() in installed:
            return name
    return pref if pref and pref.casefold() != "arial" else "DejaVu Sans"


def _ass_escape(text: str) -> str:
    return str(text).replace("\\", "").replace("{", "(").replace("}", ")")


def _format_ass_time(seconds: float) -> str:
    """Format seconds to ASS timestamp: H:MM:SS.cc (centiseconds)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _generate_ass(
    words: list[dict],
    output_path: Path,
    video_width: int = 1080,
    video_height: int = 1920,
    highlight_color: str = "#FFFF00",
    group_size: int = 4,
    font_family: str = "Arial",
    font_size: int = 92,
    smart_lines: bool = True,
):
    """Generate ASS subtitles with active-word highlighting.

    Style: bold brand font (Montserrat → Inter → DejaVu Sans Bold), white text,
    thick black outline + soft shadow, centred ~20% from the bottom, 2-5 words
    per line, each line fading in/out over 80 ms. One Dialogue event per word
    (the active word is coloured and slightly enlarged); events within a line
    are contiguous so the line does not flicker between words.
    """
    font = resolve_caption_font(font_family)
    margin_v = int(video_height * 0.20)
    outline = max(4, int(round(font_size * 0.085)))
    shadow = max(2, int(round(font_size * 0.04)))
    header = f"""[Script Info]
Title: Pipeline Captions
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,1,0,1,{outline},{shadow},2,60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    hc = highlight_color.lstrip("#")
    if len(hc) == 6:
        ass_highlight = f"&H{hc[4:6]}{hc[2:4]}{hc[0:2]}&"
    else:
        ass_highlight = "&H00FFFF&"

    if smart_lines:
        groups = group_caption_lines(words, max_words=group_size)
    else:
        groups = _group_words(words, group_size=group_size)
    events = []
    for gi, group in enumerate(groups):
        if not group:
            continue
        next_start = groups[gi + 1][0]["start"] if gi + 1 < len(groups) and groups[gi + 1] else None
        line_end = group[-1]["end"] + 0.25
        if next_start is not None:
            line_end = min(line_end, next_start)
        line_end = max(line_end, group[-1]["end"])
        for active_idx, active_word in enumerate(group):
            start = active_word["start"]
            end = group[active_idx + 1]["start"] if active_idx + 1 < len(group) else line_end
            end = max(end, start + 0.05)
            fade_in = 80 if active_idx == 0 else 0
            fade_out = 80 if active_idx == len(group) - 1 else 0
            parts = []
            for j, w in enumerate(group):
                word = _ass_escape(w["word"])
                if j == active_idx:
                    parts.append(f"{{\\c{ass_highlight}\\fscx110\\fscy110}}{word}{{\\r}}")
                else:
                    parts.append(word)
            text = f"{{\\fad({fade_in},{fade_out})}}" + " ".join(parts)
            events.append(
                f"Dialogue: 0,{_format_ass_time(start)},{_format_ass_time(end)},Default,,0,0,0,,{text}"
            )

    output_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    log(f"ASS captions saved: {output_path.name} (font: {font}, {len(groups)} lines)")
    return output_path


# ─────────────────────────────────────────────────────
# DYPOL karaoke captions (Engine B style guide)
# ─────────────────────────────────────────────────────

DYPOL_CAPTION_LANGS = {"en", "es", "pt", "de", "fr"}   # Space Grotesk covers Latin scripts only
DYPOL_CAPTION = {
    "font": "Space Grotesk",
    "size": 112,              # libass size (line height); glyph em is ~size / 1.28
    "min_size": 76,
    "center_x": 500,          # centre of the safe band x 72-929 (right 14% is Shorts UI)
    "center_y_frac": 0.63,    # ~63% of screen height
    "max_width": 790,         # px at PlayRes, incl. outline; band is 857 px wide
    "outline": 7,
    "shadow": 4,
}


def _ass_colour(hex_rgb: str, alpha: int = 0) -> str:
    h = hex_rgb.lstrip("#")
    return f"&H{alpha:02X}{h[4:6].upper()}{h[2:4].upper()}{h[0:2].upper()}"


def _ass_inline_colour(hex_rgb: str) -> str:
    h = hex_rgb.lstrip("#").upper()
    return f"&H{h[4:6]}{h[2:4]}{h[0:2]}&"


def dypol_caption_chunks(words: list[dict], max_words: int = 4, max_chars: int = 20,
                         pause: float = 0.35) -> list[list[dict]]:
    """2-4 word chunks that never run across a sentence end.

    Breaks after sentence punctuation (always), after a comma/colon once 2+ words
    are in, on a pause, or when the chunk would pass max_words/max_chars. A lone
    word then joins the chunk before it (or after it) when that keeps 4 words or
    fewer and no sentence end is crossed, else borrows the previous chunk's last word.
    """
    def ends(w, marks):
        return str(w["word"]).rstrip()[-1:] in marks
    chunks, cur = [], []
    for w in words:
        if cur:
            text_len = len(" ".join(str(x["word"]) for x in cur + [w]))
            gap = float(w["start"]) - float(cur[-1]["end"])
            if len(cur) >= max_words or (text_len > max_chars and len(cur) >= 2) or (gap > pause and len(cur) >= 2):
                chunks.append(cur)
                cur = []
        cur.append(w)
        if ends(w, ".!?") or (ends(w, ",;:") and len(cur) >= 2):
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    out = []
    i = 0
    while i < len(chunks):
        c = chunks[i]
        if len(c) == 1:
            if out and len(out[-1]) < max_words and not ends(out[-1][-1], ".!?"):
                out[-1] = out[-1] + c
                i += 1
                continue
            if i + 1 < len(chunks) and len(chunks[i + 1]) < max_words and not ends(c[0], ".!?"):
                chunks[i + 1] = c + chunks[i + 1]
                i += 1
                continue
            if out and len(out[-1]) >= 3 and not ends(out[-1][-1], ".!?"):
                out[-1], c = out[-1][:-1], [out[-1][-1]] + c   # borrow one word: 3 + 2
        out.append(c)
        i += 1
    return out


def _caption_font_path():
    from .style import font_path
    return font_path("display_bold")


def _chunk_font_size(text: str, size: int, max_width: int, min_size: int, outline: int) -> int:
    """Largest size <= `size` whose rendered width (Space Grotesk Bold) fits max_width."""
    try:
        from PIL import ImageFont
        path = str(_caption_font_path())
        while size > min_size:
            f = ImageFont.truetype(path, max(8, int(size / 1.28)))
            if f.getlength(text) + 2 * outline <= max_width:
                break
            size -= 4
        return size
    except Exception:
        # ~0.6 em per character for a bold grotesk
        while size > min_size and len(text) * size * 0.47 + 2 * outline > max_width:
            size -= 4
        return size


def generate_dypol_ass(words: list[dict], output_path: Path, video_width: int = 1080,
                       video_height: int = 1920) -> Path:
    """Word-level ASS karaoke in the DYPOL look.

    2-4 words per chunk, Space Grotesk Bold (vendored in style/fonts; the ass filter
    is given that fontsdir), off-white with a slate stroke and soft shadow; the
    word being spoken turns lime. Chunks sit centred at ~63% of the height inside
    x 100-900, so they never enter the bottom 22% or right 14% Shorts UI. A chunk
    too wide for the band is set smaller rather than wrapped. One Dialogue event per
    word; events in a chunk are contiguous, so the line never flickers.
    """
    from .style import load_style
    cfg = DYPOL_CAPTION
    pal = load_style()["look"]["palette"]
    white = _ass_colour(pal["body"])
    lime = _ass_colour(pal["lime"])
    stroke = _ass_colour(pal["background"])
    shadow = _ass_colour("#000000", 0x70)
    lime_i, white_i = _ass_inline_colour(pal["lime"]), _ass_inline_colour(pal["body"])
    cx, cy = cfg["center_x"], int(round(video_height * cfg["center_y_frac"]))
    header = f"""[Script Info]
Title: DYPOL Captions
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Dypol,{cfg['font']},{cfg['size']},{white},{lime},{stroke},{shadow},-1,0,0,0,100,100,0,0,1,{cfg['outline']},{cfg['shadow']},5,100,180,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    chunks = dypol_caption_chunks(words)
    events = []
    for gi, chunk in enumerate(chunks):
        if not chunk:
            continue
        next_start = chunks[gi + 1][0]["start"] if gi + 1 < len(chunks) and chunks[gi + 1] else None
        line_end = chunk[-1]["end"] + 0.25
        if next_start is not None:
            line_end = min(line_end, next_start)
        line_end = max(line_end, chunk[-1]["end"])
        plain = " ".join(str(w["word"]) for w in chunk)
        fs = _chunk_font_size(plain, cfg["size"], cfg["max_width"], cfg["min_size"], cfg["outline"])
        for ai, active in enumerate(chunk):
            start = active["start"]
            end = chunk[ai + 1]["start"] if ai + 1 < len(chunk) else line_end
            end = max(end, start + 0.05)
            lead = f"{{\\an5\\pos({cx},{cy})" + (f"\\fs{fs}" if fs != cfg["size"] else "")
            if ai == 0:   # chunk enters with a small 90 ms pop
                lead += "\\fad(60,0)\\fscx92\\fscy92\\t(0,90,\\fscx100\\fscy100)"
            elif ai == len(chunk) - 1:
                lead += "\\fad(0,60)"
            lead += "}"
            parts = []
            for j, w in enumerate(chunk):
                word = _ass_escape(w["word"])
                parts.append(f"{{\\c{lime_i}}}{word}{{\\c{white_i}}}" if j == ai else word)
            events.append(f"Dialogue: 0,{_format_ass_time(start)},{_format_ass_time(end)},Dypol,,0,0,0,,"
                          + lead + " ".join(parts))
    Path(output_path).write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    log(f"DYPOL ASS captions saved: {Path(output_path).name} ({len(chunks)} chunks)")
    return Path(output_path)


def use_dypol_captions(lang: str) -> bool:
    import os
    if os.environ.get("DYPOL_CAPTION_STYLE", "dypol").strip().lower() == "legacy":
        return False
    return (lang or "en")[:2].lower() in DYPOL_CAPTION_LANGS


def _generate_srt(words: list[dict], output_path: Path, group_size: int = 4, smart_lines: bool = False) -> Path:
    """Generate standard SRT file from word timestamps."""
    groups = group_caption_lines(words, max_words=group_size) if smart_lines else _group_words(words, group_size=group_size)
    lines = []

    for i, group in enumerate(groups, 1):
        if not group:
            continue
        start = group[0]["start"]
        end = group[-1]["end"]
        text = " ".join(w["word"] for w in group)

        start_ts = _srt_time(start)
        end_ts = _srt_time(end)
        lines.append(f"{i}\n{start_ts} --> {end_ts}\n{text}\n")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"SRT captions saved: {output_path.name}")
    return output_path


def _srt_time(seconds: float) -> str:
    """Format seconds to SRT timestamp: HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def generate_captions(
    audio_path: Path,
    work_dir: Path,
    lang: str = "en",
    highlight_color: str = "#FFFF00",
    words_per_group: int = 4,
    font_family: str = "Arial",
    font_size: int = 92,
    script: str = "",
) -> dict:
    """Generate captions: ASS (for burn-in) + SRT (for YouTube upload).

    Args:
        font_family: ASS Style font name. Use a CJK-capable font (e.g.
            "Noto Sans CJK KR", "Noto Sans CJK JP") for non-Latin languages,
            otherwise glyphs render as boxes. Pulled from the niche profile's
            captions.font_family field.
        font_size: ASS Style font size. Pulled from the niche profile's
            captions.font_size field.

    Returns dict with keys: srt_path, ass_path, words (for music ducking).
    """
    words = _whisper_word_timestamps(audio_path, lang)
    if script:
        words = align_words_to_script(words, script)

    result = {"words": words}

    if not words:
        log("No word timestamps — skipping caption generation")
        # Fallback: run whisper CLI for SRT only
        try:
            from .config import run_cmd
            run_cmd([
                "whisper", str(audio_path),
                "--model", "base",
                "--language", lang[:2],
                "--output_format", "srt",
                "--output_dir", str(work_dir),
            ], capture=True)
            candidates = list(work_dir.glob("*.srt"))
            if candidates:
                srt = candidates[0]
                final = audio_path.with_suffix(".srt")
                srt.rename(final)
                result["srt_path"] = str(final)
        except Exception as e:
            log(f"Whisper CLI fallback failed: {e}")
        return result

    # Generate SRT
    srt_path = work_dir / f"captions_{lang}.srt"
    _generate_srt(words, srt_path, group_size=words_per_group, smart_lines=True)
    result["srt_path"] = str(srt_path)

    # Generate ASS for burn-in (niche-aware highlight color)
    ass_path = work_dir / f"captions_{lang}.ass"
    if use_dypol_captions(lang):
        generate_dypol_ass(words, ass_path)
    else:
        _generate_ass(
            words, ass_path,
            highlight_color=highlight_color,
            group_size=words_per_group,
            font_family=font_family,
            font_size=font_size,
        )
    result["ass_path"] = str(ass_path)

    return result
