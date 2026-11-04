"""Shot-list beats: closed block-kind enum, per-kind data checks, variety + hook rules.

Each beat lines up with one b-roll segment (same index as broll_prompts /
visual_intents). Shape:

    {"block": "kinetic_text|count_up|code_panel|diagram|stock_or_ai_still",
     "keywords": [...], "fallback": "<another kind>", "data": {...}}

normalize_beats() repairs what it can (unknown kind -> fallback, bad data ->
fallback, stock in the hook -> kinetic_text, 3-in-a-row -> swap, a still as the
final beat -> takeaway card, plain kinetic_text over 30% of the beats -> diagram
or the beat's own fallback, more than 2 stills -> motion) and returns
(beats, warnings). It never raises on model output. Each beat also carries a
"role" (hook / how / catch / takeaway) for the script structure check.

enforce_sourced_numbers() keeps a count_up only when its figure appears in the
research (and fills in the research line's source when the beat gave none);
an unsourced figure becomes a diagram, never an invented number on screen.
"""

import math
import re

from .style import block_kinds, block_limits, hook_rules, max_same_run, load_style

ROLES = ("hook", "how", "catch", "takeaway")
TEXT_DOMINANT = ("kinetic_text", "code_panel")  # mostly words on screen
MAX_TEXT_RUN = 2
MOTION = ("kinetic_text", "count_up", "code_panel", "diagram")
# a limitation / trade-off sentence for the "catch" beat
CATCH_RE = re.compile(r"\b(the catch|catch is|but|however|limit(s|ation|ed)?|can't|cannot|fail(s|ed)?|struggle(s)?|"
                      r"false (positive|alarm)s?|trade-?off|downside|weak(ness|nesses)?|still miss(es)?|isn't perfect|"
                      r"not perfect|blind spot|problem is|risk)\b", re.I)
NUM_TOKEN_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
SOURCE_TAG_RE = re.compile(r"\[source:\s*([^\]]+)\]")


def _variety() -> dict:
    return dict(load_style().get("variety", {}))


def kinetic_cap(n_beats: int) -> int:
    """Most kinetic_text beats a Short may have (30% of 8 beats -> 2)."""
    share = float(_variety().get("max_kinetic_share", 0.30))
    return max(1, int(math.floor(share * n_beats + 1e-9)))


def stock_cap() -> int:
    return int(_variety().get("max_stock_beats", 2))

STOCK = "stock_or_ai_still"
CTA_RE = re.compile(r"\b(subscribe|follow (us|me|for more)|like and|hit the (like|bell)|smash|comment below|"
                    r"turn on notifications|link in (the )?bio)\b", re.I)
# setup phrases that delay the hook (first real draft opened "Let me explain how ... in sixty seconds")
INTRO_RE = re.compile(r"\b(let me (explain|show|tell|break)|let's (talk|dive|explore|break)|in (the next )?"
                      r"(30|45|60|90|thirty|forty|forty-five|fifty|sixty|ninety) seconds|in (one|a single|1) minute|"
                      r"in this (video|short)|stick around)\b", re.I)
NUMBER_RE = re.compile(r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million|billion|"
                       r"percent|half|double|twice|triple)\b", re.I)


def _words(value) -> list[str]:
    if isinstance(value, str):
        return value.split()
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    return []


def _clean_data(kind: str, data, keywords: list[str]) -> dict | None:
    """Validated data for a kind, or None when the beat can't be rendered as that kind."""
    data = data if isinstance(data, dict) else {}
    lim = block_limits(kind)
    if kind == STOCK:
        return {}
    if kind == "kinetic_text":
        words = _words(data.get("words")) or _words(data.get("text"))
        words = [w for item in words for w in str(item).split()]  # ["Financial fraud losses"] -> 3 lines
        if not words:
            return None
        words = words[: int(lim.get("max_words", 4))]
        accent = str(data.get("accent") or "").strip()
        if accent not in words and accent.split() and accent.split()[-1] in words:
            accent = accent.split()[-1]
        if accent not in words:
            accent = max(words, key=len)
        return {"words": words, "accent": accent}
    if kind == "count_up":
        raw = data.get("value")
        try:
            value = float(str(raw).replace(",", "").strip())
        except (TypeError, ValueError):
            return None
        decimals = data.get("decimals")
        if not isinstance(decimals, int):
            text = str(raw)
            decimals = len(text.split(".")[1]) if "." in text else 0
        return {"value": value, "decimals": max(0, min(2, decimals)),
                "prefix": str(data.get("prefix") or "")[:3], "suffix": str(data.get("suffix") or "")[:4],
                "label": str(data.get("label") or " ".join(keywords[:3]))[:40],
                "source": str(data.get("source") or "")[:60]}
    if kind == "code_panel":
        lines = data.get("lines")
        if isinstance(lines, str):
            lines = lines.splitlines()
        if not isinstance(lines, list):
            return None
        width = int(lim.get("max_line_chars", 34))
        lines = [str(l).rstrip()[:width] for l in lines if str(l).strip()][: int(lim.get("max_lines", 6))]
        if not lines:
            return None
        return {"lines": lines, "language": str(data.get("language") or "bash").lower(),
                "title": str(data.get("title") or "terminal")[:24]}
    if kind == "diagram":
        nodes = [str(n).strip()[:18] for n in (data.get("nodes") or []) if str(n).strip()]
        if not (int(lim.get("min_nodes", 2)) <= len(nodes)):
            return None
        return {"nodes": nodes[: int(lim.get("max_nodes", 4))],
                "title": str(data.get("title") or "")[:28]}
    return None


def _kinetic_from_keywords(keywords: list[str]) -> dict | None:
    words = [w for k in keywords for w in str(k).split()][:3]
    return _clean_data("kinetic_text", {"words": words}, keywords) if words else None


def _diagram_from_keywords(keywords: list[str]) -> dict | None:
    nodes = []
    for k in keywords:
        k = str(k).strip()
        if k and k.lower() not in {n.lower() for n in nodes}:
            nodes.append(k[:1].upper() + k[1:])
    return _clean_data("diagram", {"nodes": nodes[:4]}, keywords) if len(nodes) >= 2 else None


def _resolve(beat: dict, kinds: tuple, avoid=None, forbid_stock: bool = False):
    """(kind, data) for a beat: its own kind, then its fallback, then kinetic_text, then a
    keyword diagram, then stock. `avoid` is a kind or a set of kinds to skip."""
    avoid = {avoid} if isinstance(avoid, str) else set(avoid or ())
    keywords = beat["keywords"]
    raw = beat.get("raw", beat.get("data"))
    tried = [beat.get("block"), beat.get("fallback"), "kinetic_text", "diagram", STOCK]
    for kind in tried:
        if kind not in kinds or kind in avoid or (forbid_stock and kind == STOCK):
            continue
        data = _clean_data(kind, raw, keywords)
        if data is None and kind == "kinetic_text":
            data = _kinetic_from_keywords(keywords)
        if data is None and kind == "diagram":
            data = _diagram_from_keywords(keywords)
        if data is not None:
            return kind, data
    if forbid_stock or STOCK in avoid:  # never a still where one is not allowed
        for kind, data in (("kinetic_text", _kinetic_from_keywords(keywords or ["key", "point"])),):
            if data is not None:
                return kind, data
    return STOCK, {}


def _count(beats: list[dict], kind: str, skip: int | None = None) -> int:
    return sum(1 for j, b in enumerate(beats) if b["block"] == kind and j != skip)


def _avoid_for(beats: list[dict], i: int, extra=()) -> set:
    """Kinds beat i must not become: caps already reached, a still as the last beat, runs of 3."""
    avoid = set(extra)
    if _count(beats, "kinetic_text", skip=i) >= kinetic_cap(len(beats)):
        avoid.add("kinetic_text")
    if _count(beats, STOCK, skip=i) >= stock_cap() or i == len(beats) - 1:
        avoid.add(STOCK)
    limit = max_same_run()
    kinds = [b["block"] for b in beats]
    for kind in set(kinds):
        for start in range(max(0, i - limit), i + 1):
            window = [kind if j == i else kinds[j] for j in range(start, min(len(kinds), start + limit + 1))]
            if len(window) == limit + 1 and len(set(window)) == 1:
                avoid.add(kind)
    return avoid


def _convert(beats: list[dict], i: int, kinds: tuple, why: str, warnings: list[str], extra=()) -> bool:
    old = beats[i]["block"]
    kind, data = _resolve(beats[i], kinds, avoid=_avoid_for(beats, i, extra) | {old},
                          forbid_stock=(i == 0 or i == len(beats) - 1))
    if kind == old:
        return False
    warnings.append(f"beat {i}: {why}, {old} -> {kind}")
    beats[i]["block"], beats[i]["data"] = kind, data
    return True


def normalize_beats(raw, n_segments: int = 8, prompts: list | None = None) -> tuple[list[dict], list[str]]:
    kinds = block_kinds()
    default_fb = load_style()["blocks"].get("default_fallback", STOCK)
    warnings: list[str] = []
    prompts = prompts or []
    raw = raw if isinstance(raw, list) else []
    beats = []
    for i in range(n_segments):
        item = raw[i] if i < len(raw) and isinstance(raw[i], dict) else {}
        kw = item.get("keywords")
        keywords = [str(k) for k in kw if isinstance(k, (str, int, float))][:8] if isinstance(kw, list) else []
        if not keywords and i < len(prompts):
            keywords = str(prompts[i]).split()[:4]
        block = str(item.get("block") or item.get("kind") or "").strip().lower()
        if block not in kinds:
            if block:
                warnings.append(f"beat {i}: unknown block '{block}', using fallback")
            block = None
        fallback = str(item.get("fallback") or "").strip().lower()
        if fallback not in kinds or fallback == block:
            fallback = default_fb if block != default_fb else "kinetic_text"
        beat = {"block": block, "keywords": keywords, "fallback": fallback, "data": item.get("data")}
        last = i == n_segments - 1 and n_segments > 1
        kind, data = _resolve(beat, kinds, forbid_stock=(i == 0 and not hook_rules().get("stock_allowed_in_hook", False))
                              or last)
        if block and kind != block:
            warnings.append(f"beat {i}: {block} data unusable or not allowed, rendered as {kind}")
        role = str(item.get("role") or "").strip().lower()
        if role not in ROLES:
            role = "hook" if i == 0 else "takeaway" if last else "how"
        beats.append({"block": kind, "keywords": keywords, "fallback": fallback, "data": data, "role": role,
                      "raw": item.get("data")})

    # Variety governor: never more than max_same_run of one kind in a row.
    limit = max_same_run()
    for i in range(limit, len(beats)):
        window = [b["block"] for b in beats[i - limit:i + 1]]
        if len(set(window)) == 1:
            kind, data = _resolve(beats[i], kinds, avoid=_avoid_for(beats, i, {window[0]}),
                                  forbid_stock=(i == len(beats) - 1))
            warnings.append(f"beat {i}: {limit + 1} '{window[0]}' in a row, switched to {kind}")
            beats[i]["block"], beats[i]["data"] = kind, data

    # Plain kinetic_text at most ~30% of the beats (shots are near-equal, so ~30% of the
    # runtime): convert the middle ones first, the takeaway and the hook last.
    order = [j for j in range(1, len(beats) - 1)] + ([len(beats) - 1] if len(beats) > 1 else []) + [0]
    for j in order:
        if _count(beats, "kinetic_text") <= kinetic_cap(len(beats)):
            break
        if beats[j]["block"] == "kinetic_text":
            _convert(beats, j, kinds, "kinetic_text over 30%", warnings, {"kinetic_text"})

    # At most stock_cap stills; extra ones (from the end) become motion blocks.
    for j in reversed(range(len(beats))):
        if _count(beats, STOCK) <= stock_cap():
            break
        if beats[j]["block"] == STOCK:
            _convert(beats, j, kinds, "too many stills", warnings, {STOCK})

    # At most 2 text-dominant blocks (kinetic_text / code_panel) in a row: the middle of a
    # run of 3 becomes a diagram / count_up (visual) where the data allows.
    for i in range(MAX_TEXT_RUN, len(beats)):
        window = [b["block"] for b in beats[i - MAX_TEXT_RUN:i + 1]]
        if all(k in TEXT_DOMINANT for k in window):
            j = i - 1 if i - 1 > 0 else i
            if not _convert(beats, j, kinds, "3 text-dominant blocks in a row", warnings, set(TEXT_DOMINANT)):
                _convert(beats, i, kinds, "3 text-dominant blocks in a row", warnings, set(TEXT_DOMINANT))

    # Last resort for degenerate input (one-word keywords, no data): variety beats the caps.
    for i in range(limit, len(beats)):
        window = [b["block"] for b in beats[i - limit:i + 1]]
        if len(set(window)) == 1:
            kind, data = _resolve(beats[i], kinds, avoid={window[0]}, forbid_stock=(i == len(beats) - 1))
            if kind == window[0]:
                if i == len(beats) - 1:  # the takeaway stays a motion card: break the run one beat earlier
                    i -= 1
                    kind, data = _resolve(beats[i], kinds, avoid={window[0]})
                if kind == window[0]:
                    kind, data = STOCK, {}
            warnings.append(f"beat {i}: {limit + 1} '{window[0]}' in a row, switched to {kind}")
            beats[i]["block"], beats[i]["data"] = kind, data

    for b in beats:
        b.pop("raw", None)
    return beats, warnings


def text_run_ok(beats: list[dict]) -> bool:
    kinds = [b.get("block") for b in beats]
    return not any(all(k in TEXT_DOMINANT for k in kinds[i - MAX_TEXT_RUN:i + 1])
                   for i in range(MAX_TEXT_RUN, len(kinds)))


def has_role(beats: list[dict], role: str) -> bool:
    return any(isinstance(b, dict) and b.get("role") == role for b in beats or [])


def kinetic_share(beats: list[dict]) -> float:
    return _count(beats, "kinetic_text") / max(1, len(beats))


def _num(text: str) -> float | None:
    try:
        return float(str(text).replace(",", ""))
    except ValueError:
        return None


def research_figures(research: str) -> list[tuple[float, str]]:
    """(number, source) for every figure in the research; source = the line's [source: ...] tag."""
    text = str(research or "")
    if "No live research available" in text:
        return []
    out = []
    for line in text.splitlines():
        tag = SOURCE_TAG_RE.search(line)
        src = tag.group(1).strip() if tag else ""
        body = SOURCE_TAG_RE.sub("", line)
        for m in NUM_TOKEN_RE.findall(body):
            v = _num(m)
            if v is not None:
                out.append((v, src))
    return out


def _sourced(value: float, figures: list[tuple[float, str]]) -> tuple[bool, str]:
    for v, src in figures:
        if abs(v - value) <= max(1e-9, abs(v) * 1e-6):
            return True, src
    return False, ""


def enforce_sourced_numbers(beats: list[dict], research: str) -> tuple[list[dict], list[str], bool]:
    """Keep only count_up beats whose figure is in the research. Returns (beats, warnings, has_number).

    An unsourced count_up becomes its fallback / a diagram (never an invented figure).
    With no sourced number at all, the Short must at least carry one diagram.
    """
    kinds = block_kinds()
    figures = research_figures(research)
    warnings: list[str] = []
    has_number = False
    for i, b in enumerate(beats):
        if b.get("block") != "count_up":
            continue
        ok, src = _sourced(float(b["data"].get("value", 0)), figures)
        if ok:
            has_number = True
            if not b["data"].get("source"):
                b["data"]["source"] = ("Source: " + src)[:60] if src else ""
            continue
        b.setdefault("raw", {})
        _convert(beats, i, kinds, f"count_up figure {b['data'].get('value')} not found in research",
                 warnings, {"count_up"})
        b.pop("raw", None)
    if not has_number and not any(b.get("block") == "diagram" for b in beats) and len(beats) > 2:
        for i in list(range(1, len(beats) - 1)):
            if beats[i]["block"] in ("kinetic_text", STOCK):
                kind, data = "diagram", _diagram_from_keywords(beats[i]["keywords"])
                if data:
                    warnings.append(f"beat {i}: no sourced number, {beats[i]['block']} -> diagram")
                    beats[i]["block"], beats[i]["data"] = kind, data
                    break
    if not has_number:
        warnings.append("numbers: no count_up with a figure from the research")
    return beats, warnings, has_number


_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                       "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_SCALES = {"hundred": 100, "thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
_SMALL_OK = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100}  # "one model", "two steps", "a hundred" are not figures


def spoken_numbers(script: str) -> list[tuple[float, str]]:
    """(value, text) for every figure in the narration, digits or words ("eighty-four percent")."""
    out = []
    toks = re.findall(r"[A-Za-z]+|\d[\d,]*(?:\.\d+)?", str(script or "").replace("-", " "))
    i = 0
    while i < len(toks):
        t = toks[i].lower()
        if t[0].isdigit():
            v = _num(t)
            if v is not None:
                out.append((v, toks[i]))
            i += 1
            continue
        if t in _UNITS or t in _TENS:
            total, cur, j = 0.0, 0.0, i
            while j < len(toks) and toks[j].lower() in {**_UNITS, **_TENS, **_SCALES, "and": 0, "point": 0}:
                w = toks[j].lower()
                if w == "point":  # "one point six six billion" -> 1.66e9
                    digits = ""
                    while j + 1 < len(toks) and toks[j + 1].lower() in _UNITS and _UNITS[toks[j + 1].lower()] < 10:
                        j += 1
                        digits += str(_UNITS[toks[j].lower()])
                    if digits:
                        cur += float("0." + digits)
                elif w in _UNITS:
                    cur += _UNITS[w]
                elif w in _TENS:
                    cur += _TENS[w]
                elif w == "hundred":
                    cur = (cur or 1) * 100
                elif w in _SCALES:
                    total += (cur or 1) * _SCALES[w]
                    cur = 0
                j += 1
            out.append((total + cur, " ".join(toks[i:j])))
            i = j
            continue
        i += 1
    return out


def unsourced_spoken_numbers(script: str, research: str) -> list[str]:
    """Figures the narration says that appear nowhere in the research (in any scale).

    Small counts (one..ten, a hundred) and years are left alone. Empty when the research
    has no figures at all (nothing to check against; the prompt then forbids figures).
    """
    figures = research_figures(research)
    if not figures:
        return []
    vals = {round(v, 6) for v, _ in figures}
    bad = []
    for v, text in spoken_numbers(script):
        if v in _SMALL_OK or (1900 <= v <= 2100 and float(v).is_integer()):
            continue
        scaled = {round(v / k, 6) for k in (1, 1e3, 1e6, 1e9, 1e12)}
        if not scaled & vals:
            bad.append(text)
    return bad


def structure_problems(script: str, beats: list[dict]) -> list[str]:
    """hook -> how (2-3) -> catch -> takeaway: what is missing."""
    out = []
    if not has_role(beats, "catch") or not CATCH_RE.search(str(script or "")):
        out.append("structure: no catch/limit beat (what the technology gets wrong or cannot do)")
    if beats and beats[-1].get("block") not in MOTION:
        out.append("structure: the final beat is not a motion takeaway card")
    return out


def variety_ok(beats: list[dict]) -> bool:
    limit = max_same_run()
    kinds = [b.get("block") for b in beats]
    return not any(len(set(kinds[i - limit:i + 1])) == 1 for i in range(limit, len(kinds)))


# "Ten picometers—that is how AI reads your X-rays" (draft 5): a bare noun phrase, a dash,
# then a clause. A short lead-in followed by a dash reads oddly when spoken aloud.
# Owner rule (4 Nov 2026): the hook sentence carries a concrete number or a surprising specific.
HOOK_NUMBER_RE = re.compile(
    r"\d|\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|sixty|"
    r"seventy|eighty|ninety|hundreds?|thousands?|millions?|billions?|trillions?|dozens?|half|twice|double|triple|"
    r"percent)\b", re.I)
HOOK_SPECIFIC_RE = re.compile(
    r"\b(faster|slower|smaller|bigger|larger|cheaper|better|worse|more|fewer|less|longer|shorter|stronger|"
    r"quieter|louder|sharper|earlier|later)\s+(\w+\s+){0,4}?than\b"
    r"|\b(milli|micro|nano|pico)(seconds?|meters?|metres?|watts?)\b"
    r"|\b(every|per|each|a|an)\s+(second|minute|hour|day|heartbeat|blink)\b"
    r"|\btimes\s+(a|per|faster|more|less|smaller|bigger)\b"
    r"|\b(world's|fastest|slowest|smallest|largest|biggest)\b", re.I)


def hook_specific(script: str) -> str:
    """'number' or 'specific' when sentence 1 (the hook) has one, else ''."""
    first = first_sentence(script)
    if HOOK_NUMBER_RE.search(first):
        return "number"
    if HOOK_SPECIFIC_RE.search(first):
        return "specific"
    return ""


HOOK_FRAGMENT_RE = re.compile(r"^\W*(\S+\s+){0,3}\S*?\s*(—|–|--|\s-\s)")
_APOS = str.maketrans({"\u2019": "'", "\u2018": "'"})


def sentences(script: str) -> list[str]:
    text = " ".join(str(script or "").split())
    return [x for x in re.split(r"(?<=[.!?])\s+", text) if x]


def first_sentence(script: str) -> str:
    parts = sentences(script)
    return parts[0] if parts else ""


def banned_phrases_in(script: str, phrases) -> list[str]:
    """Banned setup/filler phrases that occur anywhere in the script (case/apostrophe-insensitive)."""
    text = " ".join(str(script or "").translate(_APOS).casefold().split())
    out = []
    for p in phrases or []:
        p2 = " ".join(str(p).translate(_APOS).casefold().split())
        if p2 and re.search(r"(?<![a-z])" + re.escape(p2) + r"(?![a-z])", text):
            out.append(str(p))
    return out


def hook_warnings(script: str) -> list[str]:
    """Script-level hook/ending checks (can't be auto-fixed; reported for review)."""
    rules = hook_rules()
    wpm = float(load_style()["pace"]["target_wpm"])
    words = str(script or "").split()
    out = []
    if not words:
        return out
    by_number = words[: max(1, int(rules.get("number_by_s", 8) * wpm / 60))]
    if not NUMBER_RE.search(" ".join(by_number)):
        out.append(f"hook: no concrete number in the first ~{rules.get('number_by_s', 8):g} s")
    if re.match(r"(hey|hi|hello|welcome|what's up|in this video|today we)", " ".join(words[:4]), re.I) \
            or INTRO_RE.search(" ".join(words[:14])):
        out.append("hook: script opens with an intro/greeting")
    first = first_sentence(script)
    max_words = int(rules.get("max_words", 12) or 12)
    n_first = len(first.split())
    if n_first > max_words:
        out.append(f"hook: sentence 1 has {n_first} words (max {max_words}); say one plain claim")
    if not hook_specific(script):
        out.append("hook: sentence 1 has no concrete number or surprising specific; state a figure from the "
                   "research, or a measured comparison, speed or size")
    if HOOK_FRAGMENT_RE.search(first):
        out.append("hook: opens with a fragment and a dash; say it as one plain conversational sentence")
    opening = " ".join(sentences(script)[:2])
    found = banned_phrases_in(opening, rules.get("banned_phrases") or [])
    found += [p for p in banned_phrases_in(script, rules.get("filler_phrases") or []) if p not in found]
    if found:
        out.append("hook: setup/filler phrase " + ", ".join(f'"{p}"' for p in found[:4])
                   + "; cut it and state the claim directly")
    if rules.get("no_end_cta", True):
        tail = " ".join(words[-25:])
        if CTA_RE.search(tail):
            out.append("ending: call-to-action found in the closing lines")
    return out
