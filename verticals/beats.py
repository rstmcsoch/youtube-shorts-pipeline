"""Shot-list beats: closed block-kind enum, per-kind data checks, variety + hook rules.

Each beat lines up with one b-roll segment (same index as broll_prompts /
visual_intents). Shape:

    {"block": "kinetic_text|count_up|code_panel|diagram|stock_or_ai_still",
     "keywords": [...], "fallback": "<another kind>", "data": {...}}

normalize_beats() repairs what it can (unknown kind -> fallback, bad data ->
fallback, stock in the hook -> kinetic_text, 3-in-a-row -> swap) and returns
(beats, warnings). It never raises on model output.
"""

import re

from .style import block_kinds, block_limits, hook_rules, max_same_run, load_style

STOCK = "stock_or_ai_still"
CTA_RE = re.compile(r"\b(subscribe|follow (us|me|for more)|like and|hit the (like|bell)|smash|comment below|"
                    r"turn on notifications|link in (the )?bio)\b", re.I)
# setup phrases that delay the hook (first real draft opened "Let me explain how ... in sixty seconds")
INTRO_RE = re.compile(r"\b(let me (explain|show|tell|break)|let's (talk|dive|explore|break)|in (the next )?"
                      r"(\d+|thirty|forty|fifty|sixty|ninety) seconds|in this (video|short)|stick around)\b", re.I)
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
        if not words:
            return None
        words = words[: int(lim.get("max_words", 4))]
        accent = str(data.get("accent") or "").strip()
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


def _resolve(beat: dict, kinds: tuple, avoid: str | None = None, forbid_stock: bool = False):
    """(kind, data) for a beat: its own kind, then its fallback, then kinetic_text, then stock."""
    keywords = beat["keywords"]
    tried = [beat.get("block"), beat.get("fallback"), "kinetic_text", STOCK]
    for kind in tried:
        if kind not in kinds or kind == avoid or (forbid_stock and kind == STOCK):
            continue
        data = _clean_data(kind, beat.get("data"), keywords)
        if data is None and kind == "kinetic_text":
            data = _kinetic_from_keywords(keywords)
        if data is not None:
            return kind, data
    return STOCK, {}


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
        kind, data = _resolve(beat, kinds, forbid_stock=(i == 0 and not hook_rules().get("stock_allowed_in_hook", False)))
        if block and kind != block:
            warnings.append(f"beat {i}: {block} data unusable or not allowed, rendered as {kind}")
        beats.append({"block": kind, "keywords": keywords, "fallback": fallback, "data": data})

    # Variety governor: never more than max_same_run of one kind in a row.
    limit = max_same_run()
    for i in range(limit, len(beats)):
        window = [b["block"] for b in beats[i - limit:i + 1]]
        if len(set(window)) == 1:
            kind, data = _resolve(beats[i], kinds, avoid=window[0])
            warnings.append(f"beat {i}: {limit + 1} '{window[0]}' in a row, switched to {kind}")
            beats[i]["block"], beats[i]["data"] = kind, data
    return beats, warnings


def variety_ok(beats: list[dict]) -> bool:
    limit = max_same_run()
    kinds = [b.get("block") for b in beats]
    return not any(len(set(kinds[i - limit:i + 1])) == 1 for i in range(limit, len(kinds)))


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
    if rules.get("no_end_cta", True):
        tail = " ".join(words[-25:])
        if CTA_RE.search(tail):
            out.append("ending: call-to-action found in the closing lines")
    return out
