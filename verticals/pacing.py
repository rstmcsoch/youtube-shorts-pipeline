"""Engine B pacing: the screen changes every 2-4 s and nothing is static past 3.5 s.

cut_points() places cuts on the narration timeline, preferring clause boundaries
(a word ending in punctuation or followed by a pause) inside the style guide's
window, then plain word gaps, then the window target. group_cuts() packs those
segments into source shots (one visual per shot, as many shots as before), so a
long shot becomes 2+ sub-shots with a different move: a punch-in or new crop on
stills and clips, a pull-back on motion blocks. Cut times are frame-quantised so
every encoded sub-shot adds up to the exact narration length.
"""

import math

from .style import load_style

TARGET_S = 2.8  # inside the 2-4 s window, under the 3.5 s static cap
CLAUSE_PUNCT = ".!?;:,—–"


def pace_limits() -> tuple[float, float]:
    """(min change interval, max static) from the style guide."""
    s = load_style().get("screen", {})
    lo = float((s.get("change_every_s") or [2.0, 4.0])[0])
    hi = min(float(s.get("max_static_s", 3.5)), float((s.get("change_every_s") or [2.0, 4.0])[1]))
    return lo, hi


def _boundaries(words: list[dict] | None, pause: float = 0.25) -> list[tuple[float, bool]]:
    """Candidate cut times between words: (time, is_clause_boundary)."""
    out = []
    words = [w for w in (words or []) if isinstance(w, dict) and "start" in w and "end" in w]
    for i, w in enumerate(words[:-1]):
        nxt = words[i + 1]
        end, start = float(w["end"]), float(nxt["start"])
        t = (end + start) / 2 if start > end else end
        clause = str(w.get("word", "")).rstrip()[-1:] in CLAUSE_PUNCT or (start - end) >= pause
        out.append((t, clause))
    return out


def _q(t: float, fps: int) -> float:
    return round(t * fps) / fps


def cut_points(duration: float, words: list[dict] | None = None, lo: float | None = None,
               hi: float | None = None, fps: int = 30, target: float = TARGET_S) -> list[float]:
    """Cut times including 0 and duration; every gap in [lo, hi] where the length allows."""
    dlo, dhi = pace_limits()
    lo = dlo if lo is None else lo
    hi = dhi if hi is None else hi
    duration = float(duration)
    if duration <= hi:
        return [0.0, duration]
    bounds = _boundaries(words)
    cuts, t = [0.0], 0.0
    while duration - t > hi:
        window = [(b, c) for b, c in bounds if t + lo <= b <= t + hi and duration - b >= min(lo, 1.0)]
        if window:
            # clause boundary first, then closeness to the target
            b, _ = min(window, key=lambda bc: (not bc[1], abs(bc[0] - (t + target))))
        else:
            b = t + min(target, (duration - t) / 2 if duration - t < 2 * lo else target)
        b = _q(b, fps)
        if b <= t + 1.0 / fps:
            b = _q(t + target, fps)
        cuts.append(b)
        t = b
    # a short tail is folded back so no segment is far under the window
    tail = duration - cuts[-1]
    if tail < lo and len(cuts) >= 2:
        prev = cuts[-1] - cuts[-2]
        if prev + tail <= hi:
            cuts.pop()
        else:
            cuts[-1] = _q(cuts[-2] + (prev + tail) / 2, fps)
    cuts.append(duration)
    return cuts


def segments(cuts: list[float]) -> list[float]:
    return [round(b - a, 6) for a, b in zip(cuts, cuts[1:])]


def group_cuts(seg_lengths: list[float], n_shots: int) -> list[list[float]]:
    """Pack consecutive segments into n_shots source shots (1+ segments each, in order)."""
    s = len(seg_lengths)
    n = max(1, min(int(n_shots), s))
    groups = [[] for _ in range(n)]
    for i, length in enumerate(seg_lengths):
        groups[min(n - 1, i * n // s)].append(length)
    return [g for g in groups if g]


def shot_groups(duration: float, words: list[dict] | None, shot_target: float, fps: int = 30) -> list[list[float]]:
    """Source shots (same count the old fixed-length planner made), each a list of sub-shot lengths."""
    segs = segments(cut_points(duration, words, fps=fps))
    n_shots = max(1, math.ceil(duration / shot_target - 1e-6))
    return group_cuts(segs, n_shots)


def max_static(groups: list[list[float]]) -> float:
    return max((x for g in groups for x in g), default=0.0)
