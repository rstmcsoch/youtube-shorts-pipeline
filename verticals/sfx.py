"""Engine B sound design: a small vendored CC0 set (sfx/, Kenney) laid under the voice.

plan_sfx() turns the shot plan into events: an impact on the hook, a whoosh or pop
on beat changes (varied, never the same clip twice running, thinned to the
style's 6-12 per minute), a riser into the first count-up, and soft clicks while
a code panel types. render_sfx_track() mixes them into one mono WAV whose peaks
sit ~12 dB under the voice's peak; assemble.prepare_audio() adds it beside the
ducked music bed before the -14 LUFS loudnorm. Every step is optional: a failure
here logs and the video renders without effects.
"""

import json
import os
import random
import subprocess
import wave
from functools import lru_cache
from pathlib import Path

from .log import log

SFX_DIR = Path(__file__).resolve().parent.parent / "sfx"
SR = 48000
UNDER_VOICE_DB = 12.0
PER_MIN = (6, 12)
# Relative to the bed level (peaks UNDER_VOICE_DB under the voice peak). The first real
# draft measured whooshes only ~9 dB under the voice RMS (style guide: ~-18 dB), loud in
# pauses, so transitions sit 6 dB lower; the hook impact stays the loudest event.
TRANSITION_GAIN = {"whoosh": -6.0, "pop": -7.0, "riser": -8.0, "impact": -2.0, "click": -9.0}


def enabled() -> bool:
    return os.environ.get("DYPOL_SFX_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}


@lru_cache(maxsize=1)
def load_manifest() -> dict:
    return json.loads((SFX_DIR / "manifest.json").read_text(encoding="utf-8"))


def clips_by_category() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for clip in load_manifest().get("clips", []):
        out.setdefault(clip["category"], []).append(clip)
    return out


def _clip(cid: str) -> dict:
    return next(c for c in load_manifest()["clips"] if c["id"] == cid)


def _event(clip: dict, at: float, kind: str, align: str = "peak") -> dict:
    if align == "end":
        start = at - float(clip["duration_s"])
    elif align == "peak":
        start = at - float(clip.get("peak_s", 0.0))
    else:
        start = at
    return {"t": round(max(0.0, start), 3), "at": round(at, 3), "clip": clip["id"], "kind": kind,
            "gain_db": TRANSITION_GAIN.get(clip["category"], 0.0)}


def plan_sfx(duration: float, beat_changes: list[float], sub_cuts: list[float] | None = None,
             hook_t: float = 0.0, typing: list[tuple[float, float]] | None = None,
             riser_at: float | None = None, seed: int = 7) -> list[dict]:
    """Event list [{t, at, clip, kind, gain_db}] sorted by start time."""
    cats = clips_by_category()
    rng = random.Random(seed)
    events = []
    if cats.get("impact"):
        events.append(_event(cats["impact"][0], max(0.0, hook_t), "hook", align="start"))
    minutes = max(duration / 60.0, 1e-6)
    lo_n = max(1, int(round(PER_MIN[0] * minutes)))
    hi_n = max(lo_n, int(PER_MIN[1] * minutes))
    min_gap = 60.0 / PER_MIN[1]
    # beat changes first; sub-shot cuts only to reach the floor
    chosen, last = [], hook_t
    for t in sorted(beat_changes):
        if t - last >= min_gap - 1e-6 and duration - t > 0.3:
            chosen.append(t)
            last = t
    if len(chosen) < lo_n:
        for t in sorted(sub_cuts or []):
            if len(chosen) >= lo_n:
                break
            if all(abs(t - c) >= min_gap * 0.6 for c in chosen + [hook_t]) and duration - t > 0.3:
                chosen.append(t)
        chosen.sort()
    chosen = chosen[:hi_n]
    pool = (cats.get("whoosh") or []) + (cats.get("pop") or [])
    prev_id = None
    for i, t in enumerate(chosen):
        if riser_at is not None and abs(t - riser_at) < 0.05 and cats.get("riser"):
            events.append(_event(cats["riser"][0], t, "riser", align="end"))
            prev_id = cats["riser"][0]["id"]
            continue
        # alternate families (whoosh, pop, whoosh ...) and rotate inside each
        fam = cats.get("whoosh" if i % 3 != 2 else "pop") or pool
        options = [c for c in fam if c["id"] != prev_id] or fam
        clip = options[rng.randrange(len(options))]
        events.append(_event(clip, t, "transition", align="peak"))
        prev_id = clip["id"]
    if riser_at is not None and cats.get("riser") and not any(e["kind"] == "riser" for e in events):
        if riser_at - float(cats["riser"][0]["duration_s"]) > hook_t + 0.8:
            events.append(_event(cats["riser"][0], riser_at, "riser", align="end"))
    clicks = cats.get("click") or []
    for start, end in typing or []:
        t, k = start, 0
        while clicks and t < min(end, duration - 0.05):
            clip = clicks[k % len(clicks)]
            ev = _event(clip, t, "typing", align="start")
            ev["gain_db"] += rng.uniform(-3.0, 0.0)
            events.append(ev)
            t += rng.uniform(0.09, 0.15)
            k += 1 + rng.randrange(2)
    events.sort(key=lambda e: e["t"])
    return events


def transitions_per_minute(events: list[dict], duration: float) -> float:
    n = sum(1 for e in events if e["kind"] in {"transition", "riser"})
    return n / max(duration / 60.0, 1e-6)


def _decode(path: Path):
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def render_sfx_track(events: list[dict], duration: float, voice_path: Path, out_path: Path,
                     under_db: float = UNDER_VOICE_DB) -> Path | None:
    """Mix events into a mono 48 kHz WAV, peaks ~under_db below the voice peak. None when empty."""
    if not events:
        return None
    import numpy as np
    n = int(round(duration * SR))
    track = np.zeros(n, dtype=np.float32)
    cache = {}
    for ev in events:
        if ev["clip"] not in cache:
            x = _decode(SFX_DIR / _clip(ev["clip"])["file"])
            peak = float(np.max(np.abs(x))) if len(x) else 0.0
            cache[ev["clip"]] = x / peak if peak > 0 else x
        x = cache[ev["clip"]] * (10 ** (ev.get("gain_db", 0.0) / 20))
        i = int(round(ev["t"] * SR))
        if i >= n:
            continue
        j = min(n, i + len(x))
        track[i:j] += x[: j - i]
    try:
        voice = _decode(voice_path)
        vpeak = float(np.max(np.abs(voice))) if len(voice) else 0.5
    except Exception:
        vpeak = 0.5
    level = max(vpeak, 1e-3) * (10 ** (-under_db / 20))
    track *= level
    pcm = (np.clip(track, -1.0, 1.0) * 32767).astype("<i2")
    out_path = Path(out_path)
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    log(f"SFX track: {len(events)} events -> {out_path.name}")
    return out_path
