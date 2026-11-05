"""Engine B sound design: soft, calm transition sounds synthesised at runtime.

No third-party samples: every clip is generated procedurally with numpy (a soft
sine chime, a band-limited noise whoosh, a gentle low swell), so licensing is
clean and nothing is vendored. The old third-party pops, impacts, clicks and squishes
are gone.

plan_sfx() turns the shot plan into events: a soft chime on the hook, a whoosh or
chime on beat changes (varied, never the same clip twice running, thinned to the
style's 6-12 per minute) and a low swell into the first count-up. Code-panel
typing gets no sound. render_sfx_track() mixes them into one mono WAV whose peak
sits UNDER_VOICE_DB (20 dB) under the voice peak; assemble.prepare_audio()
sidechain-ducks it (and the music) under speech before the -14 LUFS / -1.5 dBTP
loudnorm. Every step is optional: a failure here logs and the video renders
without effects.
"""

import os
import random
import subprocess
import wave
from functools import lru_cache
from pathlib import Path

from .log import log

SR = 48000
UNDER_VOICE_DB = 20.0          # SFX bed peak sits >= 18-20 dB under the voice peak
PER_MIN = (6, 12)
# Relative to the bed level. All transitions are soft; the hook chime is the loudest.
TRANSITION_GAIN = {"chime": -2.0, "whoosh": -3.0, "swell": -4.0}

# Procedural clip recipes. duration_s / peak_s drive event alignment in plan_sfx.
_RECIPES = [
    {"id": "chime_01", "category": "chime", "duration_s": 1.6, "peak_s": 0.06,
     "synth": {"kind": "chime", "f0": 880.0, "partials": [(1.0, 1.0), (2.0, 0.25), (3.0, 0.08)], "decay": 0.55},
     "description": "soft sine chime (A5), slow decay"},
    {"id": "chime_02", "category": "chime", "duration_s": 1.6, "peak_s": 0.06,
     "synth": {"kind": "chime", "f0": 659.25, "partials": [(1.0, 1.0), (2.0, 0.2), (4.0, 0.05)], "decay": 0.6},
     "description": "soft sine chime (E5), slow decay"},
    {"id": "whoosh_01", "category": "whoosh", "duration_s": 0.9, "peak_s": 0.6,
     "synth": {"kind": "whoosh", "f_lo": 300.0, "f_hi": 2200.0, "seed": 11},
     "description": "filtered-noise soft whoosh, rising into the cut"},
    {"id": "whoosh_02", "category": "whoosh", "duration_s": 0.8, "peak_s": 0.25,
     "synth": {"kind": "whoosh", "f_lo": 1800.0, "f_hi": 350.0, "seed": 23},
     "description": "filtered-noise soft whoosh, falling out of the cut"},
    {"id": "whoosh_03", "category": "whoosh", "duration_s": 1.0, "peak_s": 0.5,
     "synth": {"kind": "whoosh", "f_lo": 500.0, "f_hi": 1500.0, "seed": 37},
     "description": "filtered-noise air whoosh, centred"},
    {"id": "swell_01", "category": "swell", "duration_s": 1.4, "peak_s": 1.3,
     "synth": {"kind": "swell", "f0": 110.0},
     "description": "gentle low swell (A2 + fifth), slow attack"},
]


def enabled() -> bool:
    return os.environ.get("DYPOL_SFX_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}


@lru_cache(maxsize=1)
def load_manifest() -> dict:
    return {"version": 2, "license": "procedural (generated at runtime, no third-party audio)",
            "sample_rate": SR, "clips": [dict(c) for c in _RECIPES]}


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
    """Event list [{t, at, clip, kind, gain_db}] sorted by start time. `typing` is accepted but silent."""
    cats = clips_by_category()
    rng = random.Random(seed)
    events = []
    if cats.get("chime"):
        events.append(_event(cats["chime"][0], max(0.0, hook_t), "hook", align="start"))
    minutes = max(duration / 60.0, 1e-6)
    lo_n = max(1, int(round(PER_MIN[0] * minutes)))
    hi_n = max(lo_n, int(PER_MIN[1] * minutes))
    min_gap = 60.0 / PER_MIN[1]
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
    pool = (cats.get("whoosh") or []) + (cats.get("chime") or [])
    prev_id = cats["chime"][0]["id"] if cats.get("chime") else None
    for i, t in enumerate(chosen):
        if riser_at is not None and abs(t - riser_at) < 0.05 and cats.get("swell"):
            events.append(_event(cats["swell"][0], t, "riser", align="end"))
            prev_id = cats["swell"][0]["id"]
            continue
        # mostly whooshes, every third a soft chime; rotate inside each family
        fam = cats.get("whoosh" if i % 3 != 2 else "chime") or pool
        options = [c for c in fam if c["id"] != prev_id] or fam
        clip = options[rng.randrange(len(options))]
        events.append(_event(clip, t, "transition", align="peak"))
        prev_id = clip["id"]
    if riser_at is not None and cats.get("swell") and not any(e["kind"] == "riser" for e in events):
        if riser_at - float(cats["swell"][0]["duration_s"]) > hook_t + 0.8:
            events.append(_event(cats["swell"][0], riser_at, "riser", align="end"))
    events.sort(key=lambda e: e["t"])
    return events


def transitions_per_minute(events: list[dict], duration: float) -> float:
    n = sum(1 for e in events if e["kind"] in {"transition", "riser"})
    return n / max(duration / 60.0, 1e-6)


# ───────────────────────── synthesis ─────────────────────────

def synth_clip(clip: dict):
    """Peak-normalised float32 mono samples for a recipe. Deterministic."""
    import numpy as np
    s = clip["synth"]
    n = int(round(float(clip["duration_s"]) * SR))
    t = np.arange(n) / SR
    if s["kind"] == "chime":
        x = sum(amp * np.sin(2 * np.pi * s["f0"] * mult * t) * np.exp(-t * mult / s["decay"])
                for mult, amp in s["partials"])
        a = int(float(clip["peak_s"]) * SR)
        x[:a] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
    elif s["kind"] == "whoosh":
        rng = np.random.default_rng(s["seed"])
        noise = rng.standard_normal(n)
        # time-varying one-pole band: centre sweeps f_lo -> f_hi
        fc = np.geomspace(s["f_lo"], s["f_hi"], n)
        alpha = 1 - np.exp(-2 * np.pi * fc / SR)
        lp, hp_state, y = 0.0, 0.0, np.zeros(n)
        for i in range(n):
            lp += alpha[i] * (noise[i] - lp)
            hp_state += 0.02 * (lp - hp_state)        # remove rumble
            y[i] = lp - hp_state
        pk = float(clip["peak_s"]) / float(clip["duration_s"])
        u = np.linspace(0, 1, n)
        env = np.where(u < pk, np.sin(0.5 * np.pi * u / max(pk, 1e-3)) ** 2,
                       np.cos(0.5 * np.pi * (u - pk) / max(1 - pk, 1e-3)) ** 2)
        x = y * env
    elif s["kind"] == "swell":
        f0 = s["f0"]
        x = np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * f0 * 1.5 * t) + 0.25 * np.sin(2 * np.pi * f0 * 2 * t)
        pk = int(float(clip["peak_s"]) * SR)
        env = np.ones(n)
        env[:pk] = np.linspace(0, 1, pk) ** 2
        env[pk:] = np.linspace(1, 0, n - pk)
        x = x * env
    else:
        raise ValueError(f"unknown synth kind {s['kind']}")
    peak = float(np.max(np.abs(x))) or 1.0
    return (x / peak).astype(np.float32)


def write_clip_wav(clip: dict, path: Path) -> Path:
    import numpy as np
    pcm = (synth_clip(clip) * 0.7 * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return Path(path)


def _decode(path: Path):
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def render_sfx_track(events: list[dict], duration: float, voice_path: Path, out_path: Path,
                     under_db: float = UNDER_VOICE_DB) -> Path | None:
    """Mix events into a mono 48 kHz WAV whose peak is >= under_db below the voice peak. None when empty."""
    if not events:
        return None
    import numpy as np
    n = int(round(duration * SR))
    track = np.zeros(n, dtype=np.float32)
    cache = {}
    for ev in events:
        if ev["clip"] not in cache:
            cache[ev["clip"]] = synth_clip(_clip(ev["clip"]))
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
    # Unit-gain events land exactly `under_db` under the voice; overlaps never exceed it.
    peak = float(np.max(np.abs(track))) if len(track) else 0.0
    track *= level * (min(1.0, 1.0 / peak) if peak > 0 else 1.0)
    pcm = (np.clip(track, -1.0, 1.0) * 32767).astype("<i2")
    out_path = Path(out_path)
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    log(f"SFX track: {len(events)} events -> {out_path.name}")
    return out_path
