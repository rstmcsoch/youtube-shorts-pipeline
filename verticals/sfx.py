"""Engine B sound design: calm, sparse transition sounds synthesised at runtime.

No third-party samples: every clip is generated procedurally with numpy (soft low
sine chimes and a gentle low swell), so licensing is clean and nothing is vendored.
Only two families exist and both are slow-attack, low-pitched tones: a chime's
fundamental is at most CHIME_MAX_F0 (1.5 kHz) and every attack is at least
MIN_ATTACK_S (20 ms). Noise-based or percussive sounds are not part of the palette.

plan_sfx() turns the shot plan into events: a soft chime on the hook, a soft chime on
beat changes (alternating clips, thinned to the style's 6-12 per minute) and a low
swell into the first count-up, never more than one effect per section (the span
between two beat changes). Code-panel typing gets no sound. render_sfx_track() mixes
them into one mono WAV and levels EACH effect by integrated loudness (ITU-R BS.1770,
K-weighted, gated) so it sits UNDER_VOICE_LU (24 LU) below the voice's integrated
LUFS; assemble.prepare_audio() then sidechain-ducks it under speech before the
-14 LUFS / -1.5 dBTP loudnorm.

DYPOL_SFX=off (or DYPOL_SFX_ENABLED=0) disables every effect: plan_sfx() returns no
events and render_sfx_track() returns None. Every step is optional: a failure here
logs and the video renders without effects.
"""

import os
import random
import subprocess
import wave
from functools import lru_cache
from pathlib import Path

from .log import log

SR = 48000
UNDER_VOICE_LU = 24.0          # each effect's integrated loudness sits 24 LU under the voice's
UNDER_VOICE_DB = UNDER_VOICE_LU  # back-compat name (now loudness-based, not peak-based)
FALLBACK_VOICE_LUFS = -18.0    # used when the voice cannot be measured
CHIME_MAX_F0 = 1500.0
MIN_ATTACK_S = 0.020
PER_MIN = (6, 12)
ALLOWED_CATEGORIES = ("chime", "swell")
# Relative to the leveled target (all <= 0). The chime (hook + transitions) is the loudest.
TRANSITION_GAIN = {"chime": -2.0, "swell": -4.0}

_OFF = {"0", "false", "no", "off", "disabled", "none"}

# Procedural clip recipes. duration_s / peak_s drive event alignment in plan_sfx;
# attack_s is the raised-cosine fade-in (>= MIN_ATTACK_S).
_RECIPES = [
    {"id": "chime_01", "category": "chime", "duration_s": 1.6, "peak_s": 0.06, "attack_s": 0.06,
     "synth": {"kind": "chime", "f0": 880.0, "partials": [(1.0, 1.0), (2.0, 0.25), (3.0, 0.08)], "decay": 0.55},
     "description": "soft sine chime (A5), slow decay"},
    {"id": "chime_02", "category": "chime", "duration_s": 1.6, "peak_s": 0.06, "attack_s": 0.06,
     "synth": {"kind": "chime", "f0": 659.25, "partials": [(1.0, 1.0), (2.0, 0.2), (4.0, 0.05)], "decay": 0.6},
     "description": "soft sine chime (E5), slow decay"},
    {"id": "swell_01", "category": "swell", "duration_s": 1.4, "peak_s": 1.3, "attack_s": 1.3,
     "synth": {"kind": "swell", "f0": 110.0},
     "description": "gentle low swell (A2 + fifth), slow attack"},
]


def enabled() -> bool:
    """False when DYPOL_SFX=off or DYPOL_SFX_ENABLED is falsy."""
    if os.environ.get("DYPOL_SFX", "on").strip().lower() in _OFF:
        return False
    return os.environ.get("DYPOL_SFX_ENABLED", "1").strip().lower() not in _OFF


@lru_cache(maxsize=1)
def load_manifest() -> dict:
    return {"version": 3, "license": "procedural (generated at runtime, no third-party audio)",
            "sample_rate": SR, "clips": [dict(c) for c in _RECIPES]}


def clips_by_category() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for clip in load_manifest().get("clips", []):
        if clip["category"] in ALLOWED_CATEGORIES:
            out.setdefault(clip["category"], []).append(clip)
    return out


def _clip(cid: str) -> dict:
    clip = next((c for c in load_manifest()["clips"] if c["id"] == cid), None)
    if clip is None or clip["category"] not in ALLOWED_CATEGORIES:
        raise KeyError(f"unknown SFX clip {cid!r}")
    return clip


def _event(clip: dict, at: float, kind: str, align: str = "peak") -> dict:
    if align == "end":
        start = at - float(clip["duration_s"])
    elif align == "peak":
        start = at - float(clip.get("peak_s", 0.0))
    else:
        start = at
    return {"t": round(max(0.0, start), 3), "at": round(at, 3), "clip": clip["id"], "kind": kind,
            "gain_db": TRANSITION_GAIN.get(clip["category"], 0.0)}


def _section_of(t: float, bounds: list[float]) -> int:
    """Index of the section containing t; bounds are sorted section starts (bounds[0] = 0)."""
    k = 0
    for i, b in enumerate(bounds):
        if t + 1e-6 >= b:
            k = i
    return k


def plan_sfx(duration: float, beat_changes: list[float], sub_cuts: list[float] | None = None,
             hook_t: float = 0.0, typing: list[tuple[float, float]] | None = None,
             riser_at: float | None = None, seed: int = 7) -> list[dict]:
    """Event list [{t, at, clip, kind, gain_db}] sorted by start time, at most one per section.

    Sections are the spans between beat changes (the first starts at 0 and owns the hook
    chime). `typing` is accepted but silent. Returns [] when SFX are disabled.
    """
    if not enabled():
        return []
    cats = clips_by_category()
    chimes, swells = cats.get("chime") or [], cats.get("swell") or []
    rng = random.Random(seed)
    bounds = [0.0] + sorted(b for b in set(beat_changes) if 0.0 < b < duration)
    used: dict[int, dict] = {}                       # section index -> event
    hook_t = max(0.0, hook_t)
    if chimes:
        used[_section_of(hook_t, bounds)] = _event(chimes[0], hook_t, "hook", align="start")
    minutes = max(duration / 60.0, 1e-6)
    lo_n = max(1, int(round(PER_MIN[0] * minutes)))
    hi_n = max(lo_n, int(PER_MIN[1] * minutes))
    min_gap = 60.0 / PER_MIN[1]

    riser_sec = None
    if riser_at is not None and swells and riser_at - float(swells[0]["duration_s"]) > hook_t + 0.8:
        riser_sec = _section_of(riser_at, bounds)
        if riser_sec in used:                        # never stack a swell on the hook
            riser_sec = None

    def ok(t: float, chosen: list[float]) -> bool:
        return duration - t > 0.3 and all(abs(t - c) >= min_gap - 1e-6 for c in chosen + [hook_t])

    chosen: dict[int, float] = {}
    for t in bounds[1:]:
        sec = _section_of(t, bounds)
        if sec in used or sec == riser_sec:
            continue
        if ok(t, list(chosen.values())):
            chosen[sec] = t
    if len(chosen) + (riser_sec is not None) < lo_n:
        for t in sorted(sub_cuts or []):
            if len(chosen) + (riser_sec is not None) >= lo_n:
                break
            sec = _section_of(t, bounds)
            if sec in used or sec in chosen or sec == riser_sec:
                continue
            if ok(t, list(chosen.values())):
                chosen[sec] = t
    cap = hi_n - (riser_sec is not None)
    keep = sorted(chosen.items(), key=lambda kv: kv[1])[:max(0, cap)]
    prev_id = chimes[0]["id"] if chimes else None
    for sec, t in keep:
        options = [c for c in chimes if c["id"] != prev_id] or chimes
        if not options:
            break
        clip = options[rng.randrange(len(options))]
        used[sec] = _event(clip, t, "transition", align="peak")
        prev_id = clip["id"]
    if riser_sec is not None:
        used[riser_sec] = _event(swells[0], riser_at, "riser", align="end")
    events = sorted(used.values(), key=lambda e: e["t"])
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
        a = int(max(float(clip.get("attack_s", clip["peak_s"])), MIN_ATTACK_S) * SR)
        x[:a] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
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


# ───────────────────────── loudness ─────────────────────────

def _k_weight_response(freqs):
    """|H| of the BS.1770 K-weighting (high-shelf pre-filter + RLB high-pass) at 48 kHz."""
    import numpy as np
    w = 2 * np.pi * np.asarray(freqs) / SR
    z = np.exp(-1j * w)
    def biquad(b, a):
        return (b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z)
    h1 = biquad((1.53512485958697, -2.69169618940638, 1.19839281085285),
                (1.0, -1.69065929318241, 0.73248077421585))
    h2 = biquad((1.0, -2.0, 1.0), (1.0, -1.99004745483398, 0.99007225036621))
    return np.abs(h1 * h2)


def integrated_lufs(x, sr: int = SR) -> float:
    """Integrated loudness (LUFS) of mono float samples, ITU-R BS.1770-4 with gating.

    Uses pyloudnorm when installed; otherwise a numpy implementation (K-weighting applied
    as its magnitude response in the frequency domain, 400 ms blocks with 75 % overlap,
    -70 LUFS absolute and -10 LU relative gates). Returns -inf for silence.
    """
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or not np.any(x):
        return float("-inf")
    block = int(0.4 * sr)
    if x.size < block:
        x = np.concatenate([x, np.zeros(block - x.size)])
    try:
        import pyloudnorm  # type: ignore
        val = float(pyloudnorm.Meter(sr).integrated_loudness(x))
        if np.isfinite(val):
            return val
    except Exception:
        pass
    if sr != SR:
        raise ValueError("numpy loudness fallback expects 48 kHz")
    n = int(2 ** np.ceil(np.log2(x.size + 4096)))
    spec = np.fft.rfft(x, n) * _k_weight_response(np.fft.rfftfreq(n, 1.0 / sr))
    y = np.fft.irfft(spec, n)[: x.size]
    hop = block // 4
    starts = range(0, max(1, y.size - block + 1), hop)
    ms = np.array([np.mean(y[s:s + block] ** 2) for s in starts])
    with np.errstate(divide="ignore"):
        lk = -0.691 + 10 * np.log10(ms)
    g = ms[lk > -70.0]
    if g.size == 0:
        return float("-inf")
    rel = -0.691 + 10 * np.log10(np.mean(g)) - 10.0
    g = ms[(lk > -70.0) & (lk > rel)]
    if g.size == 0:
        return float("-inf")
    return float(-0.691 + 10 * np.log10(np.mean(g)))


def render_sfx_track(events: list[dict], duration: float, voice_path: Path, out_path: Path,
                     under_db: float = UNDER_VOICE_LU) -> Path | None:
    """Mix events into a mono 48 kHz WAV, each effect leveled to (voice LUFS - under_db + gain_db).

    Loudness-based, not peak-based. None when empty or when SFX are disabled.
    """
    if not events or not enabled():
        return None
    import numpy as np
    try:
        voice_lufs = integrated_lufs(_decode(voice_path))
    except Exception:
        voice_lufs = float("-inf")
    if not np.isfinite(voice_lufs):
        voice_lufs = FALLBACK_VOICE_LUFS
    n = int(round(duration * SR))
    track = np.zeros(n, dtype=np.float64)
    cache: dict[str, tuple] = {}
    placed = 0
    for ev in events:
        try:
            clip = _clip(ev["clip"])
        except KeyError:
            log(f"SFX: dropping unknown clip {ev.get('clip')!r}")
            continue
        if clip["id"] not in cache:
            x = synth_clip(clip).astype(np.float64)
            pad = np.zeros(SR)                      # measured as placed: surrounded by silence
            cache[clip["id"]] = (x, integrated_lufs(np.concatenate([pad, x, pad])))
        x, clip_lufs = cache[clip["id"]]
        target = voice_lufs - under_db + min(0.0, float(ev.get("gain_db", 0.0)))
        gain = 10 ** ((target - clip_lufs) / 20) if np.isfinite(clip_lufs) else 0.0
        i = int(round(ev["t"] * SR))
        if i >= n:
            continue
        j = min(n, i + len(x))
        track[i:j] += gain * x[: j - i]
        placed += 1
    if not placed:
        return None
    # Overlap guard: wherever effects stack, the mix must still sit under_db below the voice.
    ceiling = voice_lufs - under_db
    worst = float("-inf")
    for ev in events:
        if ev["clip"] in cache:
            i = int(round(ev["t"] * SR))
            seg = track[max(0, i - SR // 2): i + len(cache[ev["clip"]][0]) + SR // 2]
            worst = max(worst, integrated_lufs(seg))
    if np.isfinite(worst) and worst > ceiling + 0.25:
        track *= 10 ** ((ceiling - worst) / 20)
    pcm = (np.clip(track, -1.0, 1.0) * 32767).astype("<i2")
    out_path = Path(out_path)
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    log(f"SFX track: {placed} events at voice {voice_lufs:.1f} LUFS - {under_db:.0f} LU -> {out_path.name}")
    return out_path
