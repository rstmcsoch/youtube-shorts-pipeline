"""Multi-provider TTS — Kokoro-82M (free, local, default for English), Edge TTS (free fallback), ElevenLabs (premium), MiniMax, 60db (Indic + low cost), macOS say (fallback).

Kokoro-82M (Apache 2.0) runs locally on CPU through the lightweight
`kokoro-onnx` package; model + voice files come from the official kokoro-onnx
GitHub release and are cached. It is used for English; any failure falls back
to Edge TTS.
Edge TTS: free, cross-platform, 300+ voices, no API key.
ElevenLabs is premium: most natural, requires API key.
MiniMax is an alternative paid provider with streaming TTS.
60db is an alternative paid provider with native Indic-language voices and a lower per-character cost.
macOS say is the last-resort fallback.
"""

import base64
import os
from pathlib import Path

import requests

from .config import (
    VOICE_ID_EN,
    VOICE_ID_HI,
    get_60db_key,
    get_elevenlabs_key,
    get_minimax_key,
    run_cmd,
)
from .log import log
from .retry import with_retry


# ─────────────────────────────────────────────────────
# Kokoro-82M (Apache 2.0) via kokoro-onnx — free, local CPU
# ─────────────────────────────────────────────────────

KOKORO_RELEASE_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
KOKORO_MODEL_FILE = "kokoro-v1.0.onnx"
KOKORO_VOICES_FILE = "voices-v1.0.bin"
# sha256 of the official release files (verified 2026-10-29).
KOKORO_SHA256 = {
    "kokoro-v1.0.onnx": "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
    "kokoro-v1.0.int8.onnx": "6e742170d309016e5891a994e1ce1559c702a2ccd0075e67ef7157974f6406cb",
    "voices-v1.0.bin": "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
}
# One consistent brand voice. The style guide's voice block wins (am_michael since
# 3 Nov 2026: best ASR intelligibility at 2x-2.5x playback among Kokoro male voices).
KOKORO_DEFAULT_VOICE = "am_michael"
KOKORO_DEFAULT_SPEED = 1.08
# Kokoro is only used for languages whose voice matches the brand voice.
KOKORO_LANGS = {"en": "en-us"}

# Every voice in the official kokoro-onnx voices-v1.0.bin (Kokoro-82M v1.0).
KOKORO_VOICES = frozenset("""
af_alloy af_aoede af_bella af_heart af_jessica af_kore af_nicole af_nova af_river af_sarah af_sky
am_adam am_echo am_eric am_fenrir am_liam am_michael am_onyx am_puck am_santa
bf_alice bf_emma bf_isabella bf_lily bm_daniel bm_fable bm_george bm_lewis
ef_dora em_alex em_santa ff_siwis hf_alpha hf_beta hm_omega hm_psi if_sara im_nicola
jf_alpha jf_gongitsune jf_nezumi jf_tebukuro jm_kumo pf_dora pm_alex pm_santa
zf_xiaobei zf_xiaoni zf_xiaoxiao zf_xiaoyi zm_yunjian zm_yunxi zm_yunxia zm_yunyang
""".split())
KOKORO_MAX_BLEND = 4


def parse_kokoro_voice(value: str):
    """Parse a Kokoro voice id ("am_michael") or blend ("am_michael:0.6,am_adam:0.4").

    Returns a list of (voice_id, weight) with weights normalised to sum 1, or None when
    the value is not a valid Kokoro voice/blend (unknown id, bad/non-positive weight,
    duplicate id, or more than KOKORO_MAX_BLEND parts)."""
    text = str(value or "").strip().lower().replace(" ", "")
    if not text:
        return None
    if ":" not in text and "," not in text:
        return [(text, 1.0)] if text in KOKORO_VOICES else None
    parts = []
    for chunk in text.split(","):
        vid, sep, raw_w = chunk.partition(":")
        if not sep or vid not in KOKORO_VOICES or any(v == vid for v, _ in parts):
            return None
        try:
            w = float(raw_w)
        except ValueError:
            return None
        if not (0 < w <= 1e3) or w != w:
            return None
        parts.append((vid, w))
    if not parts or len(parts) > KOKORO_MAX_BLEND:
        return None
    total = sum(w for _, w in parts)
    return [(v, round(w / total, 4)) for v, w in parts]


def validate_kokoro_voice(value: str) -> str:
    """Canonical voice string for `value`, or the brand voice (am_michael) with a log line."""
    parsed = parse_kokoro_voice(value)
    if parsed is None:
        log(f"DYPOL voice {str(value or '')[:80]!r} is not a valid Kokoro voice or blend; "
            f"falling back to {KOKORO_DEFAULT_VOICE}")
        return KOKORO_DEFAULT_VOICE
    if len(parsed) == 1:
        return parsed[0][0]
    return ",".join(f"{v}:{w:g}" for v, w in parsed)


def kokoro_voice_arg(engine, voice: str):
    """What kokoro-onnx `create(voice=...)` takes: the id itself, or a weighted style
    array for a blend (sum of get_voice_style(id) * weight)."""
    parsed = parse_kokoro_voice(voice) or [(KOKORO_DEFAULT_VOICE, 1.0)]
    if len(parsed) == 1:
        return parsed[0][0]
    import numpy as np
    getter = getattr(engine, "get_voice_style", None) or (lambda name: engine.voices[name])
    return np.sum([np.asarray(getter(v)) * w for v, w in parsed], axis=0)


VO_MIN_SECONDS = 45.0
VO_MAX_SECONDS = 75.0


def speed_for_target(seconds: float, speed: float, lo: float = VO_MIN_SECONDS, hi: float = VO_MAX_SECONDS):
    """Speed that brings `seconds` into [lo, hi] (aiming 2 s inside), or None.

    Returns None when already in range or when the needed change would push
    speed outside 0.95-1.2 (then the script itself is the problem).
    """
    if lo <= seconds <= hi or seconds <= 0:
        return None
    target = lo + 2.0 if seconds < lo else hi - 2.0
    new_speed = round(speed * seconds / target, 3)
    if 0.95 <= new_speed <= KOKORO_MAX_SPEED:
        return new_speed
    return None


# Narration pace (brand target 150-165 WPM, aim 158). Kokoro's real pace depends on
# the text: am_michael is ~156 WPM at 1.08 on plain sentences but only ~116 WPM on a
# dense technical Short (draft job-mw0qn7ms, 3 Nov 2026), so speed is tuned per run.
WPM_MIN, WPM_MAX, WPM_TARGET = 150.0, 165.0, 158.0
KOKORO_MAX_SPEED = 1.4   # am_michael stays clear well past 1.25 (Engine A ships ~1.26)
KOKORO_MIN_SPEED = 0.9


def pace_band() -> tuple:
    """(min, max, target) WPM from the style guide's `pace:` block, else the brand defaults."""
    try:
        from .style import load_style
        v = dict(load_style().get("pace") or {})
        lo = float(v.get("min_wpm", WPM_MIN)); hi = float(v.get("max_wpm", WPM_MAX))
        tg = float(v.get("target_wpm", WPM_TARGET))
    except Exception:  # noqa: BLE001 - a broken style file never blocks the voiceover
        return WPM_MIN, WPM_MAX, WPM_TARGET
    if not (0 < lo <= tg <= hi):
        return WPM_MIN, WPM_MAX, WPM_TARGET
    return lo, hi, tg


def pace_speed(words: int, seconds: float, speed: float, lo: float = WPM_MIN, hi: float = WPM_MAX,
               target: float = WPM_TARGET):
    """Kokoro speed that brings words/minute into [lo, hi] (aiming at `target`), or None.

    None when already in band, when the inputs are empty, or when the clamped speed
    would not change anything."""
    if words <= 0 or seconds <= 0:
        return None
    wpm = words / seconds * 60.0
    if lo <= wpm <= hi:
        return None
    new_speed = round(min(KOKORO_MAX_SPEED, max(KOKORO_MIN_SPEED, speed * target / wpm)), 3)
    if abs(new_speed - speed) < 0.01:
        return None
    return new_speed


VO_MAX_SILENCE_PAD = 1.5  # seconds of trailing silence instead of a full re-render


def secant_speed(p0: tuple, p1: tuple, lo: float = WPM_MIN, hi: float = WPM_MAX, target: float = WPM_TARGET):
    """Speed for `target` WPM interpolated from two measured (speed, wpm) renders, or None when
    p1 is already in band. Falls back to the proportional rule when the two points are flat."""
    (s0, w0), (s1, w1) = p0, p1
    if lo <= w1 <= hi or w1 <= 0:
        return None
    if abs(w1 - w0) > 1e-6 and abs(s1 - s0) > 1e-6:
        new = s1 + (target - w1) * (s1 - s0) / (w1 - w0)
    else:
        new = s1 * target / w1
    new = round(min(KOKORO_MAX_SPEED, max(KOKORO_MIN_SPEED, new)), 3)
    return None if abs(new - s1) < 0.01 else new


def short_padding_seconds(seconds: float, lo: float = VO_MIN_SECONDS, max_pad: float = VO_MAX_SILENCE_PAD) -> float:
    """Trailing silence that lifts a voiceover just under the window into it, else 0.

    A 44.9 s render needs 0.35 s of tail silence, not a second ~20 s Kokoro pass
    at a different speed; the spoken audio is untouched.
    """
    if seconds <= 0 or seconds >= lo or lo - seconds > max_pad:
        return 0.0
    return round(lo - seconds + 0.25, 3)


def lead_silence_max() -> float:
    try:
        return max(0.0, float(_style_voice().get("lead_silence_max_s", 0.12)))
    except (TypeError, ValueError):
        return 0.12


def lead_trim_start(samples, sample_rate: int, keep_s: float = 0.12, threshold: float = 0.01) -> int:
    """Samples trim_lead_silence() drops from the front (word timings shift by this)."""
    import numpy as np
    arr = np.asarray(samples)
    if not len(arr):
        return 0
    loud = np.flatnonzero(np.abs(arr.astype("float32")) > threshold)
    if not len(loud):
        return 0
    return max(0, int(loud[0]) - int(round(keep_s * sample_rate)))


def trim_lead_silence(samples, sample_rate: int, keep_s: float = 0.12, threshold: float = 0.01):
    """Drop leading near-silence so at most `keep_s` remains before the first word
    (the hook's first word must land by 0.8 s). Returns the samples unchanged when
    the audio never rises above the threshold."""
    import numpy as np
    arr = np.asarray(samples)
    if not len(arr):
        return samples
    start = lead_trim_start(arr, sample_rate, keep_s, threshold)
    return arr[start:] if start else samples


def kokoro_model_dir() -> Path:
    default = Path.home() / ".cache" / "kokoro-onnx"
    return Path(os.environ.get("KOKORO_MODEL_DIR", str(default))).expanduser()


def _style_voice() -> dict:
    try:
        from .style import voice_settings
        return voice_settings()
    except Exception:  # noqa: BLE001 - a broken style file never blocks the voiceover
        return {}


def kokoro_settings() -> dict:
    """Resolve the Kokoro voice/speed, clamped to a sane range.

    Order: $DYPOL_VOICE / $DYPOL_VOICE_SPEED (workflow: input voice -> repo variable
    DYPOL_DEFAULT_VOICE -> am_michael), then the style guide's `voice:` block (the
    brand voice, am_michael at 1.08 since 3 Nov 2026), then the legacy $KOKORO_VOICE /
    $KOKORO_SPEED, then the module defaults. The voice may be a Kokoro id or a blend
    "a:0.6,b:0.4"; anything invalid logs a line and falls back to am_michael."""
    style = _style_voice()
    voice = (os.environ.get("DYPOL_VOICE", "") or str(style.get("kokoro_voice") or "")
             or os.environ.get("KOKORO_VOICE", "") or KOKORO_DEFAULT_VOICE).strip()
    raw_speed = (os.environ.get("DYPOL_VOICE_SPEED", "") or str(style.get("kokoro_speed") or "")
                 or os.environ.get("KOKORO_SPEED", "") or KOKORO_DEFAULT_SPEED)
    try:
        speed = float(raw_speed)
    except ValueError:
        speed = KOKORO_DEFAULT_SPEED
    speed = min(1.3, max(0.8, speed))
    voice = validate_kokoro_voice(voice)
    model = (os.environ.get("KOKORO_MODEL_FILE", "") or KOKORO_MODEL_FILE).strip()
    return {"voice": voice, "speed": speed, "model_file": model}


def _sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_kokoro_files(model_file: str = KOKORO_MODEL_FILE) -> tuple[Path, Path]:
    """Return (model, voices) paths, downloading from the official release if missing."""
    model_dir = kokoro_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in (model_file, KOKORO_VOICES_FILE):
        path = model_dir / name
        if not path.exists() or path.stat().st_size == 0:
            if os.environ.get("KOKORO_AUTO_DOWNLOAD", "1").strip() != "1":
                raise RuntimeError(f"Kokoro file missing: {path}")
            log(f"Downloading Kokoro file {name}...")
            tmp = path.with_suffix(path.suffix + ".part")
            with requests.get(f"{KOKORO_RELEASE_BASE}/{name}", stream=True, timeout=(10, 120)) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            tmp.replace(path)
        expected = KOKORO_SHA256.get(name)
        if expected and _sha256_file(path) != expected:
            path.unlink(missing_ok=True)
            raise RuntimeError(f"Kokoro file checksum mismatch: {name}")
        paths.append(path)
    return paths[0], paths[1]


# ── timestamped Kokoro: same voice, plus per-phoneme durations for exact word timings ──
# onnx-community/Kokoro-82M-v1.0-ONNX-timestamped (Apache-2.0), onnx/model.onnx at a pinned
# revision. Outputs waveform + durations, so captions and motion cues land on the spoken
# word (+-80 ms) instead of Whisper base's estimate. The workflow downloads and verifies it;
# without it (or when alignment fails) the plain model renders and Whisper times the words.
KOKORO_TIMED_REPO = "onnx-community/Kokoro-82M-v1.0-ONNX-timestamped"
KOKORO_TIMED_REVISION = "dd4401a9add81ac692d20e240d22ec9dda82cc29"
KOKORO_TIMED_FILE = "kokoro-v1.0-timestamped.onnx"
KOKORO_TIMED_URL = (f"https://huggingface.co/{KOKORO_TIMED_REPO}/resolve/{KOKORO_TIMED_REVISION}"
                    "/onnx/model.onnx")
KOKORO_TIMED_SIZE = 325_532_171
KOKORO_SHA256[KOKORO_TIMED_FILE] = "651ea8291843a92276a4a003581a215cb07d15e47dde6fcfb1b768f9a1682054"
WORD_TIMINGS_FILE = "voiceover_words.json"
ALIGN_MIN_MATCHED = 0.8   # share of script words that must find their phonemes, else Whisper


def kokoro_timed_model_path() -> Path:
    """$KOKORO_TIMED_MODEL (a file path), else <model dir>/kokoro-v1.0-timestamped.onnx."""
    env = os.environ.get("KOKORO_TIMED_MODEL", "").strip()
    return Path(env).expanduser() if env else kokoro_model_dir() / KOKORO_TIMED_FILE


def ensure_kokoro_timed_model() -> Path | None:
    """The verified timestamped model, or None (with a log line) when it is missing or bad.

    Downloaded by the workflow; the code only downloads it itself when
    KOKORO_TIMED_AUTO_DOWNLOAD=1, so a missing file never stalls a run on 325 MB."""
    path = kokoro_timed_model_path()
    if os.environ.get("KOKORO_TIMED", "1").strip() == "0":
        return None
    if not path.exists() or path.stat().st_size == 0:
        if os.environ.get("KOKORO_TIMED_AUTO_DOWNLOAD", "0").strip() != "1":
            log(f"Timestamped Kokoro model not found at {path}; word timings fall back to Whisper")
            return None
        try:
            log(f"Downloading timestamped Kokoro model ({KOKORO_TIMED_REPO}@{KOKORO_TIMED_REVISION[:7]})...")
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".part")
            with requests.get(KOKORO_TIMED_URL, stream=True, timeout=(10, 300)) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            tmp.replace(path)
        except Exception as exc:  # noqa: BLE001 - the plain model + Whisper still work
            log(f"Timestamped Kokoro model download failed ({exc}); word timings fall back to Whisper")
            return None
    if path.stat().st_size != KOKORO_TIMED_SIZE or _sha256_file(path) != KOKORO_SHA256[KOKORO_TIMED_FILE]:
        log(f"Timestamped Kokoro model at {path} failed its size/SHA-256 check; "
            "word timings fall back to Whisper")
        return None
    return path


_PHONE_SKIP = set("ˈˌː.,!?;:—–-…\"'“”‘’()[]¡¿ \t\n")


def _phone_chars(text: str) -> str:
    """Phoneme string reduced to its sounds: stress, length, punctuation and spaces dropped."""
    return "".join(c for c in str(text) if c not in _PHONE_SKIP)


def map_phoneme_timings(script: str, timings, phonemize) -> list[dict] | None:
    """Script words timed from per-phoneme timings, aligned by pronunciation.

    `timings` is the model's phoneme stream ((phoneme, start, end) tuples or objects with
    .phoneme/.start/.end); `phonemize(word)` gives one script word's phonemes. Each word's
    own pronunciation is matched against the stream, so "2024" (spoken "two thousand
    twenty four", four phoneme words) owns all of those phonemes rather than one
    space-separated slot. Words the matcher misses are spread between their timed
    neighbours. Returns [{"word", "start", "end"}] in script spelling, or None when fewer
    than ALIGN_MIN_MATCHED of the words matched (the caller then falls back to Whisper)."""
    import difflib
    words = str(script or "").split()
    stream, stress_at = [], None
    for t in timings or []:
        ph, st, en = (t.phoneme, t.start, t.end) if hasattr(t, "phoneme") else t
        if ph and all(c in "ˈˌ" for c in ph):   # a stress mark is the onset of the sound it precedes
            stress_at = float(st) if stress_at is None else stress_at
            continue
        sounds = _phone_chars(ph)
        if not sounds:
            if ph and ph != "ː":
                stress_at = None
            continue
        for c in sounds:
            stream.append((c, float(st) if stress_at is None else stress_at, float(en)))
            stress_at = None
    if not words or not stream:
        return None
    ref, owner = [], []
    for i, w in enumerate(words):
        try:
            p = _phone_chars(phonemize(w))
        except Exception:  # noqa: BLE001 - one odd token never sinks the alignment
            p = ""
        ref.extend(p)
        owner.extend([i] * len(p))
    if not ref:
        return None
    hit: dict[int, list[tuple[int, int]]] = {}   # word -> [(ref offset in word, stream index)]
    first_ref = {}
    for k, o in enumerate(owner):
        first_ref.setdefault(o, k)
    sm = difflib.SequenceMatcher(a=ref, b=[c for c, _, _ in stream], autojunk=False)
    for blk in sm.get_matching_blocks():
        for n in range(blk.size):
            k, j = blk.a + n, blk.b + n
            hit.setdefault(owner[k], []).append((k - first_ref[owner[k]], j))
    if len(hit) < ALIGN_MIN_MATCHED * len(words):
        return None
    spans: list[tuple[float, float] | None] = [None] * len(words)
    last_j = -1
    for i in range(len(words)):
        pairs = hit.get(i)
        if not pairs:
            continue
        off, j0 = pairs[0]
        j_start = max(last_j + 1, j0 - off, 0)     # unmatched leading phonemes still belong to the word
        j_start = min(j_start, j0)
        j_end = pairs[-1][1]
        spans[i] = (stream[j_start][1], stream[j_end][2])
        last_j = j_end
    out, prev_start = [], 0.0
    for i, w in enumerate(words):
        if spans[i] is None:   # spread missed words over the gap between timed neighbours
            a = i
            while a > 0 and spans[a - 1] is None:
                a -= 1
            b = i
            while b + 1 < len(words) and spans[b + 1] is None:
                b += 1
            lo = spans[a - 1][1] if a > 0 else 0.0
            hi = spans[b + 1][0] if b + 1 < len(words) else max(lo, stream[-1][2])
            step = max(0.0, hi - lo) / (b - a + 1)
            spans[i] = (lo + (i - a) * step, lo + (i - a + 1) * step)
        st, en = spans[i]
        st = max(st, prev_start)
        en = max(en, st + 0.02)
        out.append({"word": w, "start": round(st, 3), "end": round(en, 3)})
        prev_start = st
    return out


def write_word_timings(out_dir: Path, audio_name: str, words: list[dict], source: str) -> Path:
    import json
    path = Path(out_dir) / WORD_TIMINGS_FILE
    path.write_text(json.dumps({"audio": audio_name, "source": source, "words": words}, indent=1),
                    encoding="utf-8")
    return path


def load_word_timings(audio_path: Path) -> list[dict]:
    """Word timings TTS wrote for exactly this audio file (timestamped Kokoro), else []."""
    import json
    path = Path(audio_path).parent / WORD_TIMINGS_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict) or data.get("audio") != Path(audio_path).name:
        return []
    words = [w for w in data.get("words") or [] if isinstance(w, dict)
             and {"word", "start", "end"} <= set(w)]
    return words


def _clear_word_timings(out_dir: Path) -> None:
    try:
        (Path(out_dir) / WORD_TIMINGS_FILE).unlink(missing_ok=True)
    except OSError:
        pass


def kokoro_create_timed(engine, script: str, voice_arg, speed: float, lang: str):
    """(audio, rate, phoneme timings) rendered exactly like Kokoro.create() renders today.

    kokoro-onnx's create_timed() also splices extra silence after every mark once timings
    exist, and its splice point can sit after the next phoneme's start, which moved word
    timings ~100 ms late on a local sample. The same batches without that splice give the
    audio create() gives (pauses only at batch joins) with timings that stay on the audio."""
    prepare = getattr(engine, "_prepare", None)
    batches_fn = getattr(engine, "_create_batches", None)
    if prepare is None or batches_fn is None:   # another kokoro-onnx layout: the public API
        return engine.create_timed(script, voice=voice_arg, speed=speed, lang=lang)
    voice, _, batches = prepare(script, voice_arg, speed, lang, False, 0.25, 0.1)
    audio, spoken = batches_fn(batches, voice, speed, True)
    try:
        from kokoro_onnx.config import SAMPLE_RATE as rate
    except ImportError:
        rate = 24000
    return audio, rate, spoken


def _timed_engine(Kokoro, voices_path):
    """A Kokoro engine on the timestamped model, or None (logged) when unusable."""
    path = ensure_kokoro_timed_model()
    if path is None:
        return None
    try:
        engine = Kokoro(str(path), str(voices_path))
        outs = [o.name for o in engine.sess.get_outputs()]
        if len(outs) < 2:
            log(f"Timestamped Kokoro model reports no durations ({outs}); word timings fall back to Whisper")
            return None
        # kokoro-onnx looks for an output named "duration"; this export calls it "durations"
        engine.has_timings = True
        return engine
    except Exception as exc:  # noqa: BLE001 - the plain model still renders the voice
        log(f"Timestamped Kokoro model failed to load ({exc}); word timings fall back to Whisper")
        return None


def _write_wav(samples, sample_rate: int, out_path: Path) -> None:
    """Write float samples [-1, 1] as 16-bit mono PCM WAV (stdlib only)."""
    import wave
    import numpy as np
    pcm = (np.clip(np.asarray(samples, dtype="float32"), -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())


def _generate_kokoro(script: str, out_dir: Path, lang: str) -> Path:
    """Generate voiceover locally with Kokoro-82M (kokoro-onnx, CPU)."""
    kokoro_lang = KOKORO_LANGS.get(lang[:2])
    if not kokoro_lang:
        raise RuntimeError(f"Kokoro brand voice not configured for language '{lang}'")
    if not str(script or "").strip():
        raise RuntimeError("Kokoro received an empty script")
    from kokoro_onnx import Kokoro

    cfg = kokoro_settings()
    if kokoro_lang == "en-us" and cfg["voice"][:1] == "b":  # bf_/bm_ voices are British English
        kokoro_lang = "en-gb"
    model_path, voices_path = ensure_kokoro_files(cfg["model_file"])
    log(f"Generating {lang} voiceover via Kokoro-82M (voice: {cfg['voice']}, speed: {cfg['speed']})...")
    timed = _timed_engine(Kokoro, voices_path)
    engine = timed or Kokoro(str(model_path), str(voices_path))
    if timed is not None:
        log("Kokoro: timestamped model (per-phoneme durations) renders the voice and times the words")
    voice_arg = kokoro_voice_arg(engine, cfg["voice"])
    state = {"timings": []}

    def render(speed):
        if timed is not None:
            audio, rate, state["timings"] = kokoro_create_timed(timed, script, voice_arg, speed, kokoro_lang)
            return audio, rate
        return engine.create(script, voice=voice_arg, speed=speed, lang=kokoro_lang)

    samples, sample_rate = render(cfg["speed"])
    # Keep the voiceover inside the 45-75 s window when a small speed change can
    # do it (one re-render at most, speed stays within 0.95-1.2).
    n_words = max(1, len(str(script).split()))
    if samples is not None and len(samples):
        seconds = len(samples) / float(sample_rate)
        lo_wpm, hi_wpm, tg_wpm = pace_band()
        paced = pace_speed(n_words, seconds, cfg["speed"], lo_wpm, hi_wpm, tg_wpm)
        if paced is not None:
            log(f"Kokoro pace {n_words / seconds * 60:.0f} WPM outside {lo_wpm:.0f}-{hi_wpm:.0f}; "
                f"re-rendering at speed {paced}")
            first = (cfg["speed"], n_words / seconds * 60.0)
            samples, sample_rate = render(paced)
            cfg["speed"] = paced
            seconds = len(samples) / float(sample_rate)
            # Kokoro's pace is not linear in speed (draft job-mw0rja3o: 117 WPM at 1.08 -> 170 WPM at
            # 1.4), so one more render interpolates between the two measured points if needed.
            again = secant_speed(first, (paced, n_words / seconds * 60.0), lo_wpm, hi_wpm, tg_wpm)
            if again is not None:
                log(f"Kokoro pace {n_words / seconds * 60:.0f} WPM at speed {paced} still outside "
                    f"{lo_wpm:.0f}-{hi_wpm:.0f}; re-rendering at speed {again}")
                samples, sample_rate = render(again)
                cfg["speed"] = again
                seconds = len(samples) / float(sample_rate)
        pad = short_padding_seconds(seconds)
        if pad:
            import numpy as np
            log(f"Kokoro voiceover {seconds:.1f}s just under {VO_MIN_SECONDS}s; adding {pad:.2f}s tail silence "
                "instead of re-rendering")
            samples = np.concatenate([np.asarray(samples), np.zeros(int(round(pad * sample_rate)),
                                                                     dtype=np.asarray(samples).dtype)])
            seconds = len(samples) / float(sample_rate)
        new_speed = speed_for_target(seconds, cfg["speed"])
        if new_speed is not None and new_speed < cfg["speed"]:
            # Slowing down to fill the window must not drop the pace under the band;
            # a short script is the script's problem, not the voice's.
            est_wpm = n_words / seconds * 60.0 * new_speed / cfg["speed"]
            if est_wpm < pace_band()[0]:
                log(f"Kokoro voiceover {seconds:.1f}s is short, but slowing to {new_speed} would drop the "
                    f"pace to {est_wpm:.0f} WPM; keeping speed {cfg['speed']}")
                new_speed = None
        if new_speed is not None:
            log(f"Kokoro voiceover {seconds:.1f}s outside {VO_MIN_SECONDS}-{VO_MAX_SECONDS}s; re-rendering at speed {new_speed}")
            samples, sample_rate = render(new_speed)
            cfg["speed"] = new_speed
    if samples is None or len(samples) < int(sample_rate) * 0.5:
        raise RuntimeError("Kokoro produced no usable audio")
    lead = lead_trim_start(samples, sample_rate, lead_silence_max())
    if lead:
        import numpy as np
        samples = np.asarray(samples)[lead:]
    # Sanity check: spoken duration must roughly match the text length
    # (~1.2-4.5 words per second). Catches silent / truncated renders.
    words = max(1, len(str(script).split()))
    seconds = len(samples) / float(sample_rate)
    wps = words / seconds
    if not (1.2 <= wps <= 4.5):
        raise RuntimeError(f"Kokoro audio duration implausible ({seconds:.1f}s for {words} words)")
    out_path = out_dir / f"voiceover_{lang}.wav"
    _write_wav(samples, sample_rate, out_path)
    if timed is not None:
        word_timings = None
        try:
            cache: dict[str, str] = {}

            def phon(w):
                if w not in cache:
                    cache[w] = engine.tokenizer.phonemize(w, kokoro_lang)
                return cache[w]
            shift = lead / float(sample_rate)
            shifted = []
            for t in state["timings"]:
                ph, st, en = (t.phoneme, t.start, t.end) if hasattr(t, "phoneme") else t
                shifted.append((ph, float(st) - shift, float(en) - shift))
            word_timings = map_phoneme_timings(script, shifted, phon)
        except Exception as exc:  # noqa: BLE001 - Whisper still times the words
            log(f"Kokoro phoneme alignment error: {exc}")
        if word_timings:
            write_word_timings(out_dir, out_path.name, word_timings, "kokoro-timestamped")
            log(f"Kokoro word timings: {len(word_timings)} words from phoneme durations (Whisper not needed)")
        else:
            log("Kokoro phoneme-to-word alignment failed; captions fall back to Whisper word timings")
    _write_voice_meta(out_dir, "kokoro", cfg["voice"], cfg["speed"],
                      KOKORO_TIMED_FILE if timed is not None else cfg["model_file"])
    log(f"Kokoro voiceover saved: {out_path.name} ({seconds:.1f}s, {wps:.2f} words/s, {wps * 60:.0f} WPM, "
        f"speed {cfg['speed']})")
    return out_path


def _write_voice_meta(out_dir: Path, provider: str, voice: str, speed=None, model: str = "") -> None:
    """Record which TTS provider/voice actually produced the voiceover."""
    import json
    try:
        (Path(out_dir) / "voiceover_meta.json").write_text(json.dumps({
            "provider": provider, "voice": voice, "speed": speed, "model": model,
            "license": "Apache-2.0 (Kokoro-82M)" if provider == "kokoro" else "",
        }, indent=2), encoding="utf-8")
    except OSError:
        pass


# ─────────────────────────────────────────────────────
# Edge TTS — free, cross-platform, 300+ voices
# ─────────────────────────────────────────────────────

# Default Edge TTS voices per language
EDGE_VOICES = {
    "en": "en-US-GuyNeural",
    "hi": "hi-IN-MadhurNeural",
    "es": "es-MX-JorgeNeural",
    "pt": "pt-BR-AntonioNeural",
    "de": "de-DE-ConradNeural",
    "fr": "fr-FR-HenriNeural",
    "ja": "ja-JP-KeitaNeural",
    "ko": "ko-KR-InJoonNeural",
}


async def _edge_tts_generate(text: str, voice: str, output_path: Path):
    """Generate audio via edge-tts (async)."""
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(output_path))


def _generate_edge_tts(script: str, out_dir: Path, lang: str, voice_override: str = "") -> Path:
    """Generate voiceover via Edge TTS (free Microsoft voices)."""
    import asyncio

    voice = voice_override or EDGE_VOICES.get(lang[:2], EDGE_VOICES["en"])
    out_path = out_dir / f"voiceover_{lang}.mp3"

    log(f"Generating {lang} voiceover via Edge TTS (voice: {voice})...")

    try:
        # Handle event loop — works whether called from sync or async context
        try:
            loop = asyncio.get_running_loop()
            # Already in an async context, create a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    _edge_tts_generate(script, voice, out_path)
                )
                future.result(timeout=60)
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            asyncio.run(_edge_tts_generate(script, voice, out_path))

        log(f"Edge TTS voiceover saved: {out_path.name}")
        _write_voice_meta(out_dir, "edge", voice)
        return out_path
    except Exception as e:
        raise RuntimeError(f"Edge TTS failed: {e}")


# ─────────────────────────────────────────────────────
# ElevenLabs — premium, most natural
# ─────────────────────────────────────────────────────

@with_retry(max_retries=3, base_delay=2.0)
def _call_elevenlabs(script: str, voice_id: str, api_key: str, settings: dict | None = None) -> bytes:
    """Call ElevenLabs TTS API and return audio bytes."""
    voice_settings = settings or {
        "stability": 0.4,
        "similarity_boost": 0.85,
        "style": 0.3,
        "use_speaker_boost": True,
    }
    r = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        json={
            "text": script,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": voice_settings,
        },
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"ElevenLabs {r.status_code}: {r.text[:200]}")
    return r.content


def _generate_elevenlabs(
    script: str, out_dir: Path, lang: str,
    voice_id: str = "", settings: dict | None = None
) -> Path:
    """Generate voiceover via ElevenLabs."""
    api_key = get_elevenlabs_key()
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY not set")

    vid = voice_id or (VOICE_ID_HI if lang == "hi" else VOICE_ID_EN)
    out_path = out_dir / f"voiceover_{lang}.mp3"

    log(f"Generating {lang} voiceover via ElevenLabs (voice: {vid})...")
    audio_bytes = _call_elevenlabs(script, vid, api_key, settings)
    out_path.write_bytes(audio_bytes)
    log(f"ElevenLabs voiceover saved: {out_path.name}")
    return out_path


# ─────────────────────────────────────────────────────
# MiniMax TTS — AI-powered, supports streaming SSE
# ─────────────────────────────────────────────────────

MINIMAX_TTS_VOICES = [
    "English_Graceful_Lady",
    "English_Insightful_Speaker",
    "English_radiant_girl",
    "English_Persuasive_Man",
    "English_Lucky_Robot",
    "English_expressive_narrator",
]


@with_retry(max_retries=3, base_delay=2.0)
def _call_minimax_tts(text: str, voice_id: str, api_key: str, model: str = "speech-2.8-hd") -> bytes:
    """Call MiniMax TTS API (streaming SSE) and return mp3 audio bytes."""
    base_url = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io")
    base_url = base_url.rstrip("/").removesuffix("/v1")

    r = requests.post(
        f"{base_url}/v1/t2a_v2",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "text": text,
            "stream": True,
            "voice_setting": {
                "voice_id": voice_id,
                "speed": 1,
                "vol": 1,
                "pitch": 0,
            },
            "audio_setting": {
                "sample_rate": 32000,
                "bitrate": 128000,
                "format": "mp3",
                "channel": 1,
            },
        },
        stream=True,
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"MiniMax TTS {r.status_code}: {r.text[:200]}")

    audio_chunks: list[bytes] = []
    buffer = ""
    for raw in r.iter_content(chunk_size=None):
        if not raw:
            continue
        buffer += raw.decode("utf-8", errors="replace")
        lines = buffer.split("\n")
        buffer = lines.pop()
        for line in lines:
            if not line.startswith("data:"):
                continue
            json_str = line[5:].strip()
            if not json_str or json_str == "[DONE]":
                continue
            try:
                import json as _json
                event_data = _json.loads(json_str)
                audio_hex = event_data.get("data", {}).get("audio")
                if audio_hex:
                    audio_chunks.append(bytes.fromhex(audio_hex))
            except Exception:
                pass

    if not audio_chunks:
        raise RuntimeError("MiniMax TTS returned no audio data")
    return b"".join(audio_chunks)


def _generate_minimax(
    script: str, out_dir: Path, lang: str,
    voice_id: str = "", model: str = "speech-2.8-hd",
) -> Path:
    """Generate voiceover via MiniMax TTS."""
    api_key = get_minimax_key()
    if not api_key:
        raise RuntimeError("MINIMAX_API_KEY not set")

    vid = voice_id or MINIMAX_TTS_VOICES[0]
    out_path = out_dir / f"voiceover_{lang}.mp3"

    log(f"Generating {lang} voiceover via MiniMax TTS (voice: {vid})...")
    audio_bytes = _call_minimax_tts(script, vid, api_key, model)
    out_path.write_bytes(audio_bytes)
    log(f"MiniMax TTS voiceover saved: {out_path.name}")
    return out_path


# ─────────────────────────────────────────────────────
# 60db — Indic-language native, low cost
# ─────────────────────────────────────────────────────

# Documented default voice — "Zara" (Hindi female) per /default-voices.
VOICE_ID_60DB_DEFAULT = "fbb75ed2-975a-40c7-9e06-38e30524a9a1"


@with_retry(max_retries=3, base_delay=2.0)
def _call_60db(script: str, voice_id: str, api_key: str, settings: dict | None = None) -> bytes:
    """Call 60db /tts-synthesize and return raw audio bytes.

    Native 60db parameter ranges (per https://docs.60db.ai/api-reference/tts/text-to-speech):
        stability:  0..100 (lower = more expressive)
        similarity: 0..100 (voice match fidelity)
        speed:      0.5..2.0
    """
    s = settings or {}
    payload = {
        "text": script,
        "voice_id": voice_id,
        "enhance": bool(s.get("enhance", True)),
        "speed": float(s.get("speed", 1.0)),
        "stability": int(s.get("stability", 50)),
        "similarity": int(s.get("similarity", 75)),
        "output_format": "mp3",  # pinned — captions.py / assemble.py expect MP3
    }
    r = requests.post(
        "https://api.60db.ai/tts-synthesize",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=payload,
        timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(f"60db {r.status_code}: {r.text[:200]}")

    body = r.json()
    if not body.get("success", True) or not body.get("audio_base64"):
        raise RuntimeError(f"60db returned no audio: {body.get('message', 'unknown')}")
    try:
        return base64.b64decode(body["audio_base64"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"60db audio_base64 decode failed: {exc}") from exc


def _generate_60db(
    script: str, out_dir: Path, lang: str,
    voice_id: str = "", settings: dict | None = None
) -> Path:
    """Generate voiceover via 60db."""
    api_key = get_60db_key()
    if not api_key:
        raise RuntimeError("SIXTYDB_API_KEY not set")

    vid = voice_id or VOICE_ID_60DB_DEFAULT
    out_path = out_dir / f"voiceover_{lang}.mp3"

    log(f"Generating {lang} voiceover via 60db (voice: {vid})...")
    audio_bytes = _call_60db(script, vid, api_key, settings)
    out_path.write_bytes(audio_bytes)
    log(f"60db voiceover saved: {out_path.name}")
    return out_path


# ─────────────────────────────────────────────────────
# macOS say — last resort fallback
# ─────────────────────────────────────────────────────

def _generate_say(script: str, out_dir: Path) -> Path:
    """macOS 'say' fallback TTS."""
    out_path = out_dir / "voiceover_say.aiff"
    mp3_path = out_dir / "voiceover_say.mp3"
    run_cmd(["say", "-o", str(out_path), script])
    run_cmd([
        "ffmpeg", "-i", str(out_path), "-acodec", "libmp3lame",
        str(mp3_path), "-y", "-loglevel", "quiet",
    ])
    return mp3_path


# ─────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────

def get_tts_provider(name: str | None = None) -> str:
    """Resolve which TTS provider to use.

    Priority: explicit name > TTS_PROVIDER env > auto-detect.
    Auto-detect tries: kokoro > edge_tts > minimax > elevenlabs > 60db > say.
    """
    if name and name != "auto":
        return name.lower()

    from_env = os.environ.get("TTS_PROVIDER", "").lower()
    if from_env:
        return from_env

    from .config import load_config
    from_cfg = load_config().get("TTS_PROVIDER", "").lower()
    if from_cfg:
        return from_cfg

    # Auto-detect: Kokoro first (free, local), then Edge TTS
    try:
        import kokoro_onnx  # noqa: F401
        return "kokoro"
    except ImportError:
        pass
    try:
        import edge_tts  # noqa: F401
        return "edge"
    except ImportError:
        pass

    if get_minimax_key():
        return "minimax"

    if get_elevenlabs_key():
        return "elevenlabs"

    if get_60db_key():
        return "60db"

    # macOS say as last resort
    import shutil
    if shutil.which("say"):
        return "say"

    raise RuntimeError(
        "No TTS provider available. Install one:\n"
        "  pip install edge-tts  (free, recommended)\n"
        "  Set MINIMAX_API_KEY (AI-powered)\n"
        "  Set ELEVENLABS_API_KEY (premium)\n"
        "  Or use macOS (has built-in 'say')"
    )


def generate_voiceover(
    script: str,
    out_dir: Path,
    lang: str = "en",
    provider: str | None = None,
    voice_config: dict | None = None,
) -> Path:
    """Generate voiceover via the configured TTS provider.

    Args:
        script: The voiceover text.
        out_dir: Directory to save the audio file.
        lang: Language code (en, hi, es, etc.).
        provider: TTS provider name (edge, elevenlabs, say).
        voice_config: Optional voice config from niche profile.

    Returns:
        Path to the generated audio file.
    """
    provider = get_tts_provider(provider)
    voice_config = voice_config or {}
    _clear_word_timings(out_dir)  # word timings only ever describe the audio written by this call

    if provider == "kokoro":
        if lang[:2] in KOKORO_LANGS:
            try:
                return _generate_kokoro(script, out_dir, lang)
            except Exception as e:
                log(f"Kokoro TTS failed: {e}; falling back to Edge TTS")
        else:
            log(f"Kokoro brand voice is English-only; using Edge TTS for {lang}")
        provider = "edge"

    if provider == "edge":
        voice_override = str(voice_config.get("voice_id", "") or "")
        if voice_override and "Neural" not in voice_override:
            voice_override = ""  # not an Edge voice name (e.g. a Kokoro/other id)
        try:
            return _generate_edge_tts(script, out_dir, lang, voice_override)
        except Exception as e:
            log(f"Edge TTS failed: {e}")
            # Fall through to next provider
            if get_minimax_key():
                log("Falling back to MiniMax TTS...")
                provider = "minimax"
            elif get_elevenlabs_key():
                log("Falling back to ElevenLabs...")
                provider = "elevenlabs"
            elif get_60db_key():
                log("Falling back to 60db...")
                provider = "60db"
            else:
                log("Falling back to macOS say...")
                return _generate_say(script, out_dir)

    if provider == "minimax":
        try:
            return _generate_minimax(
                script, out_dir, lang,
                voice_id=voice_config.get("voice_id", ""),
                model=voice_config.get("model", "speech-2.8-hd"),
            )
        except Exception as e:
            log(f"MiniMax TTS failed: {e}")
            if get_elevenlabs_key():
                log("Falling back to ElevenLabs...")
                provider = "elevenlabs"
            elif get_60db_key():
                log("Falling back to 60db...")
                provider = "60db"
            else:
                log("Falling back to macOS say...")
                return _generate_say(script, out_dir)

    if provider == "elevenlabs":
        try:
            return _generate_elevenlabs(
                script, out_dir, lang,
                voice_id=voice_config.get("voice_id", ""),
                settings=voice_config.get("settings"),
            )
        except Exception as e:
            log(f"ElevenLabs failed: {e}")
            if get_60db_key():
                log("Falling back to 60db...")
                provider = "60db"
            else:
                log("Falling back to macOS say...")
                return _generate_say(script, out_dir)

    if provider == "60db":
        try:
            return _generate_60db(
                script, out_dir, lang,
                voice_id=voice_config.get("voice_id", ""),
                settings=voice_config.get("settings"),
            )
        except Exception as e:
            log(f"60db failed: {e}")
            log("Falling back to macOS say...")
            return _generate_say(script, out_dir)

    if provider == "say":
        return _generate_say(script, out_dir)

    raise ValueError(f"Unknown TTS provider: {provider}")
