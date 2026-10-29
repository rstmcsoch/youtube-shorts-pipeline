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
# One consistent brand voice. af_heart is the highest-graded Kokoro voice.
KOKORO_DEFAULT_VOICE = "af_heart"
KOKORO_DEFAULT_SPEED = 1.08
# Kokoro is only used for languages whose voice matches the brand voice.
KOKORO_LANGS = {"en": "en-us"}


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
    if 0.95 <= new_speed <= 1.2:
        return new_speed
    return None


def kokoro_model_dir() -> Path:
    default = Path.home() / ".cache" / "kokoro-onnx"
    return Path(os.environ.get("KOKORO_MODEL_DIR", str(default))).expanduser()


def kokoro_settings() -> dict:
    """Resolve the Kokoro voice/speed from env, clamped to a sane range."""
    voice = (os.environ.get("KOKORO_VOICE", "") or KOKORO_DEFAULT_VOICE).strip()
    try:
        speed = float(os.environ.get("KOKORO_SPEED", "") or KOKORO_DEFAULT_SPEED)
    except ValueError:
        speed = KOKORO_DEFAULT_SPEED
    speed = min(1.3, max(0.8, speed))
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
    model_path, voices_path = ensure_kokoro_files(cfg["model_file"])
    log(f"Generating {lang} voiceover via Kokoro-82M (voice: {cfg['voice']}, speed: {cfg['speed']})...")
    engine = Kokoro(str(model_path), str(voices_path))
    samples, sample_rate = engine.create(script, voice=cfg["voice"], speed=cfg["speed"], lang=kokoro_lang)
    # Keep the voiceover inside the 45-75 s window when a small speed change can
    # do it (one re-render at most, speed stays within 0.95-1.2).
    if samples is not None and len(samples):
        seconds = len(samples) / float(sample_rate)
        new_speed = speed_for_target(seconds, cfg["speed"])
        if new_speed is not None:
            log(f"Kokoro voiceover {seconds:.1f}s outside {VO_MIN_SECONDS}-{VO_MAX_SECONDS}s; re-rendering at speed {new_speed}")
            samples, sample_rate = engine.create(script, voice=cfg["voice"], speed=new_speed, lang=kokoro_lang)
            cfg["speed"] = new_speed
    if samples is None or len(samples) < int(sample_rate) * 0.5:
        raise RuntimeError("Kokoro produced no usable audio")
    # Sanity check: spoken duration must roughly match the text length
    # (~1.2-4.5 words per second). Catches silent / truncated renders.
    words = max(1, len(str(script).split()))
    seconds = len(samples) / float(sample_rate)
    wps = words / seconds
    if not (1.2 <= wps <= 4.5):
        raise RuntimeError(f"Kokoro audio duration implausible ({seconds:.1f}s for {words} words)")
    out_path = out_dir / f"voiceover_{lang}.wav"
    _write_wav(samples, sample_rate, out_path)
    _write_voice_meta(out_dir, "kokoro", cfg["voice"], cfg["speed"], cfg["model_file"])
    log(f"Kokoro voiceover saved: {out_path.name} ({seconds:.1f}s, {wps:.2f} words/s)")
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
