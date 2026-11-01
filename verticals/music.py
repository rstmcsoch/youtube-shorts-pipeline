"""Background music — license-verified track selection + ducking.

Music is ON by default (DYPOL_MUSIC_ENABLED / MUSIC_ENABLED = 0 turns it off).
Source order:
  1. a track in music/ listed in music/tracks.json with a verified no-attribution
     license (e.g. Pixabay Content License / CC0), when any exists;
  2. otherwise the Open Lo-Fi pack (CC0 1.0, https://github.com/btahir/open-lofi):
     one sha256-pinned track pulled from its release zip at build time (see
     open_lofi.py). Any download problem -> the Short renders without music.
The track is chosen deterministically per job. The bed sits DYPOL_MUSIC_GAIN_DB
(default -26, clamp -28..-24) dB below the voice's integrated loudness, fades in /
out, and is gently side-chain ducked under speech. The record goes into the work
dir's broll_manifest.json ("audio.music").
"""

import json
import os
from pathlib import Path

from .log import log

# Music directory ships with the package
MUSIC_DIR = Path(__file__).resolve().parent.parent / "music"


ALLOWED_MUSIC_LICENSES = {"pixabay content license", "pixabay license", "cc0", "cc0-1.0", "public domain"}


def music_enabled() -> bool:
    from .open_lofi import music_enabled as _enabled
    return _enabled()


def verified_tracks(music_dir: Path | None = None) -> list[dict]:
    """Tracks listed in tracks.json with a verified, no-attribution license and a present file."""
    music_dir = Path(music_dir or MUSIC_DIR)
    try:
        listing = json.loads((music_dir / "tracks.json").read_text(encoding="utf-8")).get("tracks", [])
    except (OSError, ValueError, AttributeError):
        return []
    out = []
    for t in listing:
        if not isinstance(t, dict) or not t.get("file"):
            continue
        if t.get("verified") is not True or t.get("attribution_required") is not False:
            continue
        if str(t.get("license", "")).strip().casefold() not in ALLOWED_MUSIC_LICENSES:
            continue
        path = music_dir / Path(str(t["file"])).name
        if path.exists():
            out.append({**t, "path": str(path)})
    return out


def _find_tracks() -> list[Path]:
    """Eligible (license-verified) tracks only."""
    return [Path(t["path"]) for t in verified_tracks()]


def _get_speech_regions(audio_path: Path) -> list[tuple[float, float]]:
    """Extract speech regions from Whisper word timestamps (reuses captions data).

    Falls back to treating the entire audio as one speech region.
    """
    try:
        from .captions import _whisper_word_timestamps
        words = _whisper_word_timestamps(audio_path)
        if words:
            # Merge close words into speech regions (gap < 0.5s = same region)
            regions = []
            region_start = words[0]["start"]
            region_end = words[0]["end"]

            for w in words[1:]:
                if w["start"] - region_end < 0.5:
                    region_end = w["end"]
                else:
                    regions.append((region_start, region_end))
                    region_start = w["start"]
                    region_end = w["end"]
            regions.append((region_start, region_end))
            return regions
    except Exception:
        pass

    # Fallback: get total duration and treat as one speech region
    try:
        from .assemble import get_audio_duration
        dur = get_audio_duration(audio_path)
        return [(0.0, dur)]
    except Exception:
        return [(0.0, 60.0)]


def build_duck_filter(speech_regions: list[tuple[float, float]], buffer: float = 0.3, vol_speech: float = 0.12, vol_gap: float = 0.25) -> str:
    """Build ffmpeg volume filter expression for ducking during speech.

    During speech: volume = vol_speech (default 0.12)
    During gaps: volume = vol_gap (default 0.25)
    Transitions smoothed by ±buffer seconds.
    """
    if not speech_regions:
        return f"volume={vol_gap}"

    # Build between() conditions for speech regions
    conditions = []
    for start, end in speech_regions:
        s = max(0, start - buffer)
        e = end + buffer
        conditions.append(f"between(t,{s:.2f},{e:.2f})")

    condition_expr = "+".join(conditions)
    return f"volume='if({condition_expr}, {vol_speech}, {vol_gap})':eval=frame"


def _regions_from_words(words: list[dict]) -> list[tuple[float, float]]:
    if not words:
        return []
    regions = []
    start, end = words[0]["start"], words[0]["end"]
    for w in words[1:]:
        if w["start"] - end < 0.5:
            end = w["end"]
        else:
            regions.append((start, end))
            start, end = w["start"], w["end"]
    regions.append((start, end))
    return regions


def _measure_lufs(path) -> float | None:
    """Integrated loudness of an audio file (first 120 s), or None."""
    import subprocess
    try:
        from .assemble import _parse_loudnorm_json
        r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-t", "120", "-i", str(path),
                            "-af", "loudnorm=print_format=json", "-f", "null", "-"],
                           capture_output=True, text=True, timeout=120)
        data = _parse_loudnorm_json((r.stderr or "") + (r.stdout or ""))
        return float(data["input_i"]) if data else None
    except Exception:  # noqa: BLE001 - measuring is best effort
        return None


def _job_key(work_dir: Path) -> str:
    return os.environ.get("DYPOL_JOB_ID", "").strip() or Path(work_dir).name


def select_and_prepare_music(
    voiceover_path: Path,
    work_dir: Path,
    duck_speech: float = 0.12,
    duck_gap: float = 0.25,
    words: list[dict] | None = None,
) -> dict:
    """Pick this job's music bed (on by default; never fatal).

    Returns dict with track_path, duck_filter (legacy), gain_db (the volume applied
    to the track so the bed sits the target level below the voice) and the track's
    license record; {} = render without music. `words` (Whisper timestamps from
    captions) avoids running Whisper a second time.
    """
    from . import open_lofi
    if not music_enabled():
        log("Background music disabled (DYPOL_MUSIC_ENABLED=0)")
        return {}
    target_db = open_lofi.music_gain_db()
    tracks = verified_tracks()
    if tracks:
        chosen = tracks[int(open_lofi.hashlib.sha256(_job_key(work_dir).encode()).hexdigest()[:12], 16)
                        % len(tracks)]
        chosen = {**chosen, "provider": chosen.get("provider") or "local_verified"}
    else:
        chosen = open_lofi.fetch_track(_job_key(work_dir), log=log)
        if not chosen:
            return {}
    track = Path(chosen["path"])
    gain_db = open_lofi.relative_gain_db(_measure_lufs(voiceover_path), _measure_lufs(track), target_db)
    log(f"Selected music track: {track.name} ({chosen.get('license')}); bed {target_db:g} dB under the voice "
        f"(track volume {gain_db:g} dB)")
    _record_music(work_dir, chosen, gain_db, target_db)

    # Get speech regions for ducking
    speech_regions = _regions_from_words(words) if words else _get_speech_regions(voiceover_path)
    duck_filter = build_duck_filter(speech_regions, vol_speech=duck_speech, vol_gap=duck_gap)
    log(f"Built duck filter with {len(speech_regions)} speech regions")

    return {
        "track_path": str(track),
        "duck_filter": duck_filter,
        "gain_db": gain_db,
        "relative_db": target_db,
        "license": chosen.get("license", ""),
        "source_url": chosen.get("source_url", ""),
    }


def _record_music(work_dir: Path, track: dict, gain_db: float, relative_db: float | None = None) -> None:
    manifest = Path(work_dir) / "broll_manifest.json"
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    audio = data.get("audio") or {}
    audio["music"] = {
        "provider": track.get("provider", ""), "sha256": track.get("sha256", ""),
        "relative_db": relative_db,
        "file": Path(track["path"]).name, "license": track.get("license", ""),
        "license_url": track.get("license_url", ""), "source_url": track.get("source_url", ""),
        "attribution_required": False, "gain_db": gain_db,
        "ducking": "sidechaincompress ratio 2 under speech, 1.5 s fade in / out",
    }
    data["audio"] = audio
    try:
        manifest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass
