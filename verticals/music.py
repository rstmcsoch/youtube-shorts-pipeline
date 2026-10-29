"""Background music — license-verified track selection + ducking.

Music is OFF by default (MUSIC_ENABLED=1 turns it on). Pixabay's public API has
no music endpoint, so tracks are never fetched automatically: a track is only
eligible when it sits in music/ AND is listed in music/tracks.json with a
verified no-attribution license, e.g.

    {"tracks": [{"file": "calm-pulse.mp3", "license": "Pixabay Content License",
                 "license_url": "https://pixabay.com/service/license-summary/",
                 "source_url": "https://pixabay.com/music/...", "attribution_required": false,
                 "verified": true}]}

The chosen track is recorded in the work dir's broll_manifest.json ("audio").
"""

import json
import os
import random
from pathlib import Path

from .log import log

# Music directory ships with the package
MUSIC_DIR = Path(__file__).resolve().parent.parent / "music"


ALLOWED_MUSIC_LICENSES = {"pixabay content license", "pixabay license", "cc0", "public domain"}


def music_enabled() -> bool:
    return os.environ.get("MUSIC_ENABLED", "0").strip() == "1"


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


def select_and_prepare_music(
    voiceover_path: Path,
    work_dir: Path,
    duck_speech: float = 0.12,
    duck_gap: float = 0.25,
    words: list[dict] | None = None,
) -> dict:
    """Select a license-verified track (only when MUSIC_ENABLED=1).

    Returns dict with track_path, duck_filter (legacy), gain_db and the track's
    license record. `words` (Whisper timestamps from captions) avoids running
    Whisper a second time.
    """
    if not music_enabled():
        log("Background music disabled (MUSIC_ENABLED != 1)")
        return {}
    tracks = verified_tracks()
    if not tracks:
        log("No license-verified tracks in music/tracks.json — skipping background music")
        return {}

    chosen = random.choice(tracks)
    track = Path(chosen["path"])
    log(f"Selected music track: {track.name} ({chosen.get('license')})")
    try:
        gain_db = float(os.environ.get("MUSIC_GAIN_DB", "-22"))
    except ValueError:
        gain_db = -22.0
    gain_db = min(-20.0, max(-24.0, gain_db))
    _record_music(work_dir, chosen, gain_db)

    # Get speech regions for ducking
    speech_regions = _regions_from_words(words) if words else _get_speech_regions(voiceover_path)
    duck_filter = build_duck_filter(speech_regions, vol_speech=duck_speech, vol_gap=duck_gap)
    log(f"Built duck filter with {len(speech_regions)} speech regions")

    return {
        "track_path": str(track),
        "duck_filter": duck_filter,
        "gain_db": gain_db,
        "license": chosen.get("license", ""),
        "source_url": chosen.get("source_url", ""),
    }


def _record_music(work_dir: Path, track: dict, gain_db: float) -> None:
    manifest = Path(work_dir) / "broll_manifest.json"
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    audio = data.get("audio") or {}
    audio["music"] = {
        "file": Path(track["path"]).name, "license": track.get("license", ""),
        "license_url": track.get("license_url", ""), "source_url": track.get("source_url", ""),
        "attribution_required": False, "gain_db": gain_db, "ducking": "sidechaincompress",
    }
    data["audio"] = audio
    try:
        manifest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass
