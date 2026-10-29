"""ffmpeg video assembly — frames + voiceover + music + captions."""

from pathlib import Path

from .broll import animate_frame
from .config import MEDIA_DIR, run_cmd
from .log import log


def _ffmpeg_has_libass() -> bool:
    """Check whether this ffmpeg build ships the `ass` filter (libass).

    Some builds (e.g. minimal/static ones) omit libass; burning captions in
    would fail with `No such filter: 'ass'`, so we skip burn-in instead.
    """
    try:
        r = run_cmd(["ffmpeg", "-hide_banner", "-filters"], capture=True)
        return any(line.split()[1:2] == ["ass"] for line in r.stdout.splitlines())
    except Exception:
        return False


def get_audio_duration(path: Path) -> float:
    """Get duration of an audio file in seconds."""
    r = run_cmd(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture=True,
    )
    return float(r.stdout.strip())


def assemble_video(
    frames: list[Path],
    voiceover: Path,
    out_dir: Path,
    job_id: str,
    lang: str = "en",
    ass_path: str | None = None,
    music_path: str | None = None,
    duck_filter: str | None = None,
) -> Path:
    """Assemble final video from frames, voiceover, captions, and music."""
    log("Assembling video...")
    duration = get_audio_duration(voiceover)
    effects = ["zoom_in", "pan_right", "zoom_out", "pan_left"]

    # Shot plan: cut every ~4 s so the picture keeps moving with the narration.
    # Third-party stills (Pixabay / Wikimedia) are used once and never longer
    # than 5 s; AI-generated or original frames may return with a different
    # camera move. Gradients are only used when nothing else is available.
    manifest_path = Path(out_dir) / "broll_manifest.json"
    try:
        manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
        asset_rows = manifest.get("assets", [])
    except (OSError, ValueError, AttributeError):
        asset_rows = []
    providers = []
    for i, frame in enumerate(frames):
        row = asset_rows[i] if i < len(asset_rows) else {}
        providers.append(row.get("provider", ""))
    third_party = {"pixabay", "wikimedia_commons"}
    target = 4.0
    import math
    n_shots = max(1, math.ceil(duration / target))
    shot_len = duration / n_shots
    used_third = set()
    shots = []
    order = list(range(len(frames)))
    pos = 0
    for k in range(n_shots):
        pick = None
        for step in range(len(order)):
            idx = order[(pos + step) % len(order)] if order else None
            if idx is None:
                break
            if providers[idx] in third_party and idx in used_third:
                continue
            pick = idx
            pos = (pos + step + 1) % len(order)
            break
        if pick is None:
            from .broll import _fallback_frame
            shots.append((_fallback_frame(200 + k, Path(out_dir)), shot_len, k))
            continue
        if providers[pick] in third_party:
            used_third.add(pick)
        shots.append((frames[pick], min(shot_len, 5.0), k))

    # Animate each shot with Ken Burns effect
    animated = []
    for i, (frame, shot_duration, effect_index) in enumerate(shots):
        anim = out_dir / f"anim_{i}.mp4"
        animate_frame(frame, anim, shot_duration, effects[effect_index % len(effects)])
        animated.append(anim)

    # Concat animated segments (escape single quotes for ffmpeg concat demuxer)
    concat_file = out_dir / "concat.txt"
    def _esc(p):
        return str(p).replace("'", "'\\''" )
    concat_file.write_text("\n".join(f"file '{_esc(p)}'" for p in animated))

    merged_video = out_dir / "merged_video.mp4"
    run_cmd([
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
        str(merged_video), "-y", "-loglevel", "quiet",
    ])

    # Build the final ffmpeg command with optional captions + music
    out_path = MEDIA_DIR / f"verticals_{job_id}_{lang}.mp4"

    # Determine video filter (captions via ASS)
    vf_parts = []
    if ass_path and Path(ass_path).exists():
        if _ffmpeg_has_libass():
            # Escape special chars in path for ffmpeg filter
            escaped_ass = str(ass_path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
            vf_parts.append(f"ass={escaped_ass}")
        else:
            log(
                "WARNING: this ffmpeg build has no libass — captions will NOT "
                "be burned in. The SRT is still uploaded to YouTube. Install "
                "an ffmpeg with libass (brew/apt builds include it) for "
                "burned-in captions."
            )
    vf = ",".join(vf_parts) if vf_parts else None

    if music_path and Path(music_path).exists():
        # Three inputs: video, voiceover, music
        cmd = ["ffmpeg", "-i", str(merged_video), "-i", str(voiceover)]

        # Loop music to match video duration, apply ducking
        music_filter = f"[2:a]aloop=loop=-1:size=2e+09,atrim=0:{duration}"
        if duck_filter:
            music_filter += f",{duck_filter}"
        music_filter += "[music]"

        # Mix voiceover + ducked music
        audio_filter = f"{music_filter};[1:a][music]amix=inputs=2:duration=first:dropout_transition=2[aout]"

        cmd += [
            "-stream_loop", "-1", "-i", str(music_path),
            "-filter_complex", audio_filter,
        ]

        if vf:
            cmd += ["-vf", vf]

        cmd += [
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest",
            str(out_path), "-y", "-loglevel", "quiet",
        ]
    else:
        # Two inputs: video + voiceover (no music)
        cmd = ["ffmpeg", "-i", str(merged_video), "-i", str(voiceover)]

        if vf:
            cmd += ["-vf", vf]

        cmd += [
            "-c:v", "libx264" if vf else "copy",
            "-c:a", "aac", "-shortest",
            str(out_path), "-y", "-loglevel", "quiet",
        ]

    run_cmd(cmd)
    log(f"Video assembled: {out_path}")
    return out_path
