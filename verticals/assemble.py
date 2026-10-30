"""ffmpeg video assembly — shot plan (clips + stills) + voiceover + music + captions."""

import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .broll import animate_frame, render_clip_shot, max_uses_for, THIRD_PARTY_PROVIDERS
from .config import MEDIA_DIR, run_cmd
from .log import log

EFFECTS = ["zoom_in", "pan_right", "zoom_out", "pan_left"]
SHOT_TARGET_SECONDS = 4.0
LOUDNORM = {"I": -14.0, "TP": -1.5, "LRA": 11.0}


def assemble_workers() -> int:
    """Concurrent shot encodes: $DYPOL_ASSEMBLE_WORKERS, else up to 4 (one per runner core)."""
    try:
        value = int(os.environ.get("DYPOL_ASSEMBLE_WORKERS", "") or 0)
    except ValueError:
        value = 0
    if value <= 0:
        value = min(4, os.cpu_count() or 1)
    return max(1, value)


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


# ─────────────────────────────────────────────────────
# Shot plan
# ─────────────────────────────────────────────────────

def build_sources(frames: list[Path], out_dir: Path) -> list[dict]:
    """Visual sources from the b-roll manifest: per segment an optional clip, then the still."""
    out_dir = Path(out_dir)
    try:
        rows = json.loads((out_dir / "broll_manifest.json").read_text(encoding="utf-8")).get("assets", [])
    except (OSError, ValueError, AttributeError):
        rows = []
    sources = []
    by_name = {r.get("frame"): r for r in rows}
    seen = set()
    ordered = [(Path(f), by_name.get(Path(f).name, rows[i] if i < len(rows) else {})) for i, f in enumerate(frames)]
    # Top-up frames live only in the manifest.
    for r in rows:
        if r.get("frame") and r["frame"] not in {Path(f).name for f in frames}:
            ordered.append((out_dir / r["frame"], r))
    for i, (path, row) in enumerate(ordered):
        if str(path) in seen:
            continue
        seen.add(str(path))
        seg = row.get("segment_index", i)
        clip = row.get("clip") or {}
        if clip.get("file") and (out_dir / clip["file"]).exists():
            sources.append({"path": out_dir / clip["file"], "kind": "clip", "provider": clip.get("provider", "pixabay_video"),
                            "segment": seg, "max_uses": 1,
                            "max_len": float(clip.get("duration_seconds") or SHOT_TARGET_SECONDS)})
        provider = row.get("provider", "")
        sources.append({"path": path, "kind": "still", "provider": provider, "segment": seg,
                        "max_uses": int(row.get("max_uses") or max_uses_for(provider)),
                        "max_len": 5.0 if provider in THIRD_PARTY_PROVIDERS else None})
    return sources


def plan_shots(sources: list[dict], duration: float, target: float = SHOT_TARGET_SECONDS) -> list[dict]:
    """Assign ~target-second shots to sources in narration order.

    Rules: clips and third-party stills appear once; AI/original stills at most
    twice, and a repeat gets a different camera move + crop variant; never the
    same source twice in a row; gradients only when nothing else is left.
    Returns dicts {source (index or None), duration, effect, variant}.
    """
    n_shots = max(1, math.ceil(duration / target - 1e-6))
    shot_len = duration / n_shots
    uses = [0] * len(sources)
    first_effect = {}
    n_seg = max([s.get("segment", 0) for s in sources] + [0]) + 1
    shots, prev = [], None
    for k in range(n_shots):
        seg_target = min(n_seg - 1, int(k * n_seg / n_shots))
        best, best_key = None, None
        for idx, src in enumerate(sources):
            if uses[idx] >= src.get("max_uses", 1) or idx == prev:
                continue
            dist = abs(src.get("segment", 0) - seg_target)
            behind = src.get("segment", 0) < seg_target
            key = (dist > 1, uses[idx], dist, behind, 0 if src.get("kind") == "clip" else 1, idx)
            if best_key is None or key < best_key:
                best, best_key = idx, key
        if best is None:
            shots.append({"source": None, "duration": shot_len, "effect": EFFECTS[k % 4], "variant": 0})
            prev = None
            continue
        src = sources[best]
        if uses[best] == 0:
            effect, variant = EFFECTS[k % 4], 0
            first_effect[best] = effect
        else:
            first = first_effect.get(best, EFFECTS[0])
            effect = EFFECTS[(EFFECTS.index(first) + 2) % 4]  # opposite move
            variant = uses[best]
        uses[best] += 1
        prev = best
        shots.append({"source": best, "duration": shot_len, "effect": effect, "variant": variant})
    return shots


def shot_capacity(sources: list[dict]) -> int:
    return sum(int(s.get("max_uses", 1)) for s in sources)


# ─────────────────────────────────────────────────────
# Audio: optional music bed (sidechain ducked) + loudnorm -14 LUFS
# ─────────────────────────────────────────────────────

def music_filter_graph(duration: float, music_gain_db: float = -22.0) -> str:
    """Voice [1:a] + music [2:a]; music is ducked under the voice with sidechaincompress."""
    return (
        f"[2:a]atrim=0:{duration:.3f},asetpts=N/SR/TB,volume={music_gain_db}dB,"
        f"afade=t=out:st={max(0.0, duration - 1.5):.3f}:d=1.5[bed];"
        f"[1:a]asplit=2[voice][sc];"
        f"[bed][sc]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=350[ducked];"
        f"[voice][ducked]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[mix]"
    )


def loudnorm_filter(measured: dict | None = None) -> str:
    base = f"loudnorm=I={LOUDNORM['I']:g}:TP={LOUDNORM['TP']:g}:LRA={LOUDNORM['LRA']:g}"
    if not measured:
        return base
    return (base + f":measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
            f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
            f":offset={measured['target_offset']}:linear=true")


def _parse_loudnorm_json(stderr: str) -> dict | None:
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", stderr or "", re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        float(data["input_i"])  # "-inf" for silence → ValueError
        if not math.isfinite(float(data["input_i"])):
            return None
        return data
    except (ValueError, KeyError, TypeError):
        return None


def prepare_audio(voiceover: Path, out_dir: Path, duration: float,
                  music_path: str | None = None, music_gain_db: float = -22.0) -> Path:
    """Mix (voice + optional ducked music) and normalize to -14 LUFS, two-pass."""
    out_dir = Path(out_dir)
    mix = out_dir / "audio_mix.wav"
    if music_path and Path(music_path).exists():
        run_cmd(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-i", str(voiceover),
                 "-stream_loop", "-1", "-i", str(music_path), "-filter_complex", music_filter_graph(duration, music_gain_db),
                 "-map", "[mix]", "-t", f"{duration:.3f}", "-ar", "48000", "-ac", "2", str(mix)])
    else:
        run_cmd(["ffmpeg", "-y", "-loglevel", "error", "-i", str(voiceover), "-t", f"{duration:.3f}",
                 "-ar", "48000", "-ac", "2", str(mix)])
    final = out_dir / "audio_final.wav"
    measured = None
    try:
        r = run_cmd(["ffmpeg", "-hide_banner", "-nostats", "-i", str(mix),
                     "-af", loudnorm_filter() + ":print_format=json", "-f", "null", "-"], capture=True)
        measured = _parse_loudnorm_json((r.stderr or "") + (r.stdout or ""))
    except Exception as exc:
        log(f"loudnorm measure pass failed ({exc}); using single-pass")
    run_cmd(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mix), "-af", loudnorm_filter(measured),
             "-ar", "48000", "-ac", "2", str(final)])
    return final


# ─────────────────────────────────────────────────────
# Assembly
# ─────────────────────────────────────────────────────

def assemble_video(
    frames: list[Path],
    voiceover: Path,
    out_dir: Path,
    job_id: str,
    lang: str = "en",
    ass_path: str | None = None,
    music_path: str | None = None,
    duck_filter: str | None = None,  # legacy; music is now ducked with sidechaincompress
    music_gain_db: float = -22.0,
) -> Path:
    """Assemble final video from shots, voiceover, captions, and optional music."""
    log("Assembling video...")
    out_dir = Path(out_dir)
    duration = get_audio_duration(voiceover)

    sources = build_sources(frames, out_dir)
    plan = plan_shots(sources, duration)
    (out_dir / "shot_plan.json").write_text(json.dumps([
        {**shot, "file": str(sources[shot["source"]]["path"].name) if shot["source"] is not None else "gradient",
         "kind": sources[shot["source"]]["kind"] if shot["source"] is not None else "gradient"}
        for shot in plan], indent=2), encoding="utf-8")

    def _render_shot(i, shot):
        anim = out_dir / f"anim_{i}.mp4"
        if shot["source"] is None:
            from .broll import _fallback_frame
            animate_frame(_fallback_frame(200 + i, out_dir), anim, shot["duration"], shot["effect"])
        else:
            src = sources[shot["source"]]
            if src["kind"] == "clip":
                render_clip_shot(src["path"], anim, shot["duration"])
            else:
                animate_frame(src["path"], anim, shot["duration"], shot["effect"], variant=shot["variant"])
        return anim

    # Speed: every shot is an independent ffmpeg encode (zoompan is single-threaded),
    # so shots render concurrently; the audio mix/loudnorm passes run alongside them.
    # Each shot's file and encode settings are unchanged, so the output is identical.
    workers = assemble_workers()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="shot") as pool:
        audio_future = pool.submit(prepare_audio, voiceover, out_dir, duration, music_path, music_gain_db)
        animated = list(pool.map(lambda item: _render_shot(*item), enumerate(plan)))
        audio = audio_future.result()

    concat_file = out_dir / "concat.txt"

    def _esc(p):
        return str(p).replace("'", "'\\''")
    concat_file.write_text("\n".join(f"file '{_esc(p)}'" for p in animated))

    # Every shot is encoded identically (1080x1920, 30 fps, yuv420p H.264), so a
    # stream copy concat is lossless and fast; re-encode only if it fails.
    merged_video = out_dir / "merged_video.mp4"
    try:
        run_cmd(["ffmpeg", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy",
                 str(merged_video), "-y", "-loglevel", "quiet"])
    except Exception:
        run_cmd(["ffmpeg", "-f", "concat", "-safe", "0", "-i", str(concat_file),
                 "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-r", "30",
                 str(merged_video), "-y", "-loglevel", "quiet"])

    out_path = MEDIA_DIR / f"verticals_{job_id}_{lang}.mp4"
    vf_parts = ["tpad=stop_mode=clone:stop_duration=1"]  # never cut the last words
    if ass_path and Path(ass_path).exists():
        if _ffmpeg_has_libass():
            escaped_ass = str(ass_path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
            vf_parts.append(f"ass={escaped_ass}")
        else:
            log(
                "WARNING: this ffmpeg build has no libass — captions will NOT "
                "be burned in. The SRT is still uploaded to YouTube. Install "
                "an ffmpeg with libass (brew/apt builds include it) for "
                "burned-in captions."
            )
    run_cmd([
        "ffmpeg", "-i", str(merged_video), "-i", str(audio),
        "-vf", ",".join(vf_parts), "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-t", f"{duration:.3f}",
        "-movflags", "+faststart", str(out_path), "-y", "-loglevel", "quiet",
    ])
    log(f"Video assembled: {out_path}")
    return out_path
