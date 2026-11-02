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
from .backdrop import Backdrop
from .motion import MOTION_KINDS, code_typing_window, motion_cards, render_block
from .style import STYLE_DIR

EFFECTS = ["zoom_in", "pan_right", "zoom_out", "pan_left"]
SHOT_TARGET_SECONDS = 4.0
LOUDNORM = {"I": -14.0, "TP": -1.5, "LRA": 11.0}
FPS = 30
ENCODE = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS)]
# Sub-shot moves: a punch-in on clips, a pull-back on motion blocks (keeps their text
# inside the safe band x 72-929, y <= 1080), cut on alternate sub-shots.
PUNCH_IN = "scale=1210:2150,crop=1080:1920"
PULL_BACK = "scale=994:1766,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=0x0F141C"
# With a living background a pull-back would frame the shot in a flat slate border, so
# motion sub-shots get a gentle 5% punch-in instead (text stays inside x >= 48).
MOTION_PUNCH = "scale=1134:2016,crop=1080:1920"
# Final grade: light contrast, soft vignette, subtle temporal grain. Applied before captions.
GRADE_FILTER = "eq=contrast=1.06:saturation=1.05:brightness=-0.008,vignette=angle=PI/5,noise=alls=4:allf=t"


def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off"}


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

def build_sources(frames: list[Path], out_dir: Path, beats: list | None = None) -> list[dict]:
    """Visual sources from the b-roll manifest: per segment an optional clip, then the still.

    Beats whose block is a motion kind (kinetic_text, count_up, code_panel, diagram)
    add a rendered motion source for that segment; the segment's still becomes a
    backup the planner uses only when nothing else is left nearby.
    """
    out_dir = Path(out_dir)
    try:
        rows = json.loads((out_dir / "broll_manifest.json").read_text(encoding="utf-8")).get("assets", [])
    except (OSError, ValueError, AttributeError):
        rows = []
    sources = []
    motion_segments = set()
    for seg, beat in enumerate(beats or []):
        if isinstance(beat, dict) and beat.get("block") in MOTION_KINDS and isinstance(beat.get("data"), dict):
            motion_segments.add(seg)
            sources.append({"path": out_dir / f"motion_{seg}_{beat['block']}", "kind": "motion",
                            "block": beat["block"], "data": beat["data"], "provider": "dypol_motion",
                            "segment": seg, "max_uses": 1, "max_len": None})
    # A stock beat whose only visual is a gradient fallback (no stock, no FLUX, no clip) becomes a
    # diagram/kinetic card from its keywords: a flat gradient shot reads as an empty screen (draft 4).
    seg_rows = {}
    for i, r in enumerate(rows):
        seg_rows.setdefault(r.get("segment_index", i), []).append(r)
    for seg, beat in enumerate(beats or []):
        if not isinstance(beat, dict) or seg in motion_segments or beat.get("block") in MOTION_KINDS:
            continue
        rs = seg_rows.get(seg, [])
        if rs and all(r.get("provider") == "gradient" and not (r.get("clip") or {}).get("file") for r in rs):
            kw = [str(k) for k in beat.get("keywords") or []]
            nodes = [k[:1].upper() + k[1:] for k in kw][:4]
            block, data = ("diagram", {"nodes": nodes, "title": ""}) if len(nodes) >= 2 else \
                ("kinetic_text", {"words": (kw or ["key", "point"])[:3], "accent": (kw or ["point"])[0]})
            motion_segments.add(seg)
            sources.append({"path": out_dir / f"motion_{seg}_{block}", "kind": "motion", "block": block,
                            "data": data, "provider": "dypol_motion", "segment": seg, "max_uses": 1, "max_len": None})
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
            clip_src = {"path": out_dir / clip["file"], "kind": "clip", "provider": clip.get("provider", "pixabay_video"),
                        "segment": seg, "max_uses": 1,
                        "max_len": float(clip.get("duration_seconds") or SHOT_TARGET_SECONDS)}
            if seg in motion_segments:  # a motion beat owns this segment; stock is only a backup
                clip_src["backup"] = True
            sources.append(clip_src)
        provider = row.get("provider", "")
        still = {"path": path, "kind": "still", "provider": provider, "segment": seg,
                 "max_uses": int(row.get("max_uses") or max_uses_for(provider)),
                 "max_len": 5.0 if provider in THIRD_PARTY_PROVIDERS else None}
        if seg in motion_segments:
            still.update(max_uses=1, backup=True)
        sources.append(still)
    return sources


def plan_shots(sources: list[dict], duration: float, target: float = SHOT_TARGET_SECONDS,
               durations: list[float] | None = None) -> list[dict]:
    """Assign ~target-second shots to sources in narration order.

    Rules: clips and third-party stills appear once; AI/original stills at most
    twice, and a repeat gets a different camera move + crop variant; never the
    same source twice in a row; gradients only when nothing else is left.
    `durations` (from pacing.shot_groups) sets each shot's length; otherwise shots
    are equal and ~target seconds.
    Returns dicts {source (index or None), duration, effect, variant}.
    """
    if durations:
        n_shots = len(durations)
    else:
        n_shots = max(1, math.ceil(duration / target - 1e-6))
    lengths = list(durations) if durations else [duration / n_shots] * n_shots
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
            key = (dist > 1, bool(src.get("backup")), uses[idx], dist, behind,
                   0 if src.get("kind") in ("motion", "clip") else 1, idx)
            if best_key is None or key < best_key:
                best, best_key = idx, key
        if best is None:
            shots.append({"source": None, "duration": lengths[k], "effect": EFFECTS[k % 4], "variant": 0})
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
        shots.append({"source": best, "duration": lengths[k], "effect": effect, "variant": variant})
    return shots


MAX_SHOT_SECONDS = 8.0


def final_motion_shot(plan: list[dict], sources: list[dict]) -> list[dict]:
    """The Short never ends on a still: if the planner put a still or gradient last, swap in
    the last segment's motion block (the takeaway card) when it is free, else the latest
    motion source not already shown right before it."""
    if not plan:
        return plan
    last = plan[-1]
    src = sources[last["source"]] if last["source"] is not None else {}
    if src.get("kind") == "motion":
        return plan
    used = {s["source"] for s in plan[:-1]}
    prev = plan[-2]["source"] if len(plan) > 1 else None
    motion = [i for i, s in enumerate(sources) if s.get("kind") == "motion" and i != prev]
    pick = next((i for i in sorted(motion, key=lambda i: -sources[i].get("segment", 0)) if i not in used), None)
    if pick is None and motion:
        pick = max(motion, key=lambda i: sources[i].get("segment", 0))
    if pick is not None:
        last.update(source=pick, effect=EFFECTS[0], variant=0)
        log(f"Final shot was a {src.get('kind', 'gradient')}; using motion block {sources[pick]['block']} instead")
    return plan


def backdrop_sources(plan: list[dict], sources: list[dict]) -> dict:
    """Shot index -> footage path to run (blurred, darkened, Ken Burns) behind a motion block.

    Uses the beat's own segment clip/still first (clips before stills), never a gradient
    and never an asset that is already a shot of its own. The hook and middle beats fall
    back to the nearest segment's footage; the rest get the procedural living background.
    """
    used = {s["source"] for s in plan if s["source"] is not None}
    taken, out = set(), {}
    motion_shots = [(k, sources[shot["source"]]) for k, shot in enumerate(plan)
                    if shot["source"] is not None and sources[shot["source"]].get("kind") == "motion"]

    def free():
        return [j for j, s in enumerate(sources)
                if s.get("kind") in ("clip", "still") and j not in used and j not in taken
                and s.get("provider") != "gradient" and Path(s["path"]).exists()]

    for k, src in motion_shots:  # pass 1: each beat's own segment
        seg = src.get("segment", 0)
        same = sorted((j for j in free() if sources[j].get("segment") == seg), key=lambda j: sources[j]["kind"] != "clip")
        if same:
            out[k] = sources[same[0]]["path"]
            taken.add(same[0])
    for k, src in motion_shots:  # pass 2: hook and middle borrow the nearest leftover footage
        if k in out or k == len(plan) - 1:
            continue
        seg = src.get("segment", 0)
        near = sorted(free(), key=lambda j: (abs(sources[j].get("segment", 0) - seg), sources[j]["kind"] != "clip"))
        if near:
            out[k] = sources[near[0]]["path"]
            taken.add(near[0])
    return out


def kinetic_runtime_share(plan: list[dict], sources: list[dict]) -> float:
    total = sum(s["duration"] for s in plan) or 1.0
    kin = sum(s["duration"] for s in plan if s["source"] is not None
              and sources[s["source"]].get("block") == "kinetic_text")
    return round(kin / total, 3)


def shot_count(sources: list[dict], duration: float, target: float = SHOT_TARGET_SECONDS,
               max_len: float = MAX_SHOT_SECONDS) -> int:
    """How many source shots a beat-driven video gets.

    One per primary (non-backup) source, so every beat shows once and the planner
    never runs out and fills the tail with backup stills (the first real draft ended
    on 4 off-topic stills). Never more than duration / target, never fewer than
    duration / max_len: long shots are split into 2-4 s sub-shots by pacing anyway.
    """
    primaries = sum(1 for s in sources if not s.get("backup"))
    default = max(1, math.ceil(duration / target - 1e-6))
    floor = max(1, math.ceil(duration / max_len - 1e-6))
    if primaries and all(s.get("kind") == "motion" for s in sources if not s.get("backup")):
        # every beat is a motion block that changes card per sub-shot, so a long beat is fine;
        # an extra shot would replay an earlier beat after the takeaway (draft 3 ended on a repeat)
        return min(default, primaries)
    return max(floor, min(default, primaries or default))


def shot_capacity(sources: list[dict]) -> int:
    return sum(int(s.get("max_uses", 1)) for s in sources)


# ─────────────────────────────────────────────────────
# Audio: optional music bed (sidechain ducked) + loudnorm -14 LUFS
# ─────────────────────────────────────────────────────

def music_filter_graph(duration: float, music_gain_db: float = -22.0, sfx: bool = False) -> str:
    """Voice [1:a] + music [2:a] (+ SFX bed [3:a]); music is ducked under the voice with sidechaincompress.

    The SFX bed is already set ~12 dB under the voice peak (verticals/sfx.py) and is
    mixed beside the ducked music, so effects never pump the music.
    """
    if sfx:
        return (music_filter_graph(duration, music_gain_db).replace("[voice][ducked]amix=inputs=2", "[voice][ducked][3:a]amix=inputs=3"))
    return (
        f"[2:a]atrim=0:{duration:.3f},asetpts=N/SR/TB,volume={music_gain_db}dB,"
        f"afade=t=in:d=1.5,afade=t=out:st={max(0.0, duration - 1.5):.3f}:d=1.5[bed];"
        f"[1:a]asplit=2[voice][sc];"
        # Gentle duck: the bed is already ~26 dB under the voice, so it stays audible.
        f"[bed][sc]sidechaincompress=threshold=0.05:ratio=2:attack=20:release=350[ducked];"
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
                  music_path: str | None = None, music_gain_db: float = -22.0,
                  sfx_path: str | None = None) -> Path:
    """Mix (voice + optional ducked music + optional SFX bed) and normalize to -14 LUFS, two-pass."""
    out_dir = Path(out_dir)
    mix = out_dir / "audio_mix.wav"
    has_sfx = bool(sfx_path) and Path(sfx_path).exists()
    if music_path and Path(music_path).exists():
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-i", str(voiceover),
               "-stream_loop", "-1", "-i", str(music_path)]
        if has_sfx:
            cmd += ["-i", str(sfx_path)]
        run_cmd(cmd + ["-filter_complex", music_filter_graph(duration, music_gain_db, sfx=has_sfx),
                       "-map", "[mix]", "-t", f"{duration:.3f}", "-ar", "48000", "-ac", "2", str(mix)])
    elif has_sfx:
        run_cmd(["ffmpeg", "-y", "-loglevel", "error", "-i", str(voiceover), "-i", str(sfx_path), "-filter_complex",
                 "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[mix]",
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

def _concat_copy(parts: list[Path], out: Path) -> None:
    lst = Path(str(out) + ".txt")
    lst.write_text("\n".join("file '" + str(p).replace("'", "'\\''") + "'" for p in parts))
    run_cmd(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)])


def recut_filter(cuts: list[float], move: str, fps: int = FPS) -> tuple[str, int]:
    """filter_complex that cuts one rendered shot into sub-shots; odd sub-shots get `move`."""
    n = len(cuts)
    bounds, acc = [0], 0.0
    for c in cuts:
        acc += c
        bounds.append(int(round(acc * fps)))
    graph = f"[0:v]split={n}" + "".join(f"[s{k}]" for k in range(n)) + ";"
    for k in range(n):
        xf = f",{move}" if k % 2 == 1 else ""
        graph += f"[s{k}]trim=start_frame={bounds[k]}:end_frame={bounds[k + 1]},setpts=PTS-STARTPTS{xf},setsar=1[p{k}];"
    graph += "".join(f"[p{k}]" for k in range(n)) + f"concat=n={n}:v=1:a=0,format=yuv420p[v]"
    return graph, bounds[-1]


def _recut(whole: Path, out: Path, cuts: list[float], move: str) -> None:
    graph, frames = recut_filter(cuts, move)
    run_cmd(["ffmpeg", "-y", "-loglevel", "error", "-i", str(whole), "-filter_complex", graph, "-map", "[v]",
             "-frames:v", str(frames), *ENCODE, str(out)])


def shot_timeline(plan: list[dict]) -> list[float]:
    """Start time of every shot."""
    starts, t = [], 0.0
    for shot in plan:
        starts.append(round(t, 4))
        t += shot["duration"]
    return starts


def sfx_inputs(plan: list[dict], sources: list[dict], words: list | None) -> dict:
    """Timeline facts the SFX planner needs: beat changes, sub-cuts, hook, typing, riser."""
    starts = shot_timeline(plan)
    beat_changes = starts[1:]
    sub_cuts, typing, riser_at = [], [], None
    for k, shot in enumerate(plan):
        acc = starts[k]
        for c in (shot.get("cuts") or [shot["duration"]])[:-1]:
            acc += c
            sub_cuts.append(round(acc, 4))
        src = sources[shot["source"]] if shot["source"] is not None else {}
        if src.get("kind") == "motion" and src.get("block") == "code_panel":
            t0, t1 = code_typing_window(src.get("data") or {}, shot["duration"])
            typing.append((starts[k] + t0, starts[k] + t1))
        if riser_at is None and k > 0 and src.get("kind") == "motion" and src.get("block") == "count_up":
            riser_at = starts[k]
    hook_t = 0.0
    if words:
        try:
            hook_t = max(0.0, min(0.5, float(words[0]["start"])))
        except (KeyError, TypeError, ValueError, IndexError):
            hook_t = 0.0
    return {"beat_changes": beat_changes, "sub_cuts": sub_cuts, "hook_t": hook_t, "typing": typing,
            "riser_at": riser_at}


def qc_paths(job_id: str, lang: str) -> tuple[Path, Path]:
    """qc.json + contact sheet, beside the video in MEDIA_DIR (uploaded with the draft artifact)."""
    return (MEDIA_DIR / f"verticals_{job_id}_{lang}_qc.json", MEDIA_DIR / f"verticals_{job_id}_{lang}_contact.png")


def _ass_filter(ass_path: str) -> str:
    def esc(p):
        return str(p).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    return f"ass={esc(ass_path)}:fontsdir={esc(STYLE_DIR / 'fonts')}"


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
    beats: list | None = None,
    words: list | None = None,
) -> Path:
    """Assemble final video from shots, voiceover, captions, SFX and optional music.

    Pacing: cuts land every 2-4 s on clause boundaries (verticals/pacing.py); a
    visual that holds longer is split into sub-shots with a different move. The
    final pass grades the picture, burns the DYPOL captions, and QC writes qc.json
    and a contact sheet next to the video (warnings only).
    """
    from .pacing import shot_groups
    log("Assembling video...")
    out_dir = Path(out_dir)
    duration = get_audio_duration(voiceover)

    if beats:
        from .beats import normalize_beats
        beats = normalize_beats(beats, n_segments=len(beats))[0]
    sources = build_sources(frames, out_dir, beats)
    groups = shot_groups(duration, words, SHOT_TARGET_SECONDS, fps=FPS,
                         n_shots=shot_count(sources, duration) if beats else None)
    plan = plan_shots(sources, duration, durations=[sum(g) for g in groups])
    for shot, cuts in zip(plan, groups):
        shot["cuts"] = cuts
    plan = final_motion_shot(plan, sources)
    starts = shot_timeline(plan)
    backdrops = backdrop_sources(plan, sources) if _flag("DYPOL_LIVING_BG") else {}
    for k, path in backdrops.items():
        plan[k]["backdrop"] = Path(path).name
    log(f"Kinetic-text share of runtime: {kinetic_runtime_share(plan, sources):.0%}")
    (out_dir / "shot_plan.json").write_text(json.dumps([
        {**shot, "file": str(sources[shot["source"]]["path"].name) if shot["source"] is not None else "gradient",
         "kind": sources[shot["source"]]["kind"] if shot["source"] is not None else "gradient"}
        for shot in plan], indent=2), encoding="utf-8")

    def _still_shot(i, img_for, shot, anim):
        cuts = shot.get("cuts") or [shot["duration"]]
        if len(cuts) == 1:
            animate_frame(img_for(0), anim, shot["duration"], shot["effect"], variant=shot["variant"])
            return
        parts = []
        e0 = EFFECTS.index(shot["effect"]) if shot["effect"] in EFFECTS else 0
        for j, d in enumerate(cuts):  # each sub-shot: a new move and a tighter crop
            part = out_dir / f"anim_{i}_{j}.mp4"
            animate_frame(img_for(j), part, d, EFFECTS[(e0 + j) % 4], variant=shot["variant"] + j)
            parts.append(part)
        _concat_copy(parts, anim)

    def _render_shot(i, shot):
        anim = out_dir / f"anim_{i}.mp4"
        cuts = shot.get("cuts") or [shot["duration"]]
        if shot["source"] is None:
            from .broll import _fallback_frame
            fb = _fallback_frame(200 + i, out_dir)
            _still_shot(i, lambda j: fb, shot, anim)
            return anim
        src = sources[shot["source"]]
        whole = out_dir / f"anim_{i}_whole.mp4" if len(cuts) > 1 else anim
        if src["kind"] == "motion":
            try:
                bd = None
                if _flag("DYPOL_LIVING_BG"):  # never a flat, static background behind a motion block
                    bd = Backdrop(backdrops.get(i), seed=i, duration=shot["duration"])
                if len(cuts) > 1 and src["block"] in ("kinetic_text", "count_up"):
                    # a new card per sub-shot, following the narration (never one card for 8 s)
                    parts = []
                    t_bg = 0.0
                    for j, spec in enumerate(motion_cards(src["block"], src["data"], cuts, starts[i], words)):
                        part = out_dir / f"anim_{i}_{j}.mp4"
                        render_block(spec["kind"], spec["data"], part, spec["duration"], backdrop=bd, bg_t0=t_bg)
                        parts.append(part)
                        t_bg += spec["duration"]
                    _concat_copy(parts, anim)
                    return anim
                render_block(src["block"], src["data"], whole, shot["duration"], backdrop=bd)
                if len(cuts) > 1:
                    _recut(whole, anim, cuts, MOTION_PUNCH if bd is not None else PULL_BACK)
            except Exception as exc:  # never lose the video over one motion block
                log(f"Motion block {src['block']} failed ({exc}); using a fallback frame")
                from .broll import _fallback_frame
                fb = _fallback_frame(300 + i, out_dir)
                _still_shot(i, lambda j: fb, shot, anim)
        elif src["kind"] == "clip":
            render_clip_shot(src["path"], whole, shot["duration"])
            if len(cuts) > 1:
                _recut(whole, anim, cuts, PUNCH_IN)
        else:
            _still_shot(i, lambda j: src["path"], shot, anim)
        return anim

    # SFX bed (whoosh/pop on beat changes, impact on the hook, clicks on typing).
    sfx_path = None
    if _flag("DYPOL_SFX_ENABLED"):
        try:
            from .sfx import plan_sfx, render_sfx_track
            events = plan_sfx(duration, **sfx_inputs(plan, sources, words))
            (out_dir / "sfx_plan.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
            sfx_path = render_sfx_track(events, duration, voiceover, out_dir / "sfx_track.wav")
        except Exception as exc:
            log(f"SFX skipped ({exc})")
            sfx_path = None

    # Speed: every shot is an independent ffmpeg encode (zoompan is single-threaded),
    # so shots render concurrently; the audio mix/loudnorm passes run alongside them.
    workers = assemble_workers()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="shot") as pool:
        audio_future = pool.submit(prepare_audio, voiceover, out_dir, duration, music_path, music_gain_db,
                                   sfx_path=str(sfx_path) if sfx_path else None)
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
    if _flag("DYPOL_GRADE"):
        vf_parts.append(GRADE_FILTER)
    if ass_path and Path(ass_path).exists():
        if _ffmpeg_has_libass():
            vf_parts.append(_ass_filter(ass_path))
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
    if _flag("DYPOL_QC_ENABLED") and out_path.exists():
        try:
            from .qc import run_qc
            qc_json, contact = qc_paths(job_id, lang)
            run_qc(out_path, qc_json, contact, merged_video=merged_video)
        except Exception as exc:  # QC only warns
            log(f"QC skipped ({exc})")
    return out_path
