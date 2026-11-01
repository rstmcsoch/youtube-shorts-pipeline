#!/usr/bin/env python3
"""DYPOL reference-video analyser ("learn from reference videos", Phase 3 tool).

Extracts ABSTRACT edit-grammar measurements from a reference video. It never
stores or re-uses the reference's content; outputs are statistics, palettes and
low-res contact sheets for a human/vision model to label shot types.

Two sub-commands:

  measure   <video.mp4> --id ref_xxx --kind long|short --out DIR [--info info.json]
            -> DIR/<id>.measured.json  (cuts, shot stats, transitions, motion,
               colour palette, audio/music/loudness, chapters)
            -> DIR/<id>_sheets/*.jpg  (hook sheet, overview sheets, end sheet)

  finalize  --measured X.measured.json [--srt X.srt] [--labels labels.json] --out grammar.json
            -> edit-grammar JSON in the Kimi schema (reference_id, structure,
               pacing, transitions, typography, color, audio) + extra facets
               (shot_type_mix, hook, captions, motion, transcript).

labels.json (optional, from a human or vision-LLM pass over the contact sheets):
  {"overview_tile_labels": ["screen_recording", "talking_head", ...],  # one per overview tile, in order
   "hook_devices": [...], "typography": {...}, "captions": {...}, "notes": "...",
   "structure_roles": [{"role": "hook", "t": [0, 6]}, ...]}

Deps: scenedetect[opencv-headless], opencv, numpy, pillow, librosa, ffmpeg on PATH.
"""
import argparse, json, math, os, re, subprocess, sys
from pathlib import Path

import numpy as np

SHOT_TYPES = ["screen_recording", "code_editor", "browser_demo", "talking_head",
              "talking_head_over_screen", "diagram", "slide", "broll", "text_card",
              "chart", "paper_or_doc", "other"]


# ----------------------------------------------------------------- helpers
def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def ffprobe(path):
    r = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path])
    d = json.loads(r.stdout)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    num, den = v.get("avg_frame_rate", "30/1").split("/")
    fps = float(num) / float(den or 1)
    return {"duration": float(d["format"]["duration"]), "width": int(v["width"]),
            "height": int(v["height"]), "fps": round(fps, 3),
            "has_audio": any(s["codec_type"] == "audio" for s in d["streams"])}


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def r2(x, n=2):
    return None if x is None else round(float(x), n)


# ----------------------------------------------------------------- cuts
def detect_cuts(path, threshold=27.0, min_scene_len=12):
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
    video = open_video(path)
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    sm.auto_downscale = True
    sm.detect_scenes(video, show_progress=False)
    scenes = sm.get_scene_list()
    return [(s.seconds if hasattr(s, "seconds") else s.get_seconds(), e.seconds if hasattr(e, "seconds") else e.get_seconds()) for s, e in scenes]


def shot_stats(shots, duration):
    lens = np.array([e - s for s, e in shots]) if shots else np.array([duration])
    return {
        "shot_count": int(len(lens)),
        "median_shot_s": r2(np.median(lens)), "mean_shot_s": r2(lens.mean()),
        "p10": r2(pct(lens, 10)), "p25": r2(pct(lens, 25)), "p75": r2(pct(lens, 75)),
        "p90": r2(pct(lens, 90)), "max_shot_s": r2(lens.max()),
        "cuts_per_min": r2((len(lens) - 1) / (duration / 60.0)) if duration else None,
        "share_shots_under_1s": r2((lens < 1).mean(), 3),
        "share_shots_over_10s": r2((lens > 10).mean(), 3),
    }


def window_stats(shots, t0, t1):
    cuts = [s for s, _ in shots if t0 < s < t1]
    inside = [(max(s, t0), min(e, t1)) for s, e in shots if e > t0 and s < t1]
    lens = [e - s for s, e in inside]
    return {"t": [r2(t0), r2(t1)], "cuts": len(cuts),
            "shot_count": len(inside), "median_shot_s": r2(np.median(lens)) if lens else None,
            "cuts_per_min": r2(len(cuts) / ((t1 - t0) / 60.0)) if t1 > t0 else None}


# ----------------------------------------------------------------- frames
def grab_frames(path, times, width):
    """Grab frames at the given times (seconds) with OpenCV; returns list of BGR arrays."""
    import cv2
    cap = cv2.VideoCapture(path)
    out = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0, t) * 1000.0)
        ok, fr = cap.read()
        if not ok:
            out.append(None)
            continue
        h, w = fr.shape[:2]
        out.append(cv2.resize(fr, (width, int(h * width / w)), interpolation=cv2.INTER_AREA))
    cap.release()
    return out


def contact_sheet(frames, times, cols, path, label_prefix=""):
    from PIL import Image, ImageDraw
    frames = [(f, t) for f, t in zip(frames, times) if f is not None]
    if not frames:
        return None
    h, w = frames[0][0].shape[:2]
    rows = math.ceil(len(frames) / cols)
    sheet = Image.new("RGB", (cols * w, rows * h), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    for i, (f, t) in enumerate(frames):
        im = Image.fromarray(f[:, :, ::-1])
        x, y = (i % cols) * w, (i // cols) * h
        sheet.paste(im, (x, y))
        lab = f"{label_prefix}{i}  {int(t // 60)}:{t % 60:04.1f}"
        d.rectangle([x, y, x + 8 * len(lab) + 6, y + 16], fill=(0, 0, 0))
        d.text((x + 3, y + 2), lab, fill=(255, 255, 0))
        d.rectangle([x, y, x + w - 1, y + h - 1], outline=(60, 60, 60))
    sheet.save(path, quality=80)
    return str(path)


def build_sheets(path, meta, kind, outdir, rid):
    """Hook sheet (first 12 s @2fps), overview sheets (time-uniform), end sheet (last 15 s @1fps)."""
    outdir.mkdir(parents=True, exist_ok=True)
    dur = meta["duration"]
    portrait = meta["height"] > meta["width"]
    tw = 180 if portrait else 320
    cols_hook = 8 if portrait else 6
    sheets = {}
    ht = [i * 0.5 for i in range(int(min(12, dur) * 2))]
    sheets["hook"] = contact_sheet(grab_frames(path, ht, tw), ht, cols_hook, outdir / f"{rid}_hook.jpg", "h")
    # overview: long = every ~12 s (cap 100 tiles); short = every 1.5 s
    step = 1.5 if kind == "short" else max(10.0, dur / 100.0)
    ot = list(np.arange(step / 2, dur - 0.5, step))
    per = 24 if portrait else 20
    cols = 8 if portrait else 5
    sheets["overview"] = []
    sheets["overview_times"] = [r2(t) for t in ot]
    for k in range(0, len(ot), per):
        chunk = ot[k:k + per]
        p = contact_sheet(grab_frames(path, chunk, tw), chunk, cols, outdir / f"{rid}_overview_{k // per:02d}.jpg", f"o{k}+")
        sheets["overview"].append(p)
    et = [t for t in np.arange(max(0, dur - 15), dur - 0.3, 1.0)]
    sheets["end"] = contact_sheet(grab_frames(path, et, tw), et, cols_hook, outdir / f"{rid}_end.jpg", "e")
    return sheets


# ----------------------------------------------------------------- transitions + motion
def classify_transitions(path, shots, fps, max_cuts=400):
    """Per detected cut, look at frame-diff profile in +/-6 frames: one spike = hard cut,
    spread = dissolve/whip/wipe, dark trough = dip to black."""
    import cv2
    cap = cv2.VideoCapture(path)
    cuts = [s for s, _ in shots[1:]]
    if len(cuts) > max_cuts:
        idx = np.linspace(0, len(cuts) - 1, max_cuts).astype(int)
        cuts = [cuts[i] for i in idx]
    counts = {"hard_cut": 0, "dissolve_or_motion": 0, "dip": 0}
    for c in cuts:
        f0 = int(round(c * fps)) - 6
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f0))
        frames = []
        for _ in range(13):
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(cv2.resize(fr, (96, 54)), cv2.COLOR_BGR2GRAY).astype(np.float32))
        if len(frames) < 5:
            continue
        diffs = np.array([np.abs(frames[i + 1] - frames[i]).mean() for i in range(len(frames) - 1)])
        lum = np.array([f.mean() for f in frames])
        if lum.min() < 18 and lum.min() < 0.4 * max(lum[0], lum[-1]):
            counts["dip"] += 1
        elif diffs.max() > 0 and (diffs > 0.35 * diffs.max()).sum() <= 2:
            counts["hard_cut"] += 1
        else:
            counts["dissolve_or_motion"] += 1
    cap.release()
    tot = sum(counts.values()) or 1
    return {k: r2(v / tot, 3) for k, v in counts.items()} | {"n_cuts_sampled": tot}


def shot_motion(path, shots, max_shots=150):
    """Per shot: optical flow between two frames 0.5 s apart near the middle.
    Labels: static / zoom_in / zoom_out / pan / scroll / other_motion."""
    import cv2
    cap = cv2.VideoCapture(path)
    sel = shots if len(shots) <= max_shots else [shots[i] for i in np.linspace(0, len(shots) - 1, max_shots).astype(int)]
    labels = []
    for s, e in sel:
        if e - s < 0.7:
            labels.append("too_short")
            continue
        m = (s + e) / 2
        g = []
        for t in (m - 0.25, m + 0.25):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, fr = cap.read()
            if not ok:
                break
            g.append(cv2.cvtColor(cv2.resize(fr, (160, int(160 * fr.shape[0] / fr.shape[1]))), cv2.COLOR_BGR2GRAY))
        if len(g) < 2:
            labels.append("unknown")
            continue
        flow = cv2.calcOpticalFlowFarneback(g[0], g[1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.linalg.norm(flow, axis=2)
        if mag.mean() < 0.15:
            labels.append("static")
            continue
        h, w = g[0].shape
        yy, xx = np.mgrid[0:h, 0:w]
        rx, ry = xx - w / 2, yy - h / 2
        rn = np.sqrt(rx ** 2 + ry ** 2) + 1e-6
        radial = ((flow[..., 0] * rx + flow[..., 1] * ry) / rn).mean()
        mx, my = flow[..., 0].mean(), flow[..., 1].mean()
        coherent = np.hypot(mx, my) / (mag.mean() + 1e-6)
        if abs(radial) > 0.5 * mag.mean() and abs(radial) > 0.2:
            labels.append("zoom_in" if radial > 0 else "zoom_out")
        elif coherent > 0.6:
            labels.append("scroll" if abs(my) > 2 * abs(mx) else "pan")
        else:
            labels.append("other_motion")
    cap.release()
    valid = [l for l in labels if l not in ("too_short", "unknown")]
    mix = {k: r2(valid.count(k) / len(valid), 3) for k in sorted(set(valid))} if valid else {}
    return {"shots_sampled": len(sel), "motion_mix": mix,
            "share_shots_with_motion": r2(1 - mix.get("static", 0), 3) if valid else None}


# ----------------------------------------------------------------- colour
def palette(path, duration, k=6, n=40):
    import cv2
    times = list(np.linspace(duration * 0.03, duration * 0.97, n))
    frames = [f for f in grab_frames(path, times, 160) if f is not None]
    px = np.concatenate([f.reshape(-1, 3) for f in frames]).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, lab, cen = cv2.kmeans(px, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    share = np.bincount(lab.flatten(), minlength=k) / len(lab)
    order = np.argsort(-share)
    pal = [{"hex": "#%02X%02X%02X" % tuple(int(c) for c in cen[i][::-1]), "share": r2(share[i], 3)} for i in order]
    hsv = np.concatenate([cv2.cvtColor(f, cv2.COLOR_BGR2HSV).reshape(-1, 3) for f in frames]).astype(np.float32)
    gray = np.concatenate([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).flatten() for f in frames]).astype(np.float32)
    dark_share = float((gray < 60).mean())
    return {"dominant_palette": pal, "mean_luma": r2(gray.mean(), 1), "luma_std": r2(gray.std(), 1),
            "mean_saturation": r2(hsv[:, 1].mean() / 255, 3), "dark_pixel_share": r2(dark_share, 3),
            "theme": "dark" if gray.mean() < 90 else ("light" if gray.mean() > 160 else "mixed"),
            "contrast": "high" if gray.std() > 70 else ("medium" if gray.std() > 45 else "low")}


# ----------------------------------------------------------------- audio
def audio_features(path, duration, shots):
    out = {}
    r = run(["ffmpeg", "-nostats", "-i", path, "-vn", "-af", "ebur128=peak=true", "-f", "null", "-"])
    txt = r.stderr
    m = re.findall(r"I:\s+(-?[\d.]+) LUFS", txt)
    tp = re.findall(r"Peak:\s+(-?[\d.]+) dBFS", txt)
    lra = re.findall(r"LRA:\s+(-?[\d.]+) LU", txt)
    out["integrated_lufs"] = float(m[-1]) if m else None
    out["true_peak_dbfs"] = float(tp[-1]) if tp else None
    out["loudness_range_lu"] = float(lra[-1]) if lra else None
    try:
        import librosa
        wav = Path(path).with_suffix(".tmp.wav")
        run(["ffmpeg", "-y", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "22050", str(wav)])
        y, sr = librosa.load(str(wav), sr=22050, mono=True)
        wav.unlink(missing_ok=True)
        hop = 512
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
        beat_t = librosa.frames_to_time(beats, sr=sr, hop_length=hop)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop, units="time")
        # music-bed heuristic: harmonic energy in the narration's pauses + spectral flatness
        H, P = librosa.effects.hpss(y)
        h_rms = librosa.feature.rms(y=H, hop_length=hop)[0]
        floor = np.percentile(rms, 10)
        thr = np.percentile(rms, 30)
        pause = rms < thr
        bed_in_pauses = float(np.median(h_rms[pause])) if pause.any() else 0.0
        db = lambda v: 20 * np.log10(max(v, 1e-9))
        silence_floor_db = db(floor)
        music_bed_likely = silence_floor_db > -45  # a speech-only track drops to near silence in pauses
        # energy curve in 10 bins
        bins = np.array_split(rms, 10)
        curve = [r2(db(b.mean()), 1) for b in bins]
        # cuts on beat
        cut_t = np.array([s for s, _ in shots[1:]])
        on_beat = None
        if len(cut_t) and len(beat_t):
            d = np.min(np.abs(cut_t[:, None] - beat_t[None, :]), axis=1)
            on_beat = r2((d <= 0.12).mean(), 3)
        # strong transients near cuts ~ sfx hits
        strong = onsets[onset_env[librosa.time_to_frames(onsets, sr=sr, hop_length=hop)] > np.percentile(onset_env, 97)] if len(onsets) else np.array([])
        sfx_near_cuts = None
        if len(cut_t) and len(strong):
            d2 = np.min(np.abs(cut_t[:, None] - strong[None, :]), axis=1)
            sfx_near_cuts = r2((d2 <= 0.15).mean(), 3)
        out.update({
            "tempo_bpm_estimate": r2(float(np.atleast_1d(tempo)[0]), 1),
            "onsets_per_min": r2(len(onsets) / (duration / 60), 1),
            "strong_transients_per_min": r2(len(strong) / (duration / 60), 2),
            "share_cuts_with_transient_within_150ms": sfx_near_cuts,
            "cut_on_beat_ratio": on_beat,
            "pause_floor_db": r2(silence_floor_db, 1),
            "harmonic_rms_in_pauses_db": r2(db(bed_in_pauses), 1),
            "music_bed_likely": bool(music_bed_likely),
            "rms_energy_curve_db_10bins": curve,
            "first_5s_vs_avg_db": r2(db(rms[t < 5].mean()) - db(rms.mean()), 1),
        })
    except Exception as e:  # librosa optional
        out["librosa_error"] = str(e)[:200]
    return out


# ----------------------------------------------------------------- transcript
def parse_srt(path):
    raw = Path(path).read_text(encoding="utf-8-sig").splitlines()
    def sec(s):
        h, m, x = s.strip().split()[0].replace(",", ".").split(":")
        return int(h) * 3600 + int(m) * 60 + float(x)
    cues = []
    for i, row in enumerate(raw):
        if "-->" in row:
            a, b = row.split("-->")
            cues.append((sec(a), sec(b), []))
        elif row.strip() and cues:
            if row.strip().isdigit() and i + 1 < len(raw) and "-->" in raw[i + 1]:
                continue  # cue index
            cues[-1][2].append(" ".join(re.sub(r"<[^>]+>", "", row).split()))
    # de-duplicate rolling auto-captions: a line repeated as the first of a two-line cue,
    # or alone in a cue a few ms long, is a carry-over.
    clean, last = [], None
    for a, b, lines in cues:
        for i, t in enumerate(lines):
            carried = t == last and (b - a < 0.2 or (i == 0 and len(lines) > 1))
            if not carried:
                clean.append((a, b, t))
            last = t
    return clean


def transcript_stats(srt, duration):
    cues = parse_srt(srt)
    if not cues:
        return {}
    words = sum(len(t.split()) for _, _, t in cues)
    span = cues[-1][1] - cues[0][0]
    hook_text = " ".join(t for a, _, t in cues if a < 10)
    first15 = " ".join(t for a, _, t in cues if a < 15)
    sents = re.split(r"(?<=[.?!])\s+", " ".join(t for _, _, t in cues))
    q = sum(1 for s in sents if s.endswith("?"))
    you = len(re.findall(r"\byou(r)?\b", " ".join(t for _, _, t in cues), re.I))
    hw = hook_text.split()
    return {
        "words": words, "speech_span_s": r2(span, 1),
        "wpm": r2(words / (span / 60), 1) if span > 0 else None,
        "wpm_over_runtime": r2(words / (duration / 60), 1) if duration else None,
        "first_words_10s": " ".join(hw[:45]),
        "first_15s_text": first15[:400],
        "hook_word_count_10s": len(hw),
        "questions_per_min": r2(q / (duration / 60), 2),
        "you_your_per_min": r2(you / (duration / 60), 2),
        "first_spoken_s": r2(cues[0][0], 2),
        "last_lines": " ".join(t for a, _, t in cues if a > duration - 20)[:400],
    }


# ----------------------------------------------------------------- commands
def cmd_measure(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta = ffprobe(a.video)
    dur = meta["duration"]
    shots = detect_cuts(a.video, a.threshold, a.min_scene_len)
    res = {"reference_id": a.id, "kind": a.kind, "source_meta": meta,
           "detector": {"name": "ContentDetector", "threshold": a.threshold, "min_scene_len": a.min_scene_len},
           "cuts_s": [r2(s, 2) for s, _ in shots[1:]],
           "pacing": shot_stats(shots, dur),
           "windows": {
               "first_5s": window_stats(shots, 0, min(5, dur)),
               "first_10s": window_stats(shots, 0, min(10, dur)),
               "first_60s": window_stats(shots, 0, min(60, dur)),
               "middle": window_stats(shots, dur * 0.2, dur * 0.8),
               "last_30s": window_stats(shots, max(0, dur - 30), dur)},
           "per_decile_cuts_per_min": [window_stats(shots, dur * i / 10, dur * (i + 1) / 10)["cuts_per_min"] for i in range(10)],
           }
    res["transitions"] = classify_transitions(a.video, shots, meta["fps"])
    res["motion"] = shot_motion(a.video, shots)
    res["color"] = palette(a.video, dur)
    res["audio"] = audio_features(a.video, dur, shots) if meta["has_audio"] else {}
    res["sheets"] = build_sheets(a.video, meta, a.kind, out / f"{a.id}_sheets", a.id)
    if a.info and Path(a.info).exists():
        info = json.loads(Path(a.info).read_text())
        res["chapters"] = [{"t": r2(c.get("start_time")), "title_len_words": len((c.get("title") or "").split()),
                            "title": c.get("title")} for c in (info.get("chapters") or [])]
    Path(out / f"{a.id}.measured.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {out / (a.id + '.measured.json')}")


def cmd_finalize(a):
    m = json.loads(Path(a.measured).read_text())
    dur = m["source_meta"]["duration"]
    lab = json.loads(Path(a.labels).read_text()) if a.labels else {}
    ts = transcript_stats(a.srt, dur) if a.srt else {}
    tiles = lab.get("overview_tile_labels") or []
    mix = {k: r2(tiles.count(k) / len(tiles), 3) for k in sorted(set(tiles))} if tiles else {}
    p = m["pacing"]
    ch = m.get("chapters") or []
    structure = lab.get("structure_roles") or []
    for s in structure:  # attach measured pacing to each labelled role
        t0, t1 = s["t"]
        cuts = [c for c in m["cuts_s"] if t0 < c < t1]
        s.setdefault("shot_count", len(cuts) + 1)
        s.setdefault("cuts_per_min", r2(len(cuts) / max((t1 - t0) / 60, 1e-6)))
    g = {
        "reference_id": m["reference_id"], "kind": m["kind"], "duration_s": r2(dur, 1),
        "aspect": f'{m["source_meta"]["width"]}x{m["source_meta"]["height"]}',
        "structure": structure,
        "pacing": {"median_shot_s": p["median_shot_s"], "p10": p["p10"], "p90": p["p90"],
                   "mean_shot_s": p["mean_shot_s"], "cuts_per_min": p["cuts_per_min"],
                   "share_shots_under_1s": p["share_shots_under_1s"], "share_shots_over_10s": p["share_shots_over_10s"],
                   "cut_on_beat_ratio": m.get("audio", {}).get("cut_on_beat_ratio"),
                   "wpm": ts.get("wpm"), "hook_cuts_first_5s": m["windows"]["first_5s"]["cuts"],
                   "hook_cuts_first_10s": m["windows"]["first_10s"]["cuts"],
                   "cuts_per_min_by_decile": m["per_decile_cuts_per_min"]},
        "transitions": m["transitions"],
        "shot_type_mix": mix, "shot_type_mix_basis": f"{len(tiles)} time-uniform overview tiles (time-weighted)",
        "motion": m["motion"] | ({"observed": lab["motion_observed"]} if "motion_observed" in lab else {}),
        "typography": lab.get("typography", {}),
        "captions": lab.get("captions", {}),
        "hook": {"devices": lab.get("hook_devices", []), "pattern": lab.get("hook_pattern"),
                 "word_count_10s": ts.get("hook_word_count_10s"), "first_spoken_s": ts.get("first_spoken_s")},
        "chapters": {"count": len(ch), "mean_interval_s": r2(np.mean(np.diff([c["t"] for c in ch] + [dur])), 1) if ch else None,
                     "mean_title_words": r2(np.mean([c["title_len_words"] for c in ch]), 1) if ch else None},
        "outro_cta": lab.get("outro_cta"),
        "color": {"dominant_palette": [c["hex"] for c in m["color"]["dominant_palette"]],
                  "palette_shares": [c["share"] for c in m["color"]["dominant_palette"]],
                  "theme": m["color"]["theme"], "contrast": m["color"]["contrast"],
                  "mean_saturation": m["color"]["mean_saturation"], "grain": lab.get("grain", False)},
        "audio": {k: m.get("audio", {}).get(k) for k in ["integrated_lufs", "true_peak_dbfs", "loudness_range_lu", "music_bed_likely",
                  "pause_floor_db", "tempo_bpm_estimate", "strong_transients_per_min",
                  "share_cuts_with_transient_within_150ms", "rms_energy_curve_db_10bins"]}
                 | ({"observed": lab["audio_observed"]} if "audio_observed" in lab else {}),
        "transcript": {k: ts.get(k) for k in ["words", "wpm", "wpm_over_runtime", "questions_per_min", "you_your_per_min"]},
        "notes": lab.get("notes"),
        "provenance": {"tool": "references/tools/analyze_reference.py", "detector": m["detector"],
                       "shot_types_labelled_by": lab.get("labelled_by", "unlabelled")},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(g, indent=2))
    print(f"wrote {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("video"); m.add_argument("--id", required=True)
    m.add_argument("--kind", choices=["long", "short"], required=True)
    m.add_argument("--out", required=True); m.add_argument("--info")
    m.add_argument("--threshold", type=float, default=27.0)
    m.add_argument("--min-scene-len", type=int, default=12)
    f = sub.add_parser("finalize")
    f.add_argument("--measured", required=True); f.add_argument("--srt")
    f.add_argument("--labels"); f.add_argument("--out", required=True)
    a = ap.parse_args()
    {"measure": cmd_measure, "finalize": cmd_finalize}[a.cmd](a)


if __name__ == "__main__":
    main()
