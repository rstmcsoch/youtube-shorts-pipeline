"""Post-render QC for Engine B: loudness, silences, black frames, freezes + a contact sheet.

One decode of the final video runs ebur128 + silencedetect (-35 dB, 0.6 s) on the
audio and blackdetect on the picture while a 4x4 contact sheet is tiled from 16
evenly spaced frames. blackdetect uses pix_th=0.03 so the DYPOL slate background
(#0F141C) is not mistaken for black. freezedetect runs on the pre-grade, pre-caption merge
(grain and captions would hide a frozen picture in the final). Results go to
qc.json next to the video. QC never raises and never blocks; hard failures (resolution, static > 3.5 s,
plus an off-topic script in __main__) mark the draft "needs fixes".
"""

import json
import re
import subprocess
from pathlib import Path

from .log import log

LUFS_TARGET = -14.0
LUFS_TOLERANCE = 1.0
TRUE_PEAK_MAX = -1.0
SILENCE = {"noise_db": -35, "min_s": 0.6}
FREEZE_MIN_S = 3.5
# Hard failures (owner rules): the draft is still saved, but it is marked "needs fixes".
HARD_STATIC_S = 3.5
SHORT_RES = ((720, 1280), (1080, 1920))   # 9:16, 720p-1080p, never upscaled


def _size(path: Path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                        "-of", "csv=p=0:s=x", str(path)], capture_output=True, text=True, timeout=60)
    try:
        w, h = (int(v) for v in r.stdout.strip().split("x")[:2])
        return w, h
    except ValueError:
        return None, None


def resolution_problem(width, height, bounds=SHORT_RES) -> str:
    if not width or not height:
        return ""
    (lo_w, lo_h), (hi_w, hi_h) = bounds
    if not (lo_w <= width <= hi_w and lo_h <= height <= hi_h):
        return f"resolution {width}x{height} is outside {lo_w}x{lo_h}-{hi_w}x{hi_h}"
    return ""


def hard_failures(report: dict | None, relevance: dict | None = None) -> list[str]:
    """Owner-rule failures that mark the draft "needs fixes": off-topic script, resolution outside
    720p-1080p (9:16), a static picture longer than 3.5 s."""
    out = []
    if isinstance(relevance, dict) and relevance.get("verdict") == "off_topic":
        out.append("off-topic script: " + "; ".join(relevance.get("reasons") or [])[:300])
    report = report or {}
    problem = resolution_problem(report.get("width"), report.get("height"))
    if problem:
        out.append(problem)
    for f in report.get("freezes") or []:
        if (f.get("duration") or 0) > HARD_STATIC_S:
            out.append(f"picture static for {f['duration']:.1f} s at {f['start']:.2f} s (limit {HARD_STATIC_S:g} s)")
    return out


def _duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, timeout=60)
    return float(r.stdout.strip())


def parse_ebur128(log_text: str) -> dict:
    """Integrated loudness, LRA and true peak from ebur128's summary block."""
    out = {}
    tail = log_text[log_text.rfind("Summary:"):] if "Summary:" in log_text else log_text
    for key, pat in (("integrated_lufs", r"I:\s+(-?[\d.]+|-inf)\s+LUFS"), ("lra_lu", r"LRA:\s+(-?[\d.]+)\s+LU"),
                     ("true_peak_dbfs", r"Peak:\s+(-?[\d.]+|-inf)\s+dBFS")):
        m = re.search(pat, tail)
        if m:
            out[key] = float(m.group(1)) if m.group(1) != "-inf" else float("-inf")
    return out


def parse_intervals(log_text: str, name: str) -> list[dict]:
    """silencedetect / blackdetect / freezedetect start-end pairs."""
    spans = []
    if name == "silence":
        starts = [float(x) for x in re.findall(r"silence_start:\s*(-?[\d.]+)", log_text)]
        ends = re.findall(r"silence_end:\s*(-?[\d.]+)\s*\|\s*silence_duration:\s*([\d.]+)", log_text)
        for i, s in enumerate(starts):
            if i < len(ends):
                spans.append({"start": round(s, 3), "end": round(float(ends[i][0]), 3),
                              "duration": round(float(ends[i][1]), 3)})
            else:
                spans.append({"start": round(s, 3), "end": None, "duration": None})
    elif name == "black":
        for s, e, d in re.findall(r"black_start:\s*([\d.]+)\s+black_end:\s*([\d.]+)\s+black_duration:\s*([\d.]+)",
                                  log_text):
            spans.append({"start": float(s), "end": float(e), "duration": float(d)})
    elif name == "freeze":
        starts = [float(x) for x in re.findall(r"freeze_start:\s*([\d.]+)", log_text)]
        durs = [float(x) for x in re.findall(r"freeze_duration:\s*([\d.]+)", log_text)]
        ends = [float(x) for x in re.findall(r"freeze_end:\s*([\d.]+)", log_text)]
        for i, s in enumerate(starts):
            spans.append({"start": s, "end": ends[i] if i < len(ends) else None,
                          "duration": durs[i] if i < len(durs) else None})
    return spans


def qc_warnings(report: dict) -> list[str]:
    warns = []
    loud = report.get("loudness") or {}
    i = loud.get("integrated_lufs")
    if i is None:
        warns.append("Loudness could not be measured")
    elif abs(i - LUFS_TARGET) > LUFS_TOLERANCE:
        warns.append(f"Integrated loudness {i:.1f} LUFS (target {LUFS_TARGET:g} ±{LUFS_TOLERANCE:g})")
    tp = loud.get("true_peak_dbfs")
    if tp is not None and tp > TRUE_PEAK_MAX:
        warns.append(f"True peak {tp:.1f} dBFS above {TRUE_PEAK_MAX:g}")
    for s in report.get("silences", []):
        warns.append(f"Silence {s.get('duration') or '?'} s at {s['start']:.2f} s")
    for b in report.get("black", []):
        warns.append(f"Black frames {b['duration']:.2f} s at {b['start']:.2f} s")
    for f in report.get("freezes", []):
        warns.append(f"Static picture {f.get('duration') or '?'} s at {f['start']:.2f} s (limit {FREEZE_MIN_S:g} s)")
    return warns


def run_qc(video: Path, out_json: Path, contact_png: Path, merged_video: Path | None = None) -> dict:
    """Measure the render; write qc.json and the contact sheet. Never raises."""
    video, out_json, contact_png = Path(video), Path(out_json), Path(contact_png)
    report = {"video": video.name, "status": "ok", "warnings": [], "errors": []}
    try:
        dur = _duration(video)
        report["duration_s"] = round(dur, 3)
        report["width"], report["height"] = _size(video)
        fc = (f"[0:a]ebur128=peak=true,silencedetect=noise={SILENCE['noise_db']}dB:d={SILENCE['min_s']}[ao];"
              f"[0:v]split=2[v1][v2];[v1]blackdetect=d=0.5:pix_th=0.03[vo];"
              f"[v2]fps=16/{max(dur, 1.0):.3f},scale=270:480,tile=4x4[sheet]")
        r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-y", "-i", str(video), "-filter_complex", fc,
                            "-map", "[ao]", "-map", "[vo]", "-f", "null", "-",
                            "-map", "[sheet]", "-frames:v", "1", "-update", "1", str(contact_png)],
                           capture_output=True, text=True, timeout=600)
        text = r.stderr or ""
        if r.returncode != 0:
            report["errors"].append(f"ffmpeg QC pass exited {r.returncode}: {text[-300:]}")
        report["loudness"] = parse_ebur128(text)
        report["silences"] = parse_intervals(text, "silence")
        report["black"] = parse_intervals(text, "black")
        report["contact_sheet"] = contact_png.name if contact_png.exists() else None
    except Exception as exc:
        report["errors"].append(f"QC pass failed: {exc}")
    try:
        src = Path(merged_video) if merged_video and Path(merged_video).exists() else video
        r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(src), "-an", "-vf",
                            f"freezedetect=n=-60dB:d={FREEZE_MIN_S}", "-f", "null", "-"],
                           capture_output=True, text=True, timeout=600)
        report["freezes"] = parse_intervals(r.stderr or "", "freeze")
        report["freeze_source"] = src.name
    except Exception as exc:
        report["errors"].append(f"freezedetect failed: {exc}")
    report["warnings"] = qc_warnings(report)
    if report["warnings"] or report["errors"]:
        report["status"] = "warn"
    report["hard_failures"] = hard_failures(report)
    report["verdict"] = "needs_fixes" if report["hard_failures"] else "ok"
    if report["hard_failures"]:
        report["status"] = "warn"
    try:
        out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except OSError as exc:
        log(f"QC: could not write {out_json}: {exc}")
    for w in report["warnings"]:
        log(f"QC warning: {w}")
    for w in report["hard_failures"]:
        log(f"QC NEEDS FIXES: {w}")
    return report


def hook_timing(words: list | None, limit_s: float = 0.8) -> dict:
    """When the first narrated word starts (Whisper timings) against the hook rule."""
    try:
        first = float((words or [])[0]["start"])
    except (IndexError, KeyError, TypeError, ValueError):
        return {"first_word_s": None, "ok": None, "limit_s": limit_s}
    return {"first_word_s": round(first, 3), "ok": first <= limit_s, "limit_s": limit_s}


def add_hook_timing(qc_json: Path, words: list | None, limit_s: float = 0.8) -> dict:
    """Write the hook timing into an existing qc.json (warning when the first word is late)."""
    timing = hook_timing(words, limit_s)
    try:
        report = json.loads(Path(qc_json).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return timing
    report["hook"] = timing
    if timing["ok"] is False:
        report.setdefault("warnings", []).append(
            f"hook: first word at {timing['first_word_s']:.2f} s (limit {limit_s:g} s)")
        report["status"] = "warn"
    Path(qc_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return timing
