"""Gemini Imagen b-roll generation + Ken Burns animation."""

import base64
import json
import os
from pathlib import Path

import requests
from PIL import Image

from .config import VIDEO_WIDTH, VIDEO_HEIGHT, get_gemini_key, get_gemini_image_model, get_pexels_key, run_cmd
from .log import log


def _generate_image_gemini(prompt: str, output_path: Path, api_key: str, model: str):
    """Generate an image using Gemini's native image Interactions API."""
    url = "https://generativelanguage.googleapis.com/v1beta/interactions"
    body = {
        "model": model,
        "input": [{"type": "text", "text": (
            f"Create a high-quality, visually clear vertical 9:16 b-roll image for this short video scene: "
            f"{prompt}. No text, no watermarks, cinematic but realistic."
        )}],
        "response_format": {
            "type": "image",
            "mime_type": "image/jpeg",
            "aspect_ratio": "9:16",
            "image_size": os.environ.get("GEMINI_IMAGE_SIZE", "1K"),
        },
    }
    timeout = float(os.environ.get("GEMINI_IMAGE_TIMEOUT", "90"))
    response = requests.post(
        url,
        json=body,
        timeout=timeout,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
    )
    if response.status_code != 200:
        try:
            detail = response.json().get("error", {}).get("message", response.text[:250])
        except Exception:
            detail = response.text[:250]
        raise RuntimeError(f"Gemini image Interactions API {response.status_code} for {model}: {detail}")

    data = response.json()
    # The API's interaction object exposes output_image; tolerate raw output blocks
    # too, because the REST response can include individual model-output steps.
    image_data = (data.get("output_image") or {}).get("data")
    if not image_data:
        for block in data.get("output", []) or []:
            if isinstance(block, dict) and block.get("type") in {"image", "image_output"}:
                image_data = block.get("data") or (block.get("image") or {}).get("data")
                if image_data:
                    break
    if not image_data:
        for step in data.get("steps", []) or []:
            for block in step.get("content", []) or []:
                if isinstance(block, dict) and block.get("type") in {"image", "image_output"}:
                    image_data = block.get("data") or (block.get("image") or {}).get("data")
                    if image_data:
                        break
            if image_data:
                break
    if not image_data:
        raise RuntimeError(f"Gemini Interactions API returned no image for model {model}")
    output_path.write_bytes(base64.b64decode(image_data))


def _save_portrait(path: Path) -> None:
    """Crop an image to the engine's 9:16 video dimensions."""
    with Image.open(path) as source:
        img = source.convert("RGB")
    target_w, target_h = VIDEO_WIDTH, VIDEO_HEIGHT
    orig_w, orig_h = img.size
    if orig_w < 1 or orig_h < 1:
        raise ValueError("Image has invalid dimensions")
    scale = max(target_w / orig_w, target_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = max(0, (new_w - target_w) // 2)
    top = max(0, (new_h - target_h) // 2)
    img = img.crop((left, top, left + target_w, top + target_h))
    img.save(path, format="PNG")


def _pexels_fallback(prompt: str, output_path: Path) -> bool:
    """Try a Pexels portrait stock image when Gemini generation is unavailable."""
    key = get_pexels_key()
    if not key:
        return False
    try:
        response = requests.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": key},
            params={"query": " ".join(str(prompt).split()[:8]), "orientation": "portrait", "per_page": 1},
            timeout=25,
        )
        response.raise_for_status()
        photos = response.json().get("photos", [])
        if not photos:
            return False
        src = photos[0].get("src", {})
        image_url = src.get("portrait") or src.get("large2x") or src.get("large")
        if not image_url:
            return False
        image_response = requests.get(image_url, timeout=30)
        image_response.raise_for_status()
        output_path.write_bytes(image_response.content)
        _save_portrait(output_path)
        return True
    except Exception as exc:
        log(f"Pexels fallback failed: {exc}")
        return False


def _fallback_frame(i: int, out_dir: Path) -> Path:
    """Deterministic gradient fallback; never pretend it was AI-generated."""
    palettes = [((13, 24, 52), (55, 24, 84)), ((10, 42, 48), (24, 30, 70)), ((46, 22, 30), (18, 28, 56))]
    top, bottom = palettes[i % len(palettes)]
    img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT))
    pixels = img.load()
    for y in range(VIDEO_HEIGHT):
        ratio = y / max(1, VIDEO_HEIGHT - 1)
        color = tuple(int(top[c] * (1 - ratio) + bottom[c] * ratio) for c in range(3))
        for x in range(VIDEO_WIDTH):
            pixels[x, y] = color
    path = out_dir / f"broll_{i}.png"
    img.save(path)
    return path


THIRD_PARTY_PROVIDERS = {"pixabay", "wikimedia_commons", "pixabay_video"}
MAX_AI_FRAMES = 10


def max_uses_for(provider: str) -> int:
    """How many shots one asset may fill: third-party once, AI/original twice."""
    return 1 if provider in THIRD_PARTY_PROVIDERS else 2


def _get_judge():
    try:
        from .visual_judge import get_judge
        return get_judge()
    except Exception:  # noqa: BLE001 - the judge must never block the pipeline
        return None


def _judge_block(judge):
    if judge is None:
        return {"judge_status": "disabled"}
    try:
        return judge.summary("short")
    except Exception as exc:  # noqa: BLE001
        return {"judge_status": "disabled", "error": str(exc)[:200]}


def generate_broll(prompts: list, out_dir: Path, visual_intents: list | None = None,
                   video_title: str = "") -> list[Path]:
    """Resolve real visuals through rights-gated providers, then gradient fallback.

    Per segment: an optional Pixabay video clip (motion tier, tried first in the
    shot plan) plus one still (Wikimedia → Pixabay still → FLUX → gradient).
    """
    from .visuals import select_visual, select_video_clip, flux_prompt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_list = list(prompts or [])[:8] or ["Cinematic technology background"]
    intents = list(visual_intents or [])
    while len(intents) < len(prompt_list):
        prompt = str(prompt_list[len(intents)])
        intents.append({
            "type": "concept", "segment": prompt, "entities": [],
            "keywords": prompt.split()[:12], "real_image_essential": False,
        })

    manifest = []
    frames = []
    judge = _get_judge()
    if judge is not None:
        judge.begin_video("short")
    judge_ctx = {"video_title": video_title or ""}
    shot_review_reasons = []
    used_hashes = set()
    used_clip_ids = set()
    clips_enabled = os.environ.get("ENABLE_PIXABAY_VIDEO", "1").strip() != "0"
    review_required = False
    gemini_disabled = False
    gemini_key = get_gemini_key()
    gemini_first_try = os.environ.get("ENABLE_GEMINI_IMAGE_FIRST_TRY", "0").strip() == "1"
    gemini_models = [m.strip() for m in os.environ.get("GEMINI_IMAGE_MODELS", get_gemini_image_model()).split(",") if m.strip()]
    for i, prompt in enumerate(prompt_list):
        out_path = out_dir / f"broll_{i}.png"
        intent = dict(intents[i])
        intent.setdefault("prompt", str(prompt))
        metadata = None
        needs_review = False
        # Optional single-model fast path. Disabled by default; a zero-quota 429
        # disables further Gemini attempts for this job immediately.
        if gemini_first_try and gemini_key and not gemini_disabled:
            model = gemini_models[0] if gemini_models else get_gemini_image_model()
            try:
                _generate_image_gemini(flux_prompt(intent), out_path, gemini_key, model)
                _save_portrait(out_path)
                digest = __import__("hashlib").sha256(out_path.read_bytes()).hexdigest()
                if digest not in used_hashes:
                    used_hashes.add(digest)
                    metadata = {
                        "provider": "gemini", "source_url": "", "asset_id": model,
                        "license": "AI-generated; review applicable model terms", "license_url": "",
                        "retrieved_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                        "sha256": digest,
                    }
                else:
                    out_path.unlink(missing_ok=True)
            except Exception as exc:
                message = str(exc).casefold()
                log(f"Optional Gemini image attempt skipped: {exc}")
                if "429" in message and ("limit: 0" in message or "quota" in message):
                    gemini_disabled = True
        if metadata is None:
            metadata, needs_review = select_visual(intent, out_path, used_hashes, ctx=judge_ctx)
            if (metadata.get("judge") or {}).get("needs_review"):
                shot_review_reasons.append(f"Segment {i}: visual judge: {metadata['judge'].get('review_reason')}")
        review_required = review_required or needs_review
        source = metadata.get("provider", "gradient")
        if source == "gradient":
            _fallback_frame(i, out_dir)
        else:
            try:
                _save_portrait(out_path)
                metadata["sha256"] = __import__("hashlib").sha256(out_path.read_bytes()).hexdigest()
            except Exception as exc:
                log(f"Selected visual could not be normalized; using gradient: {exc}")
                _fallback_frame(i, out_dir)
                metadata = {
                    "provider": "gradient", "source_url": "", "asset_id": "gradient-fallback",
                    "license": "Original generated gradient", "license_url": "",
                    "retrieved_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                    "sha256": "",
                }
                source = "gradient"
                if intent.get("type") in {"news_event", "product"} and intent.get("real_image_essential"):
                    review_required = True
        clip_meta = None
        if clips_enabled:
            clip_meta = select_video_clip(intent, out_dir / f"clip_{i}.mp4", used_clip_ids, ctx=judge_ctx)
            if clip_meta:
                log(f"Segment {i}: Pixabay clip {clip_meta['asset_id']} ({clip_meta['duration_seconds']}s)")
        frames.append(out_path)
        manifest.append({
            "frame": out_path.name,
            "segment": str(intent.get("segment") or prompt),
            "intent": {key: intent.get(key) for key in ("type", "entities", "keywords", "real_image_essential")},
            "provider": metadata.get("provider", source),
            "source_url": metadata.get("source_url", ""),
            "asset_id": metadata.get("asset_id", ""),
            "license": metadata.get("license", ""),
            "license_url": metadata.get("license_url", ""),
            "retrieved_at": metadata.get("retrieved_at", ""),
            "sha256": metadata.get("sha256", ""),
            "relevance_score": metadata.get("relevance_score"),
            "max_shot_seconds": 5 if metadata.get("provider") in {"wikimedia_commons", "pixabay"} else None,
            "max_uses": max_uses_for(metadata.get("provider", source)),
            "segment_index": i,
            "clip": clip_meta,
            "judge_status": judge.status if judge is not None else "disabled",
            **({"judge": metadata["judge"]} if metadata.get("judge") else {}),
        })

    judge_summary = _judge_block(judge)
    reasons = []
    if any(row.get("provider") == "gradient" and (row.get("intent") or {}).get("type") in {"news_event", "product"}
           and (row.get("intent") or {}).get("real_image_essential") for row in manifest) or \
            (review_required and not shot_review_reasons):
        reasons.append("A news/product segment requiring a real image used a fallback.")
    reasons += shot_review_reasons
    for reason in judge_summary.get("needs_review_reasons") or []:
        reasons.append(f"visual judge: {reason}")
    review_required = bool(review_required or reasons)
    manifest_path = out_dir / "broll_manifest.json"
    manifest_path.write_text(json.dumps({
        "assets": manifest,
        "review_required": review_required,
        "review_reasons": reasons,
        "credits": _generic_visual_credits(manifest),
        "judge_status": judge_summary.get("judge_status", "disabled"),
        "judge": judge_summary,
        # Providers that answered 429 / quota during this run (console quota panel).
        "provider_limits": _provider_limits(),
    }, indent=2), encoding="utf-8")
    (out_dir / "visual_review_required.json").write_text(
        json.dumps({"review_required": review_required}, indent=2), encoding="utf-8"
    )
    log(f"B-roll source manifest saved: {manifest_path}; review_required={review_required}")
    return frames


def _provider_limits() -> dict:
    try:
        from .visuals import provider_limits
        return provider_limits()
    except Exception:  # noqa: BLE001
        return {}


def _generic_visual_credits(manifest: list[dict]) -> str:
    """Credits are deliberately generic; never expose creator/photographer names."""
    providers = {item.get("provider") for item in manifest}
    providers |= {(item.get("clip") or {}).get("provider") for item in manifest}
    labels = []
    if "pixabay" in providers or "pixabay_video" in providers:
        labels.append("Pixabay")
    if "wikimedia_commons" in providers:
        labels.append("Wikimedia Commons (public domain)")
    if "cloudflare_workers_ai" in providers:
        labels.append("AI-generated (FLUX)")
    if "gemini" in providers:
        labels.append("AI-generated")
    return "Visuals: " + ", ".join(labels) if labels else ""


def top_up_ai_frames(out_dir: Path, shots_needed: int) -> int:
    """Add FLUX frames when the manifest cannot fill `shots_needed` shots under the
    reuse caps (third-party once, AI/original twice). Never exceeds MAX_AI_FRAMES
    FLUX frames in total. Returns the number of frames added."""
    from .visuals import _cloudflare_generate, flux_prompt

    out_dir = Path(out_dir)
    manifest_path = out_dir / "broll_manifest.json"
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    rows = data.get("assets", [])
    capacity = 0
    for row in rows:
        capacity += row.get("max_uses") or max_uses_for(row.get("provider", ""))
        if (row.get("clip") or {}).get("file") and (out_dir / row["clip"]["file"]).exists():
            capacity += 1
    ai_count = sum(1 for r in rows if r.get("provider") == "cloudflare_workers_ai")
    added = 0
    attempts = 0
    segments = [r for r in rows if r.get("intent")] or [{"intent": {"keywords": ["technology hardware"]}, "segment_index": 0}]
    while capacity < shots_needed and ai_count < MAX_AI_FRAMES and attempts < MAX_AI_FRAMES:
        attempts += 1
        # Spread extra frames over the segments that have the fewest visuals.
        per_seg = {}
        for r in rows:
            seg = r.get("segment_index", 0)
            per_seg[seg] = per_seg.get(seg, 0) + 1 + (1 if r.get("clip") else 0)
        base = min(segments, key=lambda r: (per_seg.get(r.get("segment_index", 0), 0), r.get("segment_index", 0)))
        intent = dict(base.get("intent") or {})
        variant = 1 + ai_count + added
        out_path = out_dir / f"broll_extra_{len(rows)}.png"
        meta = _cloudflare_generate(flux_prompt(intent, variant=variant), out_path)
        if not meta:
            break  # FLUX unavailable / rate limited: gradients fill the gap
        try:
            _save_portrait(out_path)
            meta["sha256"] = __import__("hashlib").sha256(out_path.read_bytes()).hexdigest()
        except Exception:
            out_path.unlink(missing_ok=True)
            continue
        rows.append({
            "frame": out_path.name, "segment": base.get("segment", ""), "intent": base.get("intent"),
            **{k: meta.get(k, "") for k in ("provider", "source_url", "asset_id", "license", "license_url",
                                           "retrieved_at", "sha256")},
            "relevance_score": None, "max_shot_seconds": None, "max_uses": 2,
            "segment_index": base.get("segment_index", 0), "clip": None, "top_up": True,
        })
        capacity += 2
        ai_count += 1
        added += 1
    if added:
        data["assets"] = rows
        data["credits"] = _generic_visual_credits(rows)
        data["provider_limits"] = _provider_limits()
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log(f"Added {added} FLUX frame(s) so no frame appears more than twice")
    return added


def judge_contact_sheet(out_dir: Path, video_title: str = "") -> dict | None:
    """Post-render check: one numbered 4x3 contact sheet of the stills (in segment order)
    sent to the visual judge. Flags review_required in the manifest on bad shots,
    circuit/caps; never raises and never blocks the pipeline."""
    judge = _get_judge()
    out_dir = Path(out_dir)
    manifest_path = out_dir / "broll_manifest.json"
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if judge is None:
        return None
    try:
        judge.begin_video("short")
        rows = sorted(data.get("assets", []), key=lambda r: (r.get("segment_index", 0), bool(r.get("top_up"))))
        tiles = [(out_dir / r["frame"], r.get("segment", ""), r) for r in rows
                 if r.get("frame") and r.get("provider") != "gradient" and (out_dir / r["frame"]).exists()]
        result = None
        if judge.active and tiles:
            result = judge.check_contact_sheet([(str(p), t) for p, t, _ in tiles], video_title, out_dir, "short")
            for bad in (result or {}).get("bad_shots", []):
                n = bad["n"] - 1
                if 0 <= n < len(tiles):
                    row = tiles[n][2]
                    row["contact_sheet_flag"] = bad.get("reason", "")
            for sheet in out_dir.glob("contact_sheet_short_*.jpg"):
                sheet.unlink(missing_ok=True)
        summary = judge.summary("short")
        data["judge"] = summary
        data["judge_status"] = summary.get("judge_status", "disabled")
        reasons = [r for r in data.get("review_reasons", []) if not str(r).startswith("visual judge: ")]
        reasons += [f"visual judge: {r}" for r in summary.get("needs_review_reasons") or []]
        seg_reasons = [r for r in data.get("review_reasons", []) if str(r).startswith("Segment ")]
        reasons = list(dict.fromkeys(seg_reasons + reasons))
        data["review_reasons"] = reasons
        data["review_required"] = bool(data.get("review_required") or reasons)
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        if summary.get("needs_review"):
            log(f"Visual judge needs review: {'; '.join(summary['needs_review_reasons'])}")
        return {"contact_sheet": result, "summary": summary}
    except Exception as exc:  # noqa: BLE001
        log(f"Visual judge contact sheet skipped: {exc}")
        return None


def animate_frame(img_path: Path, out_path: Path, duration: float, effect: str = "zoom_in", variant: int = 0):
    """Ken Burns animation on a single frame.

    `variant` > 0 is used when a frame appears a second time: the move is
    anchored on a different region (tighter, off-centre crop) so the repeat
    reads as a new shot rather than a copy.
    """
    fps = 30
    frames = max(1, int(round(duration * fps)))
    w, h = VIDEO_WIDTH, VIDEO_HEIGHT
    # Focus point as a fraction of the free (zoomed-out) area; centre by default.
    fx, fy = ((0.5, 0.5), (0.2, 0.3), (0.8, 0.7), (0.3, 0.75))[variant % 4]
    zmax = 1.12 if not variant else 1.28
    pz = 1.15 if not variant else 1.3

    if effect == "zoom_in":
        vf = (
            f"scale={int(w * zmax)}:{int(h * zmax)},"
            f"zoompan=z='{zmax}-{zmax - 1:.2f}*on/{frames}':x='(iw-iw/zoom)*{fx}':y='(ih-ih/zoom)*{fy}'"
            f":d={frames}:s={w}x{h}:fps={fps}"
        )
    elif effect == "pan_right":
        vf = (
            f"scale={int(w * pz)}:{int(h * pz)},"
            f"zoompan=z={pz}:x='(iw-iw/zoom)*on/{frames}':y='(ih-ih/zoom)*{fy}'"
            f":d={frames}:s={w}x{h}:fps={fps}"
        )
    elif effect == "pan_left":
        vf = (
            f"scale={int(w * pz)}:{int(h * pz)},"
            f"zoompan=z={pz}:x='(iw-iw/zoom)*(1-on/{frames})':y='(ih-ih/zoom)*{fy}'"
            f":d={frames}:s={w}x{h}:fps={fps}"
        )
    else:  # zoom_out
        vf = (
            f"scale={int(w * zmax)}:{int(h * zmax)},"
            f"zoompan=z='1.0+{zmax - 1:.2f}*on/{frames}':x='(iw-iw/zoom)*{fx}':y='(ih-ih/zoom)*{fy}'"
            f":d={frames}:s={w}x{h}:fps={fps}"
        )

    run_cmd([
        "ffmpeg", "-loop", "1", "-i", str(img_path),
        "-vf", vf, "-frames:v", str(frames), "-r", str(fps),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", str(out_path), "-y", "-loglevel", "quiet",
    ])


def render_clip_shot(clip_path: Path, out_path: Path, duration: float):
    """Fit a normalized clip into exactly `duration` seconds (clone last frame if short)."""
    fps = 30
    frames = max(1, int(round(duration * fps)))
    run_cmd([
        "ffmpeg", "-i", str(clip_path),
        "-vf", f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=increase,"
               f"crop={VIDEO_WIDTH}:{VIDEO_HEIGHT},fps={fps},setsar=1,"
               f"tpad=stop_mode=clone:stop_duration={duration:.2f},format=yuv420p",
        "-frames:v", str(frames), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        str(out_path), "-y", "-loglevel", "quiet",
    ])
