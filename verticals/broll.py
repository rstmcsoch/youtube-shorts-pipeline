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


def generate_broll(prompts: list, out_dir: Path, visual_intents: list | None = None) -> list[Path]:
    """Resolve real visuals through rights-gated providers, then gradient fallback."""
    from .visuals import select_visual

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_list = list(prompts or [])[:3] or ["Cinematic technology background"]
    intents = list(visual_intents or [])
    while len(intents) < len(prompt_list):
        prompt = str(prompt_list[len(intents)])
        intents.append({
            "type": "concept", "segment": prompt, "entities": [],
            "keywords": prompt.split()[:12], "real_image_essential": False,
        })

    manifest = []
    frames = []
    used_hashes = set()
    review_required = False
    for i, prompt in enumerate(prompt_list):
        out_path = out_dir / f"broll_{i}.png"
        intent = dict(intents[i])
        intent.setdefault("prompt", str(prompt))
        metadata, needs_review = select_visual(intent, out_path, used_hashes)
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
        })

    manifest_path = out_dir / "broll_manifest.json"
    manifest_path.write_text(json.dumps({
        "assets": manifest,
        "review_required": review_required,
        "review_reasons": ["A news/product segment requiring a real image used a fallback."] if review_required else [],
        "credits": _generic_visual_credits(manifest),
    }, indent=2), encoding="utf-8")
    (out_dir / "visual_review_required.json").write_text(
        json.dumps({"review_required": review_required}, indent=2), encoding="utf-8"
    )
    log(f"B-roll source manifest saved: {manifest_path}; review_required={review_required}")
    return frames


def _generic_visual_credits(manifest: list[dict]) -> str:
    """Credits are deliberately generic; never expose creator/photographer names."""
    providers = {item.get("provider") for item in manifest}
    labels = []
    if "pixabay" in providers:
        labels.append("Pixabay")
    if "wikimedia_commons" in providers:
        labels.append("Wikimedia Commons (public domain)")
    if "cloudflare_workers_ai" in providers:
        labels.append("AI-generated (FLUX)")
    if "gemini" in providers:
        labels.append("AI-generated")
    return "Visuals: " + ", ".join(labels) if labels else ""


def animate_frame(img_path: Path, out_path: Path, duration: float, effect: str = "zoom_in"):
    """Ken Burns animation on a single frame."""
    fps = 30
    frames = int(duration * fps)
    w, h = VIDEO_WIDTH, VIDEO_HEIGHT

    if effect == "zoom_in":
        vf = (
            f"scale={int(w * 1.12)}:{int(h * 1.12)},"
            f"zoompan=z='1.12-0.12*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={frames}:s={w}x{h}:fps={fps}"
        )
    elif effect == "pan_right":
        vf = (
            f"scale={int(w * 1.15)}:{int(h * 1.15)},"
            f"zoompan=z=1.15:x='0.15*iw*on/{frames}':y='ih*0.075'"
            f":d={frames}:s={w}x{h}:fps={fps}"
        )
    else:  # zoom_out
        vf = (
            f"scale={int(w * 1.12)}:{int(h * 1.12)},"
            f"zoompan=z='1.0+0.12*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={frames}:s={w}x{h}:fps={fps}"
        )

    run_cmd([
        "ffmpeg", "-loop", "1", "-i", str(img_path),
        "-vf", vf, "-t", str(duration), "-r", str(fps),
        "-pix_fmt", "yuv420p", str(out_path), "-y", "-loglevel", "quiet",
    ])
