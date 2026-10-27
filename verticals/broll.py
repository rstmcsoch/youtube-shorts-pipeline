"""Gemini Imagen b-roll generation + Ken Burns animation."""

import base64
import json
import os
from pathlib import Path

import requests
from PIL import Image

from .config import VIDEO_WIDTH, VIDEO_HEIGHT, get_gemini_key, get_gemini_image_model, get_pexels_key, run_cmd
from .log import log
from .retry import with_retry


@with_retry(max_retries=3, base_delay=2.0)
def _generate_image_gemini(prompt: str, output_path: Path, api_key: str, model: str):
    """Generate a portrait image through the current Gemini generateContent API."""
    url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"
    body = {
        "contents": [{"parts": [{"text": f"Create a high-quality, visually clear vertical 9:16 b-roll image for this short video scene: {prompt}. No text, no watermarks, cinematic but realistic."}]}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {
                "aspectRatio": "9:16",
                "imageSize": os.environ.get("GEMINI_IMAGE_SIZE", "1K"),
            },
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
        raise RuntimeError(f"Gemini image API {response.status_code} for {model}: {detail}")
    data = response.json()
    for part in data.get("candidates", [{}])[0].get("content", {}).get("parts", []):
        inline = part.get("inlineData") or part.get("inline_data")
        if inline and inline.get("data"):
            output_path.write_bytes(base64.b64decode(inline["data"]))
            return
    raise RuntimeError(f"Gemini image API returned no image for model {model}")


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


def generate_broll(prompts: list, out_dir: Path) -> list[Path]:
    """Generate up to three frames with ordered fallbacks and source manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_list = list(prompts or [])[:3] or ["Cinematic technology background"]
    api_key = get_gemini_key()
    configured = os.environ.get("GEMINI_IMAGE_MODELS", "").strip()
    models = [m.strip() for m in configured.split(",") if m.strip()] if configured else [
        get_gemini_image_model(), "gemini-3.1-flash-image", "gemini-3-pro-image"
    ]
    # Keep model order deterministic while removing accidental duplicates.
    models = list(dict.fromkeys(models))
    manifest = []
    frames = []

    for i, prompt in enumerate(prompt_list):
        out_path = out_dir / f"broll_{i}.png"
        source, used_model = "gradient", None
        if api_key:
            for model in models:
                try:
                    log(f"Generating b-roll frame {i + 1}/{len(prompt_list)} with {model}...")
                    _generate_image_gemini(str(prompt), out_path, api_key, model)
                    _save_portrait(out_path)
                    source, used_model = "gemini", model
                    break
                except Exception as exc:
                    log(f"Gemini image model {model} failed: {exc}")
        else:
            log("GEMINI_API_KEY not set; trying configured non-AI fallbacks.")

        if source != "gemini":
            if _pexels_fallback(str(prompt), out_path):
                source = "pexels"
            else:
                _fallback_frame(i, out_dir).replace(out_path)
                source = "gradient"
        frames.append(out_path)
        manifest.append({
            "frame": out_path.name,
            "prompt": str(prompt),
            "source": source,
            "model": used_model,
        })

    manifest_path = out_dir / "broll_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"B-roll source manifest saved: {manifest_path}")
    return frames


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
