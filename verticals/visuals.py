"""Reusable rights-aware still-image providers for DYPOL verticals.

No social/news-site scraping. Provider failures are non-fatal; rights are an
eligibility gate, not a relevance-score component.
"""
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image

ALLOWED_WIKIMEDIA_LICENSES = {"public domain", "pd", "cc0", "cc-zero", "public domain mark", "pd-mark"}
TIMEOUT = (5, 15)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _tokens(value):
    return {t.casefold() for t in re.findall(r"[A-Za-z0-9]{3,}", str(value or ""))}


# Words too generic to prove a stock hit is on-topic (colours, moods, framing).
WEAK_TERMS = {
    "dark", "blue", "red", "green", "black", "white", "light", "lights", "glow", "glowing",
    "ambient", "abstract", "background", "cinematic", "modern", "moody", "neon", "bright",
    "close", "closeup", "shot", "view", "scene", "photo", "image", "concept", "style",
    "high", "detail", "detailed", "new", "big", "large", "small", "the", "and", "with", "for",
}
# Subjects that are almost never right for a tech/news segment unless the
# segment itself is about them (the "kitten for cybersecurity" failure).
OFF_TOPIC_SUBJECTS = {
    "cat", "cats", "kitten", "kittens", "kitty", "dog", "dogs", "puppy", "puppies", "pet", "pets",
    "animal", "animals", "bird", "birds", "horse", "horses", "wildlife", "mammal", "rabbit", "hamster",
    "portrait", "woman", "women", "man", "men", "girl", "girls", "boy", "boys", "face", "faces",
    "model", "selfie", "beauty", "baby", "child", "children", "kid", "kids", "fashion", "bride",
    "wedding", "people", "person",
}


def _norm_tokens(value):
    """Lower-cased word tokens with a naive plural strip, for tag matching."""
    out = set()
    for t in re.findall(r"[A-Za-z0-9]{2,}", str(value or "")):
        t = t.casefold()
        out.add(t)
        if len(t) > 3 and t.endswith("s"):
            out.add(t[:-1])
    return out


def topic_terms(intent):
    """Strong (non-generic) terms describing what a segment is about."""
    query = " ".join([*(intent.get("keywords") or []), *(intent.get("entities") or [])])
    return {t for t in _norm_tokens(query) if t not in WEAK_TERMS and len(t) >= 3}


def tag_overlap(candidate, intent):
    """Strong query/topic terms that also appear in a stock hit's tags/title."""
    return topic_terms(intent) & (_norm_tokens(candidate.get("title", "")) - WEAK_TERMS)


def off_topic_subject(candidate, intent):
    """True when a hit shows animals/pets/portraits the segment is not about."""
    tags = _norm_tokens(candidate.get("title", ""))
    hits = tags & OFF_TOPIC_SUBJECTS
    if not hits:
        return False
    about = _norm_tokens(" ".join([
        str(intent.get("segment") or ""), " ".join(intent.get("keywords") or []),
        " ".join(intent.get("entities") or []),
    ]))
    return not (hits & about)


# A tech-niche stock hit must show something technological. Generic words such as
# "defense" or "silhouette" matched an aikido photo and a sunset in field tests.
TECH_ANCHORS = {
    "computer", "computers", "server", "servers", "network", "networks", "data", "datacenter",
    "code", "coding", "programming", "programmer", "software", "hardware", "cyber", "cybersecurity",
    "security", "hacker", "hacking", "technology", "tech", "digital", "chip", "chips", "processor",
    "cpu", "gpu", "circuit", "electronics", "electronic", "laptop", "keyboard", "monitor", "screen",
    "internet", "cable", "cables", "fiber", "optic", "robot", "robotics", "artificial", "intelligence",
    "ai", "algorithm", "smartphone",     "binary", "matrix", "router", "padlock", "encryption", "firewall", "virus",
    "malware", "password", "website", "dashboard", "analytics", "automation", "engineer", "engineering", "semiconductor", "silicon", "microchip", "motherboard",
}


def tech_anchored(candidate):
    return bool(_norm_tokens(candidate.get("title", "")) & TECH_ANCHORS)


def stock_candidate_ok(candidate, intent):
    """Relevance gate for Pixabay stills/videos: topical tag overlap, a visible
    technology subject (tech niche), and no off-topic subjects."""
    if not tag_overlap(candidate, intent) or off_topic_subject(candidate, intent):
        return False
    if (intent.get("niche") or "tech") == "tech" and not tech_anchored(candidate):
        return False
    return True


def mean_luma(path, video=False):
    """Average brightness 0-255 of an image or (sampled) video; None if unknown."""
    import subprocess
    try:
        args = ["ffmpeg", "-v", "error", "-i", str(path)]
        args += ["-vf", "fps=2,scale=64:-2,format=gray", "-f", "rawvideo", "-"] if video else \
                ["-vf", "scale=64:-2,format=gray", "-frames:v", "1", "-f", "rawvideo", "-"]
        out = subprocess.run(args, capture_output=True, timeout=60).stdout
        return (sum(out) / len(out)) if out else None
    except Exception:
        return None


MIN_LUMA = float(__import__("os").environ.get("MIN_VISUAL_LUMA", "38"))


def too_dark(path, video=False):
    value = mean_luma(path, video=video)
    return value is not None and value < MIN_LUMA


def relevance_score(candidate, intent, target=(9, 16)):
    """Deterministic relevance/portrait-fit score; duplicates are filtered separately."""
    query = " ".join([*(intent.get("entities") or []), *(intent.get("keywords") or [])])
    terms = _tokens(query)
    title = _tokens(candidate.get("title", ""))
    entity_terms = _tokens(" ".join(intent.get("entities") or []))
    entity_match = len(entity_terms & title) / max(1, len(entity_terms))
    keyword_match = len((_tokens(" ".join(intent.get("keywords") or []))) & title) / max(1, len(_tokens(" ".join(intent.get("keywords") or []))))
    width, height = candidate.get("width", 0), candidate.get("height", 0)
    ratio = (width / height) if width and height else 0
    orientation_fit = 1.0 - min(1.0, abs(ratio - (target[0] / target[1])))
    return round(0.55 * entity_match + 0.30 * keyword_match + 0.15 * orientation_fit, 4)


def _allowed_wikimedia_license(info):
    short = str(info.get("shortname") or "").strip().casefold()
    name = str(info.get("name") or "").strip().casefold()
    url = str(info.get("url") or "").casefold()
    text = " ".join((short, name, url))
    # Explicit allowlist only. Reject attribution/share-alike licenses first.
    if any(term in text for term in ("cc-by", "cc by", "attribution", "share-alike", "share alike")):
        return False
    if short in {"cc0", "cc zero", "pd-mark", "public domain mark", "public domain"}:
        return True
    if "public domain" in name or "public domain" in short:
        return True
    if "creativecommons.org/publicdomain/zero/" in url:
        return True
    if "pd-mark" in url or "publicdomain/mark/" in url:
        return True
    return False


def _download_candidate(candidate, output_path):
    response = requests.get(candidate["download_url"], timeout=TIMEOUT)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    with Image.open(output_path) as im:
        im.verify()
    with Image.open(output_path) as im:
        width, height = im.size
        im.convert("RGB").save(output_path, format="PNG")
    candidate["width"], candidate["height"] = width, height
    candidate["sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()
    candidate["retrieved_at"] = _now()
    return candidate


def _wikimedia_candidates(query):
    params = {
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}", "gsrnamespace": 6,
        "gsrlimit": 8, "prop": "imageinfo", "iiprop": "url|size|extmetadata",
        "iiurlwidth": 1200,
    }
    response = requests.get("https://commons.wikimedia.org/w/api.php", params=params, timeout=TIMEOUT,
                            headers={"User-Agent": "DYPOLCreatorConsole/1.0 (visual asset lookup)"})
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})
    out = []
    for page in pages.values():
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}
        license_info = meta.get("LicenseShortName") or {}
        license_name = license_info.get("value", "")
        license_url = (meta.get("LicenseUrl") or {}).get("value", "")
        if not _allowed_wikimedia_license({"shortname": license_name, "url": license_url,
                                           "name": (meta.get("UsageTerms") or {}).get("value", "")}):
            continue
        # Never publish creator names; only the generic source/rights category is retained.
        out.append({
            "provider": "wikimedia_commons", "source_url": info.get("descriptionurl", ""),
            "asset_id": str(page.get("pageid", "")), "license": license_name or "Public domain/CC0",
            "license_url": license_url, "download_url": info.get("thumburl") or info.get("url"),
            "title": re.sub(r"<[^>]+>", " ", page.get("title", "")),
            "width": info.get("thumbwidth") or info.get("width", 0),
            "height": info.get("thumbheight") or info.get("height", 0),
        })
    return out


def _pixabay_candidates(query):
    key = os.environ.get("PIXABAY_API_KEY", "").strip()
    if not key:
        return []
    cache_dir = Path(os.environ.get("VERTICALS_HOME", str(Path.home() / ".verticals"))) / "cache" / "pixabay"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(query.casefold().encode()).hexdigest()
    cache_file = cache_dir / f"{cache_key}.json"
    data = None
    # Pixabay API terms require caching API results for 24 hours.
    if cache_file.exists() and time.time() - cache_file.stat().st_mtime < 86400:
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
    if data is None:
        try:
            response = requests.get("https://pixabay.com/api/", params={
                "key": key, "q": query[:100], "image_type": "photo",
                "orientation": "vertical", "safesearch": "true", "per_page": 12,
            }, timeout=TIMEOUT)
            if response.status_code == 429:
                return []
            response.raise_for_status()
            data = response.json()
            cache_file.write_text(json.dumps(data), encoding="utf-8")
        except (requests.RequestException, ValueError, OSError):
            return []
    candidates = []
    for hit in data.get("hits", []):
        candidates.append({
            "provider": "pixabay", "source_url": hit.get("pageURL", ""),
            "asset_id": str(hit.get("id", "")), "license": "Pixabay Content License",
            "license_url": "https://pixabay.com/service/license-summary/",
            "download_url": hit.get("largeImageURL") or hit.get("webformatURL"),
            "title": hit.get("tags", ""), "width": hit.get("imageWidth", 0),
            "height": hit.get("imageHeight", ""),
        })
    return candidates


# Company/product names FLUX tends to paint as fake logos. Entity names from the
# intent are stripped too; this list catches brands that leak via keywords.
BRAND_TERMS = {
    "google", "deepmind", "gemini", "alphabet", "openai", "chatgpt", "gpt", "microsoft", "copilot",
    "azure", "apple", "iphone", "ipad", "macbook", "nvidia", "geforce", "amd", "intel", "meta",
    "facebook", "instagram", "whatsapp", "amazon", "aws", "alexa", "tesla", "spacex", "samsung",
    "galaxy", "anthropic", "claude", "xai", "grok", "twitter", "tiktok", "youtube", "netflix",
    "sony", "playstation", "xbox", "nintendo", "huawei", "xiaomi", "oneplus", "pixel", "android",
    "windows", "linux", "ibm", "oracle", "cisco", "qualcomm", "snapdragon", "arm", "tsmc", "adobe",
    "salesforce", "uber", "airbnb", "paypal", "visa", "mastercard", "bitcoin", "ethereum", "perplexity",
    "mistral", "llama", "deepseek", "baidu", "alibaba", "tencent", "bytedance", "reddit", "discord",
}
FLUX_NEGATIVE = (
    "absolutely no text, no letters, no numbers, no words, no screens with writing, "
    "no logos, no brand names, no trademarks, no signage, no watermark"
)
FLUX_FRAMINGS = [
    "", "wide establishing shot", "macro close-up detail", "low-angle shot",
    "top-down view", "shallow depth of field side view", "silhouette backlit shot",
    "slow-shutter motion blur", "symmetrical corridor perspective", "overhead light rays",
]


def flux_prompt(intent, variant=0):
    """Generic, brand-free scene description for FLUX (no company/product/entity names)."""
    banned = set(BRAND_TERMS)
    for entity in intent.get("entities") or []:
        banned |= _norm_tokens(entity)
    words = []
    for phrase in intent.get("keywords") or []:
        for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", str(phrase)):
            if not (_norm_tokens(w) & banned):
                words.append(w.lower() if not w.isupper() else w)
    scene = " ".join(dict.fromkeys(words)).strip() or "unbranded computer hardware in a dim technology lab"
    framing = FLUX_FRAMINGS[variant % len(FLUX_FRAMINGS)] if variant else ""
    lead = f"{framing}, " if framing else ""
    return (f"Vertical 9:16 cinematic photo, {lead}generic unbranded scene: {scene}. "
            f"Plain unmarked equipment, {FLUX_NEGATIVE}.")


def _cloudflare_generate(prompt, output_path):
    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    if not account or not token:
        return None
    url = f"https://api.cloudflare.com/client/v4/accounts/{quote(account, safe='')}/ai/run/@cf/black-forest-labs/flux-1-schnell"
    try:
        response = requests.post(url, headers={"Authorization": f"Bearer {token}"},
                                 json={"prompt": prompt},
                                 timeout=(8, 60))
        if response.status_code == 429:
            return None
        response.raise_for_status()
        data = response.json()
        image_data = data.get("result", {}).get("image")
        if not image_data:
            return None
        import base64
        output_path.write_bytes(base64.b64decode(image_data))
        with Image.open(output_path) as im:
            im.verify()
        return {
            "provider": "cloudflare_workers_ai", "source_url": url.split("/accounts/")[0] + "/ai/run/@cf/black-forest-labs/flux-1-schnell", "asset_id": "flux-1-schnell",
            "license": "AI-generated; review applicable model terms",
            "license_url": "https://developers.cloudflare.com/workers-ai/",
            "retrieved_at": _now(), "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        }
    except (requests.RequestException, ValueError, OSError):
        return None


def select_visual(intent, output_path, used_hashes=None):
    """Resolve one still using the intent-specific order; returns (manifest, review_required)."""
    used_hashes = used_hashes if used_hashes is not None else set()
    kind = intent.get("type", "concept")
    query = " ".join([*(intent.get("entities") or []), *(intent.get("keywords") or [])]).strip()
    query = query or str(intent.get("prompt", "technology concept"))
    providers = ["wikimedia", "pixabay"] if kind in {"news_event", "product"} else ["pixabay"]
    candidates = []
    for provider in providers:
        try:
            candidates = _wikimedia_candidates(query) if provider == "wikimedia" else _pixabay_candidates(query)
        except (requests.RequestException, ValueError, KeyError):
            candidates = []
        ranked = sorted(candidates, key=lambda c: relevance_score(c, intent), reverse=True)
        for candidate in ranked:
            if not candidate.get("download_url"):
                continue
            score = relevance_score(candidate, intent)
            if score < 0.20:
                continue
            if candidate.get("provider") == "pixabay" and not stock_candidate_ok(candidate, intent):
                continue
            if int(candidate.get("width") or 0) < 360 or int(candidate.get("height") or 0) < 640:
                continue
            if kind in {"news_event", "product"} and intent.get("entities"):
                entity_terms = _tokens(" ".join(intent.get("entities") or []))
                title_terms = _tokens(candidate.get("title", ""))
                entity_match = len(entity_terms & title_terms) / max(1, len(entity_terms))
                if entity_match < 0.25:
                    continue
            try:
                metadata = _download_candidate(candidate, output_path)
                if too_dark(output_path):
                    output_path.unlink(missing_ok=True)
                    continue
                digest = metadata["sha256"]
                if digest in used_hashes:
                    output_path.unlink(missing_ok=True)
                    continue
                used_hashes.add(digest)
                metadata.pop("download_url", None)
                metadata["relevance_score"] = relevance_score(candidate, intent)
                return metadata, False
            except (requests.RequestException, OSError, ValueError):
                output_path.unlink(missing_ok=True)
                continue
    ai = _cloudflare_generate(flux_prompt(intent), output_path)
    if ai:
        digest = ai["sha256"]
        if digest not in used_hashes:
            used_hashes.add(digest)
            return ai, False
        output_path.unlink(missing_ok=True)
    output_path.unlink(missing_ok=True)
    return {"provider": "gradient", "source_url": "", "asset_id": "gradient-fallback",
            "license": "Original generated gradient", "license_url": "", "retrieved_at": _now(),
            "sha256": ""}, bool(kind in {"news_event", "product"} and intent.get("real_image_essential"))


# ─────────────────────────────────────────────────────
# Pixabay VIDEO tier (Pixabay Content License; motion b-roll)
# ─────────────────────────────────────────────────────

PIXABAY_LICENSE_URL = "https://pixabay.com/service/license-summary/"
CLIP_MAX_SECONDS = 4.0
CLIP_MAX_DOWNLOAD_BYTES = 40 * 1024 * 1024


def _pixabay_video_search(query):
    """Search Pixabay videos with the required 24 h cache (keyed by query)."""
    key = os.environ.get("PIXABAY_API_KEY", "").strip()
    if not key or not query:
        return {}
    cache_dir = Path(os.environ.get("VERTICALS_HOME", str(Path.home() / ".verticals"))) / "cache" / "pixabay_videos"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{hashlib.sha256(query.casefold().encode()).hexdigest()}.json"
    if cache_file.exists() and time.time() - cache_file.stat().st_mtime < 86400:
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    try:
        response = requests.get("https://pixabay.com/api/videos/", params={
            "key": key, "q": query[:100], "video_type": "film",
            "safesearch": "true", "per_page": 15,
        }, timeout=TIMEOUT)
        if response.status_code == 429:
            return {}
        response.raise_for_status()
        data = response.json()
        cache_file.write_text(json.dumps(data), encoding="utf-8")
        return data
    except (requests.RequestException, ValueError, OSError):
        return {}


def video_queries(intent):
    """Short (1-3 word) Pixabay queries from keywords; entity/brand names excluded."""
    banned = set(BRAND_TERMS)
    for entity in intent.get("entities") or []:
        banned |= _norm_tokens(entity)
    queries = []
    for phrase in intent.get("keywords") or []:
        words = [w for w in re.findall(r"[A-Za-z0-9]+", str(phrase)) if not (_norm_tokens(w) & banned)]
        q = " ".join(words[:3]).strip()
        if q and q.casefold() not in {x.casefold() for x in queries} and (_norm_tokens(q) - WEAK_TERMS):
            queries.append(q)
    return queries[:2]


def _pick_rendition(videos):
    """Prefer medium (≈1080p); fall back to small if medium is missing or too big."""
    for name in ("medium", "small"):
        r = (videos or {}).get(name) or {}
        if r.get("url") and int(r.get("size") or 0) <= CLIP_MAX_DOWNLOAD_BYTES:
            return name, r
    return None, None


def pixabay_video_candidates(intent):
    out = []
    for query in video_queries(intent):
        for hit in (_pixabay_video_search(query) or {}).get("hits", []):
            rendition_name, rendition = _pick_rendition(hit.get("videos"))
            if not rendition:
                continue
            out.append({
                "provider": "pixabay_video", "source_url": hit.get("pageURL", ""),
                "asset_id": str(hit.get("id", "")), "license": "Pixabay Content License",
                "license_url": PIXABAY_LICENSE_URL, "download_url": rendition["url"],
                "rendition": rendition_name, "title": hit.get("tags", ""),
                "width": int(rendition.get("width") or 0), "height": int(rendition.get("height") or 0),
                "duration": float(hit.get("duration") or 0), "query": query,
            })
        if out:
            break
    return out


def _probe_duration(path):
    import subprocess
    r = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, timeout=30)
    return float((r.stdout or "0").strip() or 0)


def normalize_clip(raw_path, out_path, width=1080, height=1920, max_seconds=CLIP_MAX_SECONDS, fps=30):
    """Trim to ≤max_seconds and scale/crop to a 1080x1920 30 fps silent H.264 clip."""
    import subprocess
    duration = _probe_duration(raw_path)
    if duration < 2.0:
        raise ValueError("clip too short")
    start = 1.0 if duration >= max_seconds + 1.5 else 0.0  # skip fade-ins
    length = min(max_seconds, duration - start)
    vf = (f"scale={width}:{height}:force_original_aspect_ratio=increase,"
          f"crop={width}:{height},fps={fps},setsar=1,format=yuv420p")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", str(raw_path),
        "-t", f"{length:.2f}", "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "20", "-movflags", "+faststart", str(out_path),
    ], check=True, capture_output=True, timeout=180)
    return round(_probe_duration(out_path), 3)


def select_video_clip(intent, out_path, used_ids=None):
    """Best relevant Pixabay clip for a segment → (metadata | None). Never raises."""
    used_ids = used_ids if used_ids is not None else set()
    try:
        candidates = pixabay_video_candidates(intent)
    except Exception:
        return None
    ranked = sorted(
        (c for c in candidates if c.get("duration", 0) >= 3 and c["asset_id"] not in used_ids
         and stock_candidate_ok(c, intent)),
        key=lambda c: (len(tag_overlap(c, intent)), c["height"] > c["width"], relevance_score(c, intent)),
        reverse=True,
    )
    raw = Path(out_path).with_name(Path(out_path).stem + "_raw.mp4")
    for candidate in ranked[:3]:
        try:
            with requests.get(candidate["download_url"], stream=True, timeout=(5, 60)) as r:
                r.raise_for_status()
                size = 0
                with open(raw, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        size += len(chunk)
                        if size > CLIP_MAX_DOWNLOAD_BYTES:
                            raise ValueError("clip download too large")
                        f.write(chunk)
            clip_seconds = normalize_clip(raw, out_path)
            if too_dark(out_path, video=True):
                raise ValueError("clip too dark")
            used_ids.add(candidate["asset_id"])
            meta = {k: candidate[k] for k in ("provider", "source_url", "asset_id", "license", "license_url",
                                                "rendition", "query")}
            meta.update({
                "file": Path(out_path).name, "duration_seconds": clip_seconds,
                "source_width": candidate["width"], "source_height": candidate["height"],
                "sha256": hashlib.sha256(Path(out_path).read_bytes()).hexdigest(),
                "retrieved_at": _now(), "relevance_score": relevance_score(candidate, intent),
                "matched_tags": sorted(tag_overlap(candidate, intent)), "max_uses": 1,
            })
            return meta
        except Exception:
            Path(out_path).unlink(missing_ok=True)
            continue
        finally:
            raw.unlink(missing_ok=True)
    return None
