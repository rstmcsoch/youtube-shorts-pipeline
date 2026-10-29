"""YouTube altered/synthetic-content disclosure (status.containsSyntheticMedia).

A video is declared as containing synthetic media when its b-roll manifest
includes any AI-generated asset (FLUX on Cloudflare Workers AI, Gemini images,
or anything tagged "ai-generated"). YOUTUBE_CONTAINS_SYNTHETIC_MEDIA
(auto|true|false, default auto) overrides the detection.

Pure stdlib so it is importable from scripts/ and unit tests without the
Google client libraries.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

SYNTHETIC_MEDIA_ENV = "YOUTUBE_CONTAINS_SYNTHETIC_MEDIA"
SYNTHETIC_MEDIA_MODES = ("auto", "true", "false")
# Lower-cased substrings matched against provider / asset_id / license.
AI_ASSET_MARKERS = ("flux", "cloudflare", "ai-generated", "ai_generated", "ai generated", "gemini")


def _is_ai_asset(entry: dict) -> bool:
    for key in ("provider", "asset_id", "license"):
        value = str(entry.get(key) or "").strip().lower()
        if value and any(marker in value for marker in AI_ASSET_MARKERS):
            return True
    return False


def manifest_has_ai_asset(manifest) -> bool:
    """True when any asset (or its clip) in a visual manifest is AI-generated.

    Accepts the broll_manifest.json shape ``{"assets": [...]}``, a bare list of
    entries, or nested ``clip`` / ``asset`` dicts."""
    stack = [manifest]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, dict):
            if _is_ai_asset(item):
                return True
            stack.extend(v for v in item.values() if isinstance(v, (list, dict)))
    return False


def synthetic_media_mode(env=None) -> str:
    env = os.environ if env is None else env
    mode = str(env.get(SYNTHETIC_MEDIA_ENV, "auto") or "auto").strip().lower()
    if mode in ("1", "yes", "on"):
        mode = "true"
    elif mode in ("0", "no", "off"):
        mode = "false"
    if mode not in SYNTHETIC_MEDIA_MODES:
        print(f"WARNING: invalid {SYNTHETIC_MEDIA_ENV}={mode!r}; using auto.")
        mode = "auto"
    return mode


def load_draft_manifest(draft: dict, lang: str = "en", media_dir: Path | None = None):
    """Load the b-roll manifest recorded for a draft, or None when it cannot be found."""
    candidates = []
    if draft.get("broll_manifest"):
        candidates.append(Path(str(draft["broll_manifest"])))
    if media_dir is not None and draft.get("job_id"):
        candidates.append(Path(media_dir) / f"work_{draft['job_id']}_{lang}" / "broll_manifest.json")
    for path in candidates:
        try:
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return None


def draft_has_ai_visuals(draft: dict, lang: str = "en", media_dir: Path | None = None) -> bool:
    manifest = load_draft_manifest(draft, lang, media_dir)
    if manifest is not None:
        return manifest_has_ai_asset(manifest)
    # Manifest unavailable: fall back to what was recorded on the draft when the
    # manifest was merged (flag, or the generic "AI-generated" credit line).
    if draft.get("contains_ai_visuals"):
        return True
    approved = draft.get("upload_metadata") or {}
    text = " ".join(str(x or "") for x in (draft.get("youtube_description"), approved.get("description")))
    return "ai-generated" in text.lower()


def resolve_contains_synthetic_media(draft: dict, lang: str = "en", media_dir: Path | None = None, env=None) -> bool:
    mode = synthetic_media_mode(env)
    if mode == "true":
        return True
    if mode == "false":
        return False
    return draft_has_ai_visuals(draft, lang, media_dir)
