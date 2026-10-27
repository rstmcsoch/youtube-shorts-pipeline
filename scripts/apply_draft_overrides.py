#!/usr/bin/env python3
"""Validate an approved Verticals draft and apply review-panel metadata."""

import argparse
import json
import sys
from pathlib import Path

VALID_VISIBILITY = {"private", "unlisted", "public"}


def load_and_validate(path: Path, lang: str = "") -> dict:
    if not path.is_file():
        raise ValueError(f"Approved draft not found: {path}. No upload was attempted.")
    try:
        draft = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Approved draft JSON is unreadable: {exc}. No upload was attempted.")
    if not isinstance(draft, dict) or not isinstance(draft.get("_pipeline_state"), dict):
        raise ValueError("Draft is missing pipeline state. No upload was attempted.")

    state = draft["_pipeline_state"]
    assemble = state.get("assemble", {})
    if not isinstance(assemble, dict) or assemble.get("status") != "done":
        raise ValueError("Draft has no completed assemble stage. No upload was attempted.")

    video_keys = [f"video_{lang}"] if lang else [k for k in draft if k.startswith("video_") and k not in {"video_path"}]
    video_path = next((str(draft.get(k) or "") for k in video_keys if draft.get(k)), "")
    if not video_path or not Path(video_path).is_file():
        raise ValueError(f"Produced video file is missing ({video_path or 'no video path'}). No upload was attempted.")
    return draft


def apply_overrides(draft: dict, title: str, description: str, tags: str, privacy: str) -> dict:
    visibility = str(privacy or "private").strip().lower()
    if visibility not in VALID_VISIBILITY:
        raise ValueError(f"Invalid privacy setting {visibility!r}; expected private, unlisted, or public.")
    meta = draft.setdefault("upload_metadata", {})
    if title.strip():
        meta["title"] = title.strip()[:100]
    else:
        meta.setdefault("title", str(draft.get("youtube_title") or draft.get("news") or "Untitled Video")[:100])
    if description.strip():
        meta["description"] = description.strip()
    else:
        meta.setdefault("description", str(draft.get("youtube_description") or ""))
    if tags.strip():
        meta["tags"] = tags.strip()
    else:
        meta.setdefault("tags", str(draft.get("youtube_tags") or ""))
    meta["privacy"] = visibility

    # Ensure cmd_upload performs the reviewed upload instead of skipping a prior upload stage.
    draft.setdefault("_pipeline_state", {})["upload"] = {"status": "pending"}
    return draft


def insert_body_preview(draft: dict) -> dict:
    meta = draft.get("upload_metadata") or {}
    tags = meta.get("tags", "")
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    return {
        "snippet": {
            "title": str(meta.get("title") or draft.get("youtube_title") or draft.get("news") or "Untitled Video")[:100],
            "description": str(meta.get("description") or ""),
            "tags": tags,
            "categoryId": "20",
        },
        "status": {
            "privacyStatus": meta.get("privacy", "private"),
            "selfDeclaredMadeForKids": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", required=True)
    parser.add_argument("--lang", default="")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--title", default="")
    parser.add_argument("--description", default="")
    parser.add_argument("--tags", default="")
    parser.add_argument("--privacy", default="private")
    parser.add_argument("--print-insert-body", action="store_true")
    args = parser.parse_args()
    path = Path(args.draft)

    try:
        draft = load_and_validate(path, args.lang)
        if args.apply:
            draft = apply_overrides(draft, args.title, args.description, args.tags, args.privacy)
            path.write_text(json.dumps(draft, indent=2, ensure_ascii=False), encoding="utf-8")
        if args.print_insert_body:
            body = insert_body_preview(draft)
            print(json.dumps(body, indent=2, ensure_ascii=False))
        if args.validate:
            print(f"Approved draft validated: {path}")
        return 0
    except Exception as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
