#!/usr/bin/env python3
"""Write a verified YouTube upload result for DYPOL Creator Console."""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


def video_id_from_url(url: str) -> str:
    value = str(url or "").strip()
    if not value:
        return ""
    patterns = [
        r"youtu\.be/([A-Za-z0-9_-]{6,})",
        r"[?&]v=([A-Za-z0-9_-]{6,})",
        r"/shorts/([A-Za-z0-9_-]{6,})",
        r"/embed/([A-Za-z0-9_-]{6,})",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    return ""


def find_verified_upload(draft: dict) -> dict | None:
    state = draft.get("_pipeline_state") or {}
    stage = state.get("upload") or {}
    if not isinstance(stage, dict) or stage.get("status") != "done":
        return None
    artifacts = stage.get("artifacts") or {}
    candidates = [
        artifacts.get("youtube_url"),
        artifacts.get("url"),
        draft.get("youtube_url_en"),
        draft.get("youtube_url_hi"),
        draft.get("youtube_url_es"),
        draft.get("youtube_url_pt"),
        draft.get("youtube_url_de"),
        draft.get("youtube_url_fr"),
        draft.get("youtube_url_ja"),
        draft.get("youtube_url_ko"),
    ]
    for url in candidates:
        video_id = video_id_from_url(str(url or ""))
        if video_id:
            return {"youtube_id": video_id, "youtube_url": str(url), "title": (draft.get("upload_metadata") or {}).get("title") or draft.get("youtube_title", "")}
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", required=True)
    parser.add_argument("--job-id", default="")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--producing-run-id", default="")
    parser.add_argument("--draft-source", default="same-run")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    draft_path = Path(args.draft)
    try:
        draft = json.loads(draft_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Cannot read upload draft: {exc}", file=sys.stderr)
        return 2
    found = find_verified_upload(draft)
    if not found:
        print("No verified YouTube ID found in completed upload state; no result file written.")
        return 0

    meta = draft.get("upload_metadata") or {}
    visibility = str(meta.get("privacy") or "private")
    result = {
        "job_id": args.job_id or str(draft.get("job_id") or ""),
        "run_id": args.run_id,
        "producing_run_id": args.producing_run_id,
        "draft_source": args.draft_source,
        "status": "uploaded",
        "visibility": visibility,
        "youtube_id": found["youtube_id"],
        "youtube_url": found["youtube_url"],
        "title": found["title"],
        "video_count": 1,
        "videos": [{
            "kind": "short",
            "youtube_id": found["youtube_id"],
            "youtube_url": found["youtube_url"],
            "title": found["title"],
            "visibility": visibility,
        }],
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Verified upload result written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
