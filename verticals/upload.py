"""YouTube API upload + thumbnail + captions."""

import os
from pathlib import Path

from .config import get_youtube_token_path, write_secret_file
from .log import log
from .retry import with_retry


@with_retry(max_retries=2, base_delay=5.0)
def upload_to_youtube(
    video_path: Path,
    draft: dict,
    srt_path: Path = None,
    lang: str = "en",
    thumbnail_path: Path = None,
) -> str:
    """Upload video to YouTube with metadata, captions, and optional thumbnail."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    token_path = get_youtube_token_path()
    creds = Credentials.from_authorized_user_file(str(token_path))
    if creds.expired:
        if creds.refresh_token:
            creds.refresh(Request())
            write_secret_file(token_path, creds.to_json())
        else:
            raise RuntimeError(
                "YouTube OAuth token is expired and has no refresh token.\n"
                "Re-run: python3 scripts/setup_youtube_oauth.py"
            )

    youtube = build("youtube", "v3", credentials=creds)
    log(f"Uploading {video_path.name}...")

    approved = draft.get("upload_metadata") or {}
    title = str(approved.get("title") or draft.get("youtube_title") or draft.get("news") or "Untitled Video")[:100]
    description = str(approved.get("description") if approved.get("description") is not None else draft.get("youtube_description", ""))
    raw_tags = approved.get("tags") if approved.get("tags") is not None else draft.get("youtube_tags", "")
    if isinstance(raw_tags, str):
        tags = [tag.strip() for tag in raw_tags.split(",") if tag.strip()]
    else:
        tags = [str(tag).strip() for tag in (raw_tags or []) if str(tag).strip()]
    privacy = str(approved.get("privacy") or os.environ.get("YOUTUBE_PRIVACY_STATUS", "private")).strip().lower()
    if privacy not in {"private", "unlisted", "public"}:
        raise ValueError(f"Invalid YouTube privacy status {privacy!r}; refusing upload.")
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": "20",
            "defaultLanguage": lang,
            "defaultAudioLanguage": lang,
        },
        "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False},
    }
    # Safety gate: the insert body must match both the reviewer choice and the workflow policy.
    expected_privacy = str(os.environ.get("YOUTUBE_PRIVACY_STATUS", privacy) or privacy).strip().lower()
    if privacy != expected_privacy or body["status"]["privacyStatus"] != expected_privacy:
        raise RuntimeError("YouTube insert body visibility mismatch; upload refused.")

    media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True)
    req = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        status, response = req.next_chunk()
        if status:
            log(f"Upload progress: {int(status.progress() * 100)}%")

    video_id = response["id"]
    url = f"https://youtu.be/{video_id}"
    log(f"Uploaded: {url}")

    # Upload SRT if available
    if srt_path and srt_path.exists():
        try:
            youtube.captions().insert(
                part="snippet",
                body={
                    "snippet": {
                        "videoId": video_id,
                        "language": lang,
                        "name": lang.upper(),
                        "isDraft": False,
                    }
                },
                media_body=MediaFileUpload(str(srt_path), mimetype="application/octet-stream"),
            ).execute()
            log("Captions uploaded.")
        except Exception as e:
            log(f"Caption upload failed: {e}")

    # Upload thumbnail if available
    if thumbnail_path and thumbnail_path.exists():
        try:
            youtube.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(str(thumbnail_path), mimetype="image/png"),
            ).execute()
            log("Thumbnail uploaded.")
        except Exception as e:
            log(f"Thumbnail upload failed: {e}")

    return url
