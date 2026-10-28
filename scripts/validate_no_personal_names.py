"""Heuristic no-personal-names gate for public-facing draft content."""
import argparse
import json
import re
import sys
from pathlib import Path

ALLOWED = {
    "AI", "API", "CEO", "NASA", "YouTube", "TikTok", "Instagram", "Google",
    "Gemini", "OpenAI", "Microsoft", "Apple", "Meta", "Amazon", "Nvidia",
    "Anthropic", "ChatGPT", "GitHub", "Reddit", "Pexels", "Python", "Android",
    "Linux", "Windows", "Cloudflare", "DYPOL", "Signal", "Shorts", "Reels",
    "The", "This", "That", "These", "But", "Here", "What", "Why", "How",
    "When", "Now", "Today", "More", "Traditional", "Quality", "Adapt", "Most",
    "First", "New", "Live", "Video", "Technology", "Tech", "Artificial",
    "Intelligence", "Machine", "Learning", "Generative", "Autonomous",
    "Cinematic", "Minimalist", "Photorealistic", "Electric", "Blue", "Dark",
    "Moody", "High", "Detail", "Shallow", "Future", "Work", "Replacing",
    "Editors", "Agents", "Explained", "Breaking", "Watch", "Discover", "Are", "Is", "By", "For", "And", "Of", "With", "From", "On", "In", "To", "At", "As", "Over"
}
NAME_RUN = re.compile(r"\b[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?(?:\s+[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?){1,3}\b")
NAME_HINT = re.compile(r"\b(?:Elon|Sundar|Satya|Jensen|Demis|Dario|Sam Altman|Mark Zuckerberg|Bill Gates|Steve Jobs|Tim Cook)\b", re.I)

def scan_text(text):
    text = str(text)
    hits = []
    for match in NAME_RUN.finditer(text):
        phrase = match.group(0)
        if all(word in ALLOWED for word in phrase.split()):
            continue
        if phrase in {"YouTube Shorts", "Instagram Reels", "Artificial Intelligence",
                      "Machine Learning", "Future of Work", "AI Video Agents"}:
            continue
        hits.append(phrase)
    hits.extend(match.group(0) for match in NAME_HINT.finditer(text))
    return sorted(set(hits), key=str.casefold)

def validate_draft(draft, caption_paths=()):
    fields = {}
    for key in ("script", "youtube_title", "youtube_description", "youtube_tags",
                "instagram_caption", "tiktok_caption", "thumbnail_prompt",
                "thumbnail_text", "title", "description", "tags", "captions"):
        if key in draft:
            fields[key] = str(draft[key])
    for index, value in enumerate(draft.get("broll_prompts", []) or []):
        fields["broll_prompts[" + str(index) + "]"] = str(value)
    metadata = draft.get("upload_metadata") or {}
    for key in ("title", "description", "tags"):
        if key in metadata:
            value = metadata[key]
            fields["upload_metadata." + key] = ", ".join(map(str, value)) if isinstance(value, list) else str(value)
    issues = []
    for field, value in fields.items():
        found = scan_text(value)
        if found:
            issues.append(field + ": likely personal-name text detected: " + ", ".join(found))
    for path in caption_paths:
        if not path.exists():
            issues.append(str(path) + ": caption file missing")
            continue
        found = scan_text(path.read_text(encoding="utf-8", errors="replace"))
        if found:
            issues.append(path.name + ": likely personal-name text detected: " + ", ".join(found))
    return issues

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--captions", action="append", type=Path, default=[])
    args = parser.parse_args()
    try:
        draft = json.loads(args.draft.read_text(encoding="utf-8"))
        issues = validate_draft(draft, args.captions)
    except Exception as exc:
        print("NO-PERSONAL-NAMES GATE: ERROR: " + str(exc), file=sys.stderr)
        return 2
    if issues:
        print("NO-PERSONAL-NAMES GATE: BLOCKED")
        for issue in issues:
            print("- " + issue)
        return 1
    print("NO-PERSONAL-NAMES GATE: PASS (heuristic scan; human review remains required)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
