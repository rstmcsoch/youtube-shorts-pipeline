import json
import subprocess
import sys
from pathlib import Path

from scripts.validate_no_personal_names import scan_text, validate_draft

def test_detects_person_name_in_script_and_title():
    assert scan_text("Elon Musk announced a new tool.")
    assert scan_text("Today, Alex Johnson explains AI.")

def test_allows_common_technical_brands_and_generic_title():
    assert not scan_text("AI Video Agents Are Replacing Editors")
    assert not scan_text("AI Agents Are Changing Video Editing Forever")
    assert not scan_text("Agents Are Changing Video, Editing Forever")
    assert not scan_text("Google Gemini and YouTube Shorts are tools.")


def test_still_blocks_real_person_names():
    assert scan_text("Jane Doe explains how AI agents work.")
    assert scan_text("The report was written by Jane Doe.")

def test_checks_description_tags_and_thumbnail_text():
    draft = {
        "script": "General technology explainer.",
        "youtube_title": "AI Video Tools",
        "youtube_description": "A report by Alex Johnson about new tools.",
        "youtube_tags": "AI, Alex Johnson, technology",
        "thumbnail_text": "AI Video",
        "thumbnail_prompt": "Minimalist dark technology scene",
    }
    issues = validate_draft(draft)
    assert any("youtube_description" in issue for issue in issues)
    assert any("youtube_tags" in issue for issue in issues)

def test_checks_caption_files_and_blocks_cli(tmp_path):
    draft_path = tmp_path / "draft.json"
    caption_path = tmp_path / "captions.srt"
    draft_path.write_text(json.dumps({
        "script": "General technology explainer.",
        "youtube_title": "AI Video Tools",
        "youtube_description": "A short explainer.",
        "youtube_tags": ["AI", "technology"],
        "thumbnail_text": "AI Video",
    }), encoding="utf-8")
    caption_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nAlex Johnson explains AI.\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "scripts/validate_no_personal_names.py",
         "--draft", str(draft_path), "--captions", str(caption_path)],
        text=True, capture_output=True,
    )
    assert result.returncode == 1
    assert "BLOCKED" in result.stdout
    assert "captions.srt" in result.stdout

def test_missing_caption_fails_closed(tmp_path):
    assert any("missing" in issue for issue in validate_draft({}, [tmp_path / "missing.srt"]))


def test_allows_title_case_news_phrases():
    assert not scan_text("It topped the Artificial Analysis Intelligence Index v5.0.")
    assert not scan_text("Gemini 4 Carbon Changes Cybersecurity Forever")
    assert not scan_text("Google DeepMind Launches Gemini 4 Carbon")
