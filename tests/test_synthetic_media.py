"""YouTube altered/synthetic-content disclosure (status.containsSyntheticMedia)."""

import json
from pathlib import Path

import pytest

from verticals.synthetic_media import manifest_has_ai_asset, resolve_contains_synthetic_media
from verticals.upload import build_insert_body

REPO = Path(__file__).resolve().parents[1]
FLUX = {"provider": "cloudflare_workers_ai", "asset_id": "flux-1-schnell",
        "license": "AI-generated; review applicable model terms", "clip": None}
PIXABAY = {"provider": "pixabay", "asset_id": "123", "license": "Pixabay Content License",
           "clip": {"provider": "pixabay_video", "asset_id": "9"}}
WIKI = {"provider": "wikimedia_commons", "asset_id": "File:x.jpg", "license": "Public domain", "clip": None}
GRADIENT = {"provider": "gradient", "asset_id": "gradient-fallback", "license": "Original generated gradient"}


def _draft(tmp_path, assets, **extra):
    manifest = tmp_path / "broll_manifest.json"
    manifest.write_text(json.dumps({"assets": assets, "review_required": False}), encoding="utf-8")
    draft = {"job_id": "1", "youtube_title": "T", "youtube_description": "D", "youtube_tags": "a,b",
             "broll_manifest": str(manifest), "upload_metadata": {"privacy": "unlisted"}}
    draft.update(extra)
    return draft


@pytest.mark.parametrize("entry", [
    FLUX,
    {"provider": "FLUX"},
    {"provider": "ai-generated"},
    {"provider": "gemini"},
    {"provider": "pixabay", "clip": {"provider": "cloudflare_workers_ai"}},
])
def test_ai_assets_detected(entry):
    assert manifest_has_ai_asset({"assets": [PIXABAY, entry]}) is True


def test_non_ai_manifest_not_flagged():
    assert manifest_has_ai_asset({"assets": [PIXABAY, WIKI, GRADIENT]}) is False
    assert manifest_has_ai_asset({"assets": []}) is False


def test_insert_body_auto_true_with_flux(tmp_path, monkeypatch):
    monkeypatch.setenv("YOUTUBE_PRIVACY_STATUS", "unlisted")
    monkeypatch.delenv("YOUTUBE_CONTAINS_SYNTHETIC_MEDIA", raising=False)
    body = build_insert_body(_draft(tmp_path, [PIXABAY, FLUX]), "en", media_dir=tmp_path)
    assert body["status"] == {"privacyStatus": "unlisted", "selfDeclaredMadeForKids": False,
                              "containsSyntheticMedia": True}


def test_insert_body_auto_false_without_ai(tmp_path, monkeypatch):
    monkeypatch.setenv("YOUTUBE_PRIVACY_STATUS", "unlisted")
    monkeypatch.delenv("YOUTUBE_CONTAINS_SYNTHETIC_MEDIA", raising=False)
    body = build_insert_body(_draft(tmp_path, [PIXABAY, WIKI, GRADIENT]), "en", media_dir=tmp_path)
    assert body["status"]["containsSyntheticMedia"] is False
    assert body["status"]["privacyStatus"] == "unlisted"


@pytest.mark.parametrize("assets,env_value,expected", [
    ([PIXABAY], "true", True),
    ([FLUX], "false", False),
    ([FLUX], "auto", True),
    ([FLUX], "garbage", True),  # unknown value falls back to auto
])
def test_env_override(tmp_path, monkeypatch, assets, env_value, expected):
    monkeypatch.setenv("YOUTUBE_PRIVACY_STATUS", "unlisted")
    monkeypatch.setenv("YOUTUBE_CONTAINS_SYNTHETIC_MEDIA", env_value)
    body = build_insert_body(_draft(tmp_path, assets), "en", media_dir=tmp_path)
    assert body["status"]["containsSyntheticMedia"] is expected


def test_privacy_gate_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("YOUTUBE_PRIVACY_STATUS", "private")
    with pytest.raises(RuntimeError, match="visibility mismatch"):
        build_insert_body(_draft(tmp_path, [FLUX]), "en", media_dir=tmp_path)


def test_manifest_fallback_to_media_dir_and_draft_flag(tmp_path):
    work = tmp_path / "work_42_en"
    work.mkdir()
    (work / "broll_manifest.json").write_text(json.dumps({"assets": [FLUX]}), encoding="utf-8")
    draft = {"job_id": "42", "broll_manifest": "/nonexistent/broll_manifest.json"}
    assert resolve_contains_synthetic_media(draft, "en", tmp_path, env={}) is True
    # No manifest anywhere: use the flag recorded when the manifest was merged.
    assert resolve_contains_synthetic_media({"job_id": "7", "contains_ai_visuals": True}, "en", tmp_path, env={}) is True
    assert resolve_contains_synthetic_media({"job_id": "7"}, "en", tmp_path, env={}) is False


def test_insert_body_preview_script_matches(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("apply_draft_overrides", REPO / "scripts" / "apply_draft_overrides.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("VERTICALS_HOME", str(tmp_path))
    monkeypatch.delenv("YOUTUBE_CONTAINS_SYNTHETIC_MEDIA", raising=False)
    assert module.insert_body_preview(_draft(tmp_path, [FLUX]))["status"]["containsSyntheticMedia"] is True
    assert module.insert_body_preview(_draft(tmp_path, [PIXABAY]))["status"]["containsSyntheticMedia"] is False
