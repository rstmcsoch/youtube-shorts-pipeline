"""NVIDIA hosted FLUX.1-schnell fallback (only after Cloudflare FLUX hits its daily limit)."""
import base64
import io
import json

import pytest
from PIL import Image

from verticals import broll, visuals
from verticals.synthetic_media import manifest_has_ai_asset, resolve_contains_synthetic_media

CF_429 = '{"errors":[{"message":"you have used up your daily free allocation of 10,000 neurons"}]}'


def _jpeg_b64(color=(30, 60, 90)):
    buf = io.BytesIO()
    Image.new("RGB", (36, 64), color).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


class _Resp:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise visuals.requests.HTTPError(str(self.status_code))


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    visuals.reset_provider_limits()
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    yield
    visuals.reset_provider_limits()


def _router(cf, nv, calls):
    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append((url, json))
        return (nv if "nvidia" in url else cf)(json)
    return fake_post


def _ok_nv(body):
    return _Resp(200, {"artifacts": [{"base64": _jpeg_b64(((body["seed"] * 70) % 255, 80, 160)),
                                      "finishReason": "SUCCESS"}]})


def test_cloudflare_ok_never_calls_nvidia(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(visuals.requests, "post", _router(
        lambda b: _Resp(200, {"result": {"image": _jpeg_b64()}}), _ok_nv, calls))
    meta = visuals._ai_generate("prompt, no text", tmp_path / "a.png")
    assert meta["provider"] == "cloudflare_workers_ai"
    assert [u for u, _ in calls if "nvidia" in u] == []


def test_fallback_after_cloudflare_limit(monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr(visuals.requests, "post", _router(lambda b: _Resp(429, text=CF_429), _ok_nv, calls))
    prompt = visuals.flux_prompt({"keywords": ["fraud detection servers"]})
    meta = visuals._ai_generate(prompt, tmp_path / "a.png", seed=5)
    out = capsys.readouterr().out
    assert "DYPOL_PROVIDER_LIMIT provider=flux " in out and "provider=nvidia_flux" not in out
    assert meta["provider"] == "nvidia_flux" and meta["fallback_for"] == "cloudflare_workers_ai"
    (_, cf_body), (nv_url, nv_body) = calls
    assert nv_url == visuals.NVIDIA_FLUX_URL and nv_body["prompt"] == cf_body["prompt"] == prompt
    assert "no text" in prompt and "no logos" in prompt
    assert (nv_body["width"], nv_body["height"]) == (768, 1344) and nv_body["steps"] == 4
    calls.clear()
    visuals._ai_generate(prompt, tmp_path / "b.png", seed=6)
    assert [u for u, _ in calls] == [visuals.NVIDIA_FLUX_URL]   # Cloudflare skipped for the rest of the run


def test_nvidia_limit_logged_once_then_gradient(monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr(visuals.requests, "post", _router(
        lambda b: _Resp(429, text=CF_429), lambda b: _Resp(429, text="Too Many Requests"), calls))
    for k in range(3):
        assert visuals._ai_generate("p", tmp_path / f"{k}.png") is None
    out = capsys.readouterr().out
    assert out.count("DYPOL_PROVIDER_LIMIT provider=nvidia_flux") == 1
    assert len(calls) == 2
    assert set(visuals.provider_limits()) == {"flux", "nvidia_flux"}
    # select_visual falls through to the stock / gradient path
    monkeypatch.setattr(visuals, "_pixabay_candidates", lambda *a, **k: [])
    monkeypatch.setattr(visuals, "_wikimedia_candidates", lambda *a, **k: [])
    meta, _ = visuals.select_visual({"type": "concept", "keywords": ["fraud"], "entities": []},
                                    tmp_path / "s.png", set())
    assert meta["provider"] in {"gradient", "pixabay", "wikimedia_commons"}
    assert len(calls) == 2  # no further AI requests


def test_no_cloudflare_credentials_never_uses_nvidia(monkeypatch, tmp_path):
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN")
    calls = []
    monkeypatch.setattr(visuals.requests, "post", _router(lambda b: _Resp(500), _ok_nv, calls))
    assert visuals._ai_generate("p", tmp_path / "a.png") is None and calls == []


def test_square_retry_disable_and_throttle(monkeypatch, tmp_path):
    visuals.mark_provider_limit("flux", 429)
    sizes, sleeps = [], []
    monkeypatch.setenv("DYPOL_NVIDIA_FLUX_MIN_INTERVAL", "1.5")
    monkeypatch.setattr(visuals.time, "sleep", lambda s: sleeps.append(s))

    def nv(body):
        sizes.append((body["width"], body["height"]))
        return _Resp(422, text="bad size") if body["width"] != 1024 else _ok_nv(body)

    monkeypatch.setattr(visuals.requests, "post", _router(lambda b: _Resp(500), nv, []))
    assert visuals._ai_generate("p", tmp_path / "a.png", seed=1)["provider"] == "nvidia_flux"
    assert sizes == [(768, 1344), (1024, 1024)]
    assert sleeps and 1.0 < sleeps[-1] <= 1.5   # <= 40 requests / minute
    calls = []
    monkeypatch.setattr(visuals.requests, "post", _router(lambda b: _Resp(500), lambda b: _Resp(403), calls))
    assert visuals._ai_generate("p", tmp_path / "b.png") is None
    assert visuals._ai_generate("p", tmp_path / "c.png") is None
    assert len(calls) == 1 and "nvidia_flux" not in visuals.provider_limits()


def test_top_up_records_nvidia_provider(monkeypatch, tmp_path):
    visuals.mark_provider_limit("flux", 429)
    monkeypatch.setattr(visuals.requests, "post", _router(lambda b: _Resp(500), _ok_nv, []))
    frame = tmp_path / "broll_0.png"
    Image.new("RGB", (1080, 1920)).save(frame)
    (tmp_path / "broll_manifest.json").write_text(json.dumps({"assets": [
        {"frame": frame.name, "provider": "pixabay", "max_uses": 1, "segment_index": 0,
         "intent": {"keywords": ["fraud detection"]}}]}), encoding="utf-8")
    added = broll.top_up_ai_frames(tmp_path, 4)
    data = json.loads((tmp_path / "broll_manifest.json").read_text())
    ai_rows = [r for r in data["assets"] if r.get("top_up")]
    assert added >= 1 and all(r["provider"] == "nvidia_flux" for r in ai_rows)
    assert all(r["fallback_for"] == "cloudflare_workers_ai" for r in ai_rows)
    assert "AI-generated (FLUX)" in data["credits"]
    assert manifest_has_ai_asset(data) is True


@pytest.mark.parametrize("provider, expected", [
    ("nvidia_flux", True), ("cloudflare_workers_ai", True), ("gemini", True),
    ("pixabay", False), ("pixabay_video", False), ("wikimedia_commons", False), ("gradient", False),
])
def test_disclosure_by_provider(tmp_path, provider, expected):
    manifest = tmp_path / "broll_manifest.json"
    manifest.write_text(json.dumps({"assets": [{"provider": "pixabay"}, {"provider": provider}],
                                    "audio": {"music": {"provider": "open_lofi", "license": "CC0-1.0"}}}))
    draft = {"job_id": "j", "broll_manifest": str(manifest)}
    assert resolve_contains_synthetic_media(draft, env={}) is expected


def test_draft_metadata_records_disclosure(tmp_path, monkeypatch):
    from verticals.__main__ import _apply_visual_manifest
    monkeypatch.delenv("YOUTUBE_CONTAINS_SYNTHETIC_MEDIA", raising=False)
    for providers, expected in ((["pixabay", "nvidia_flux"], True), (["pixabay", "gradient"], False)):
        manifest = tmp_path / "broll_manifest.json"
        manifest.write_text(json.dumps({"assets": [{"provider": p} for p in providers]}))
        draft_path = tmp_path / "d.json"
        draft = {"job_id": "j", "youtube_description": "x"}
        _apply_visual_manifest(draft, draft_path, manifest)
        saved = json.loads(draft_path.read_text())
        assert saved["contains_synthetic_media"] is expected
        assert saved["image_providers"] == sorted(providers)
