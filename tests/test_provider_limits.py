"""Provider 429 / quota handling: one DYPOL_PROVIDER_LIMIT line per provider, the
source is skipped for the rest of the run, and nothing raises (non-fatal)."""
import json

import pytest

from verticals import visuals


class FakeResp:
    def __init__(self, status, text="", payload=None):
        self.status_code = status
        self.text = text
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise visuals.requests.HTTPError(f"HTTP {self.status_code}")


@pytest.fixture(autouse=True)
def _reset(monkeypatch, tmp_path):
    visuals.reset_provider_limits()
    monkeypatch.setenv("VERTICALS_HOME", str(tmp_path))
    monkeypatch.setenv("PIXABAY_API_KEY", "k")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    yield
    visuals.reset_provider_limits()


def test_pixabay_429_logs_once_and_skips_rest_of_run(monkeypatch, capsys):
    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        return FakeResp(429, "Too Many Requests")

    monkeypatch.setattr(visuals.requests, "get", fake_get)
    assert visuals._pixabay_candidates("robot arm") == []
    assert visuals._pixabay_candidates("server rack") == []
    assert visuals._pixabay_video_search("data center") == {}
    out = capsys.readouterr().out
    assert out.count("DYPOL_PROVIDER_LIMIT provider=pixabay") == 1
    assert "status=429" in out
    assert len(calls) == 1  # exhausted: no further Pixabay requests this run
    assert "pixabay" in visuals.provider_limits()


def test_flux_daily_allocation_is_a_limit(monkeypatch, capsys, tmp_path):
    calls = []

    def fake_post(url, **kw):
        calls.append(url)
        return FakeResp(429, '{"errors":[{"code":4006,"message":"you have used up your daily free allocation of 10,000 neurons"}]}')

    monkeypatch.setattr(visuals.requests, "post", fake_post)
    assert visuals._cloudflare_generate("prompt", tmp_path / "a.png") is None
    assert visuals._cloudflare_generate("prompt", tmp_path / "b.png") is None
    out = capsys.readouterr().out
    assert out.count("DYPOL_PROVIDER_LIMIT provider=flux") == 1
    assert len(calls) == 1


def test_wikimedia_429_is_non_fatal(monkeypatch, capsys):
    monkeypatch.setattr(visuals.requests, "get", lambda url, **kw: FakeResp(429, "rate limited"))
    assert visuals._wikimedia_candidates("satellite") == []
    assert visuals._wikimedia_candidates("satellite") == []
    assert capsys.readouterr().out.count("DYPOL_PROVIDER_LIMIT provider=wikimedia") == 1


def test_quota_words_in_4xx_body_count_but_plain_errors_do_not():
    assert visuals.is_limit_response(FakeResp(403, "Quota exceeded for this key"))
    assert visuals.is_limit_response(FakeResp(429, ""))
    assert not visuals.is_limit_response(FakeResp(400, "bad request: invalid q"))
    assert not visuals.is_limit_response(FakeResp(500, "quota"))  # 5xx is not a quota signal
    assert not visuals.is_limit_response(FakeResp(200, "ok"))


def test_other_providers_still_used_after_one_is_exhausted(monkeypatch, capsys):
    visuals.mark_provider_limit("pixabay", 429)
    hits = {"query": {"pages": {}}}
    monkeypatch.setattr(visuals.requests, "get", lambda url, **kw: FakeResp(200, "", hits))
    assert visuals._wikimedia_candidates("satellite") == []  # still queried, no limit line
    out = capsys.readouterr().out
    assert "provider=wikimedia" not in out


def test_manifest_records_provider_limits(monkeypatch, tmp_path):
    from verticals import broll
    visuals.mark_provider_limit("flux", 429)
    assert broll._provider_limits()["flux"]["status"] == 429
    json.dumps(broll._provider_limits())  # serialisable for broll_manifest.json
