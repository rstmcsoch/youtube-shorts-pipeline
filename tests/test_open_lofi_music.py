"""Open Lo-Fi (CC0) background music for Engine B Shorts."""
import hashlib
import json

from verticals import music, open_lofi


def _fake_pool(monkeypatch, data=b"fake-mp3"):
    sha = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(open_lofi, "POOL", (("a.mp3", sha), ("b.mp3", sha), ("c.mp3", sha)))
    return data


def _work(tmp_path):
    work = tmp_path / "work_job-1_en"
    work.mkdir()
    (work / "broll_manifest.json").write_text(json.dumps({"assets": [{"provider": "pixabay"}]}))
    return work


def test_on_by_default_uses_open_lofi(monkeypatch, tmp_path):
    monkeypatch.delenv("DYPOL_MUSIC_ENABLED", raising=False)
    monkeypatch.delenv("MUSIC_ENABLED", raising=False)
    monkeypatch.setenv("DYPOL_JOB_ID", "job-1")
    monkeypatch.setenv("DYPOL_MUSIC_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(music, "MUSIC_DIR", tmp_path / "no-local-tracks")
    data = _fake_pool(monkeypatch)
    monkeypatch.setattr(open_lofi, "_download_member", lambda name: data)
    monkeypatch.setattr(music, "_measure_lufs", lambda p: -18.0 if str(p).endswith("v.wav") else -12.0)
    work = _work(tmp_path)
    r = music.select_and_prepare_music(work / "v.wav", work, words=[{"word": "a", "start": 0, "end": 1}])
    assert r["gain_db"] == -32.0 and r["relative_db"] == -26.0 and r["license"] == "CC0-1.0"
    rec = json.loads((work / "broll_manifest.json").read_text())["audio"]["music"]
    assert rec["provider"] == "open_lofi" and rec["relative_db"] == -26.0 and rec["attribution_required"] is False
    # deterministic: the same job gets the same track
    again = music.select_and_prepare_music(work / "v.wav", work, words=[{"word": "a", "start": 0, "end": 1}])
    assert again["track_path"] == r["track_path"]


def test_download_failure_renders_without_music(monkeypatch, tmp_path):
    monkeypatch.setenv("DYPOL_MUSIC_ENABLED", "1")
    monkeypatch.setenv("DYPOL_MUSIC_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(music, "MUSIC_DIR", tmp_path / "none")
    _fake_pool(monkeypatch)

    def boom(name):
        raise OSError("offline")

    monkeypatch.setattr(open_lofi, "_download_member", boom)
    work = _work(tmp_path)
    assert music.select_and_prepare_music(work / "v.wav", work, words=[]) == {}
    monkeypatch.setattr(open_lofi, "_download_member", lambda n: b"tampered")
    assert music.select_and_prepare_music(work / "v.wav", work, words=[]) == {}


def test_music_record_is_not_an_ai_visual(tmp_path):
    from verticals.synthetic_media import manifest_has_ai_asset
    assert manifest_has_ai_asset({"assets": [{"provider": "pixabay"}],
                                  "audio": {"music": {"provider": "open_lofi", "license": "CC0-1.0"}}}) is False


def test_pool_is_pinned():
    assert len(open_lofi.POOL) >= 10
    assert all(n.endswith(".mp3") and len(s) == 64 for n, s in open_lofi.POOL)
    assert open_lofi.OPEN_LOFI_ZIP_URL.startswith("https://github.com/btahir/open-lofi/releases/download/")
