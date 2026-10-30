"""Engine B integration of the AI visual judge (mocked HTTP; no network)."""
import json
from pathlib import Path

from PIL import Image

from verticals import broll
from verticals import visual_judge as vj
import verticals.visuals as visuals

GOOD = {"description": "server racks", "subject_match": "yes", "issues": [], "reason": "fits", "score": 8}
MEH = {"description": "circuit", "subject_match": "partial", "issues": ["generic_filler"], "reason": "x", "score": 5}
CAT = {"description": "kitten", "subject_match": "no", "issues": ["wrong_subject"], "reason": "cat", "score": 1}
RIOT = {"description": "riot police", "subject_match": "no", "issues": ["sensitive_content"], "reason": "x",
        "score": 1}


def _resp(v):
    return 200, {}, json.dumps({"choices": [{"message": {"content": json.dumps(v)}}]})


def _install(responses):
    queue = list(responses)
    bodies = []

    def transport(url, headers, body, timeout):
        bodies.append(json.loads(body))
        return queue.pop(0) if len(queue) > 1 else queue[0]
    judge = vj.VisualJudge(env={"NVIDIA_API_KEY": "nvapi-test", "VISUAL_JUDGE_MODELS": "m1"},
                           transport=transport, sleep=lambda s: None, store=vj.JudgeStore(vj.MemoryBackend()))
    vj.reset_judge(judge)
    return judge, bodies


INTENT = {"type": "concept", "segment": "Racks of servers hum inside the data center.", "entities": [],
          "keywords": ["server", "data", "center"], "real_image_essential": False}


def _cand(i, tags="server, data, center, computer"):
    return {"provider": "pixabay", "source_url": f"https://pixabay.com/{i}", "asset_id": str(i),
            "license": "Pixabay Content License", "license_url": "", "download_url": f"https://example.test/{i}.jpg",
            "title": tags, "width": 1080, "height": 1920}


def _patch(monkeypatch, cands):
    monkeypatch.setattr(visuals, "_pixabay_candidates", lambda q: list(cands))
    monkeypatch.setattr(visuals, "_wikimedia_candidates", lambda q: [])
    monkeypatch.setattr(visuals, "too_dark", lambda p, video=False: False)
    monkeypatch.setattr(visuals, "_cloudflare_generate", lambda prompt, path: None)
    downloads = []

    def fake_download(candidate, path):
        downloads.append(candidate["download_url"])
        n = len(downloads)
        Image.new("RGB", (90, 160), (n * 41 % 255, n * 83 % 255, 150)).save(path)
        candidate["sha256"] = visuals.hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return candidate
    monkeypatch.setattr(visuals, "_download_candidate", fake_download)
    return downloads


def test_select_visual_rejects_then_accepts(monkeypatch, tmp_path):
    judge, bodies = _install([_resp(CAT), _resp(GOOD)])
    downloads = _patch(monkeypatch, [_cand(1), _cand(2), _cand(3)])
    meta, review = visuals.select_visual(dict(INTENT), tmp_path / "broll_0.png", set(),
                                         ctx={"video_title": "Data centers"})
    assert meta["asset_id"] == "2" and review is False
    assert meta["judge"]["decision"] == "PASS" and meta["judge"]["judge_model"] == "m1"
    assert len(downloads) == 2 and "VIDEO TOPIC: Data centers" in bodies[0]["messages"][1]["content"][0]["text"]
    assert (tmp_path / "broll_0.png").exists()
    assert judge.is_rejected("pixabay:1", "server data center")


def test_select_visual_marginal_then_review(monkeypatch, tmp_path):
    _install([_resp(MEH), _resp(CAT)])
    _patch(monkeypatch, [_cand(1), _cand(2)])
    meta, review = visuals.select_visual(dict(INTENT), tmp_path / "broll_0.png", set())
    assert meta["asset_id"] == "1" and review is True
    assert meta["judge"]["decision"] == "MARGINAL" and meta["judge"]["needs_review"]
    assert (tmp_path / "broll_0.png").exists() and not (tmp_path / "broll_0_marginal.png").exists()


def test_all_rejected_falls_back_to_gradient_with_review(monkeypatch, tmp_path):
    judge, bodies = _install([_resp(RIOT)])
    _patch(monkeypatch, [_cand(i) for i in range(1, 10)])
    meta, review = visuals.select_visual(dict(INTENT), tmp_path / "broll_0.png", set())
    assert meta["provider"] == "gradient" and review is True
    assert meta["judge"]["needs_review"] and "rejected all" in meta["judge"]["review_reason"]
    assert len(bodies) == 8   # 5 candidates + 3 broadened
    assert judge.is_rejected("pixabay:4", "any query at all").startswith("global:")


def test_rejection_memory_skips_download(monkeypatch, tmp_path):
    judge, _ = _install([_resp(GOOD)])
    judge.store.record_rejection("pixabay:1", "whatever", ["watermark_or_text"])
    downloads = _patch(monkeypatch, [_cand(1), _cand(2)])
    meta, _ = visuals.select_visual(dict(INTENT), tmp_path / "f.png", set())
    assert meta["asset_id"] == "2" and downloads == ["https://example.test/2.jpg"]


def test_generate_broll_manifest_records_judge(monkeypatch, tmp_path):
    _install([_resp(GOOD)])
    _patch(monkeypatch, [_cand(i) for i in range(1, 6)])
    monkeypatch.setenv("ENABLE_PIXABAY_VIDEO", "0")
    broll.generate_broll(["server room"], tmp_path, [dict(INTENT)], video_title="Data centers")
    data = json.loads((tmp_path / "broll_manifest.json").read_text())
    assert data["judge_status"] == "active" and data["judge"]["calls"] == 1
    row = data["assets"][0]
    assert row["judge"]["score"] == 8 and row["judge"]["prompt_version"] == vj.PROMPT_VERSION
    assert data["review_required"] is False


def test_generate_broll_disabled_records_status(monkeypatch, tmp_path):
    downloads = _patch(monkeypatch, [_cand(1)])
    monkeypatch.setenv("ENABLE_PIXABAY_VIDEO", "0")
    broll.generate_broll(["server room"], tmp_path, [dict(INTENT)])
    data = json.loads((tmp_path / "broll_manifest.json").read_text())
    assert data["judge_status"] == "disabled" and data["review_required"] is False
    assert "judge" not in data["assets"][0] and downloads == ["https://example.test/1.jpg"]
    assert broll.judge_contact_sheet(tmp_path, "T")["summary"]["judge_status"] == "disabled"


def test_circuit_open_flags_review_and_keeps_mechanical(monkeypatch, tmp_path):
    _install([(402, {}, "")])
    _patch(monkeypatch, [_cand(1), _cand(2)])
    monkeypatch.setenv("ENABLE_PIXABAY_VIDEO", "0")
    broll.generate_broll(["server room"], tmp_path, [dict(INTENT)])
    data = json.loads((tmp_path / "broll_manifest.json").read_text())
    assert data["assets"][0]["asset_id"] == "1"
    assert data["judge_status"] == "circuit_open" and data["review_required"] is True
    assert any("circuit open" in r for r in data["review_reasons"])


def test_contact_sheet_flags_manifest(monkeypatch, tmp_path):
    _install([_resp(GOOD), (200, {}, json.dumps({"choices": [{"message": {"content": json.dumps(
        {"bad_shots": [{"n": 1, "reason": "sea lion"}], "overall": "needs_review"})}}]}))])
    _patch(monkeypatch, [_cand(1)])
    monkeypatch.setenv("ENABLE_PIXABAY_VIDEO", "0")
    broll.generate_broll(["server room"], tmp_path, [dict(INTENT)])
    out = broll.judge_contact_sheet(tmp_path, "T")
    data = json.loads((tmp_path / "broll_manifest.json").read_text())
    assert out["contact_sheet"]["bad_shots"][0]["n"] == 1
    assert data["review_required"] and data["assets"][0]["contact_sheet_flag"] == "sea lion"
    assert any("contact sheet" in r for r in data["review_reasons"])


def test_clip_needs_pass(monkeypatch, tmp_path):
    _install([_resp(MEH)])
    monkeypatch.setattr(visuals, "pixabay_video_candidates", lambda intent: [dict(
        _cand(7), provider="pixabay_video", rendition="medium", duration=8.0, query="server")])

    class R:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def raise_for_status(self): pass
        def iter_content(self, chunk_size): yield b"data"
    monkeypatch.setattr(visuals.requests, "get", lambda *a, **k: R())
    monkeypatch.setattr(visuals, "normalize_clip", lambda raw, out: (Path(out).write_bytes(b"v"), 3.5)[1])
    monkeypatch.setattr(visuals, "too_dark", lambda p, video=False: False)
    fractions = []

    def fake_frame(clip, fraction, out, duration=None):
        fractions.append(fraction)
        Image.new("RGB", (32, 32), (9, 9, 9)).save(out, format="JPEG")
        return Path(out)
    monkeypatch.setattr(vj, "extract_clip_frame", fake_frame)
    monkeypatch.setattr(vj, "probe_duration", lambda p: 3.5)
    assert visuals.select_video_clip(dict(INTENT), tmp_path / "clip_0.mp4", set()) is None
    assert fractions == [0.5]
