"""Engine B piece 5: stock-first visuals, NVIDIA FLUX circuit breaker, no gradient shots,
hook rules + validator, brand male voice, hook timing QC, stage timings. No network."""
import json
from pathlib import Path

import pytest
import requests
from PIL import Image

from verticals import assemble, broll, tts
from verticals import visuals
from verticals import draft as draft_mod
from verticals.beats import INTRO_RE, banned_phrases_in, first_sentence, hook_warnings
from verticals.qc import add_hook_timing, hook_timing
from verticals.style import hook_rules, load_style, voice_settings

INTENT = {"type": "concept", "segment": "The detector catches the rays that pass through.", "entities": [],
          "keywords": ["digital detector", "radiation"], "real_image_essential": False}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    visuals.reset_provider_limits()
    monkeypatch.delenv("DYPOL_VOICE", raising=False)
    monkeypatch.delenv("DYPOL_VOICE_SPEED", raising=False)
    yield
    visuals.reset_provider_limits()


def _png(path, color=(90, 110, 140)):
    Image.new("RGB", (720, 1280), color).save(path)


# ── 1. stock-first order ─────────────────────────────────────────────

def _install_fakes(monkeypatch, clip_for=(), stock_for=(), topic_clip=False, topic_stock=False, flux_ok=True):
    """Fake providers. Records every call in order as (provider, segment, topic?)."""
    calls = []

    def fake_clip(intent, out_path, used_ids=None, ctx=None):
        seg = int(Path(out_path).stem.split("_")[-1])
        is_topic = bool(intent.get("topic_fallback"))
        calls.append(("clip", seg, is_topic))
        if (seg in clip_for and not is_topic) or (is_topic and topic_clip):
            Path(out_path).write_bytes(b"mp4")
            return {"provider": "pixabay_video", "asset_id": f"v{seg}{'t' if is_topic else ''}",
                    "duration_seconds": 4.0, "file": Path(out_path).name}
        return None

    def fake_stock(intent, out_path, used_hashes=None, ctx=None):
        seg = int(Path(out_path).stem.split("_")[-1])
        is_topic = bool(intent.get("topic_fallback"))
        calls.append(("stock", seg, is_topic))
        if (seg in stock_for and not is_topic) or (is_topic and topic_stock):
            _png(out_path)
            return {"provider": "pixabay", "asset_id": f"p{seg}", "source_url": "", "license": "Pixabay",
                    "license_url": "", "retrieved_at": "", "sha256": f"h{seg}{is_topic}"}, False, ""
        return None, False, ""

    def fake_ai(prompt, output_path, seed=0):
        calls.append(("flux", int(Path(output_path).stem.split("_")[-1]), False))
        if not flux_ok:
            return None
        _png(output_path, (30, 30, 30))
        return {"provider": "cloudflare_workers_ai", "source_url": "", "asset_id": "flux", "license": "AI",
                "license_url": "", "retrieved_at": "", "sha256": f"flux{seed}"}

    monkeypatch.setattr(visuals, "select_video_clip", fake_clip)
    monkeypatch.setattr(visuals, "select_stock_visual", fake_stock)
    monkeypatch.setattr(visuals, "_ai_generate", fake_ai)
    monkeypatch.setattr(broll, "_get_judge", lambda: None)
    return calls


def test_clip_then_stock_then_flux_order(monkeypatch, tmp_path):
    calls = _install_fakes(monkeypatch)
    broll.generate_broll(["a"], tmp_path, [dict(INTENT)], topic="How AI reads medical X-rays")
    assert calls == [("clip", 0, False), ("clip", 0, True), ("stock", 0, False), ("stock", 0, True),
                     ("flux", 0, False)]


def test_no_flux_when_a_clip_covers_the_segment(monkeypatch, tmp_path):
    calls = _install_fakes(monkeypatch, clip_for={0})
    broll.generate_broll(["a"], tmp_path, [dict(INTENT)], topic="How AI reads medical X-rays")
    assert not [c for c in calls if c[0] == "flux"]
    row = json.loads((tmp_path / "broll_manifest.json").read_text())["assets"][0]
    assert row["clip"]["asset_id"] == "v0" and row["provider"] == "gradient"


def test_topic_stock_beats_flux(monkeypatch, tmp_path):
    calls = _install_fakes(monkeypatch, topic_stock=True)
    broll.generate_broll(["a", "b"], tmp_path, [dict(INTENT), dict(INTENT)], topic="How AI reads medical X-rays")
    assert not [c for c in calls if c[0] == "flux"]
    data = json.loads((tmp_path / "broll_manifest.json").read_text())
    assert [r["provider"] for r in data["assets"]] == ["pixabay", "pixabay"]
    assert data["video_title"] == "How AI reads medical X-rays"


def test_topic_clip_used_when_segment_has_none(monkeypatch, tmp_path):
    _install_fakes(monkeypatch, topic_clip=True, stock_for={0})
    broll.generate_broll(["a"], tmp_path, [dict(INTENT)], topic="How noise-cancelling earbuds use AI")
    row = json.loads((tmp_path / "broll_manifest.json").read_text())["assets"][0]
    assert row["clip"]["topic_fallback"] is True and row["provider"] == "pixabay"


def test_gradient_only_as_last_resort(monkeypatch, tmp_path):
    calls = _install_fakes(monkeypatch, flux_ok=False)
    broll.generate_broll(["a"], tmp_path, [dict(INTENT)], topic="How AI reads medical X-rays")
    row = json.loads((tmp_path / "broll_manifest.json").read_text())["assets"][0]
    assert row["provider"] == "gradient" and calls[-1][0] == "flux"


def test_top_up_prefers_topic_stock_over_flux(monkeypatch, tmp_path):
    calls = _install_fakes(monkeypatch, topic_stock=True)
    (tmp_path / "broll_manifest.json").write_text(json.dumps({"video_title": "How AI reads medical X-rays", "assets": [
        {"frame": "broll_0.png", "provider": "pixabay", "max_uses": 1, "segment_index": 0, "intent": dict(INTENT),
         "sha256": "x", "clip": None}]}))
    calls.clear()
    added = broll.top_up_ai_frames(tmp_path, 3)
    rows = json.loads((tmp_path / "broll_manifest.json").read_text())["assets"]
    assert added >= 2 and not [c for c in calls if c[0] == "flux"]
    assert all(r["provider"] == "pixabay" for r in rows) and rows[-1]["max_uses"] == 1


# ── topic queries and relevance ──────────────────────────────────────

def test_topic_phrases():
    assert visuals.topic_phrases("How AI reads medical X-rays")[0] == "medical x-rays"
    assert visuals.topic_phrases("How noise-cancelling earbuds use AI")[0] == "noise-cancelling earbuds"
    assert "credit" in visuals.topic_phrases("How AI detects credit card fraud")[0]
    assert "detects" not in visuals.topic_phrases("How AI detects credit card fraud")[0]
    assert visuals.topic_phrases("") == []


def test_topic_intent_relaxes_tech_anchor_but_keeps_off_topic_rule():
    t = visuals.topic_intent(dict(INTENT), "How AI reads medical X-rays")
    xray = {"provider": "pixabay", "title": "x-ray, skeleton, medical, radiology", "download_url": "u",
            "width": 1080, "height": 1920}
    kitten = dict(xray, title="kitten, cat, medical, x-ray")
    assert visuals._prefilter_ok(xray, t, "concept")
    assert not visuals._prefilter_ok(kitten, t, "concept")
    assert not visuals.stock_candidate_ok(xray, dict(INTENT))  # segment intents still need a tech anchor


def test_wikimedia_still_first_for_news(monkeypatch, tmp_path):
    order = []
    monkeypatch.setattr(visuals, "_judge", lambda: None)
    monkeypatch.setattr(visuals, "_wikimedia_candidates", lambda q: order.append("wikimedia") or [])
    monkeypatch.setattr(visuals, "_pixabay_candidates", lambda q: order.append("pixabay") or [])
    found, _, _ = visuals.select_stock_visual(dict(INTENT, type="news_event"), tmp_path / "x.png", set())
    assert found is None and order == ["wikimedia", "pixabay"]


# ── NVIDIA FLUX circuit breaker ──────────────────────────────────────

def test_nvidia_timeout_trips_breaker(monkeypatch, tmp_path):
    posts = []
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("DYPOL_NVIDIA_FLUX_MIN_INTERVAL", "0")
    visuals.mark_provider_limit("flux", 429, "Cloudflare Workers AI")

    def hang(*a, **k):
        posts.append(k.get("timeout"))
        raise requests.ReadTimeout("read timed out")
    monkeypatch.setattr(visuals.requests, "post", hang)
    assert visuals._nvidia_flux_generate("p", tmp_path / "a.png") is None
    assert visuals.provider_exhausted(visuals.NVIDIA_FLUX_PROVIDER)
    assert visuals._ai_generate("p", tmp_path / "b.png") is None
    assert len(posts) == 1 and posts[0][1] <= 45


def test_nvidia_timeout_env(monkeypatch):
    monkeypatch.setenv("DYPOL_NVIDIA_FLUX_TIMEOUT", "bad")
    assert visuals.nvidia_flux_timeout() == 40.0
    monkeypatch.setenv("DYPOL_NVIDIA_FLUX_TIMEOUT", "1")
    assert visuals.nvidia_flux_timeout() == 5.0


# ── no gradient shot (draft 5's 5.4 s shot) ──────────────────────────

DRAFT5_ROWS = [("gradient", None), ("gradient", None), ("gradient", "clip_2.mp4"), ("gradient", None),
               ("pixabay", "clip_4.mp4"), ("gradient", None), ("gradient", None), ("gradient", None)]
DRAFT5_BEATS = ["count_up", "diagram", "stock_or_ai_still", "diagram", "kinetic_text", "kinetic_text",
                "count_up", "diagram"]


def _draft5(tmp_path):
    rows, frames = [], []
    for i, (prov, clip) in enumerate(DRAFT5_ROWS):
        f = tmp_path / f"broll_{i}.png"
        _png(f)
        frames.append(f)
        row = {"frame": f.name, "provider": prov, "segment_index": i, "max_uses": 1 if prov == "pixabay" else 2,
               "clip": None}
        if clip:
            (tmp_path / clip).write_bytes(b"mp4")
            row["clip"] = {"file": clip, "provider": "pixabay_video", "duration_seconds": 4.0}
        rows.append(row)
    (tmp_path / "broll_manifest.json").write_text(json.dumps({"assets": rows}))
    data = {"count_up": {"value": 10, "label": "x"}, "diagram": {"nodes": ["A", "B"]},
            "kinetic_text": {"words": ["a", "b"], "accent": "a"}}
    beats = [{"block": b, "data": data.get(b, {}), "keywords": ["detector", "model"]} for b in DRAFT5_BEATS]
    return frames, beats


def test_draft5_plan_never_uses_a_gradient(tmp_path):
    frames, beats = _draft5(tmp_path)
    sources = assemble.build_sources(frames, tmp_path, beats)
    n = assemble.shot_count(sources, 61.4)
    plan = assemble.final_motion_shot(assemble.plan_shots(sources, 61.4, durations=[61.4 / n] * n), sources)
    used = [sources[s["source"]] for s in plan if s["source"] is not None]
    assert len(used) == len(plan)
    assert not [s for s in used if s.get("provider") == "gradient"]


def test_empty_or_gradient_slot_renders_a_narration_card(monkeypatch, tmp_path):
    rendered, stills = [], []
    monkeypatch.setattr(assemble, "MEDIA_DIR", tmp_path)
    monkeypatch.setattr(assemble, "get_audio_duration", lambda p: 8.0)
    monkeypatch.setattr(assemble, "animate_frame", lambda *a, **k: stills.append(a))
    monkeypatch.setattr(assemble, "render_block",
                        lambda kind, data, out, d, **k: rendered.append((kind, data)) or Path(out))
    monkeypatch.setattr(assemble, "_concat_copy", lambda parts, out: None)
    monkeypatch.setattr(assemble, "prepare_audio", lambda *a, **k: tmp_path / "a.wav")
    monkeypatch.setattr(assemble, "run_cmd", lambda cmd, **k: None)
    monkeypatch.setenv("DYPOL_SFX_ENABLED", "0")
    monkeypatch.setenv("DYPOL_QC_ENABLED", "0")
    f = tmp_path / "broll_0.png"
    _png(f)
    (tmp_path / "broll_manifest.json").write_text(json.dumps({"assets": [
        {"frame": f.name, "provider": "gradient", "segment_index": 0, "max_uses": 2, "clip": None}]}))
    words = [{"word": w, "start": i * 0.4, "end": i * 0.4 + 0.3}
             for i, w in enumerate("Detectors catch radiation passing through your body tissue".split())]
    assemble.assemble_video([f], tmp_path / "vo.wav", tmp_path, "j", words=words)
    assert rendered and all(k == "kinetic_text" for k, _ in rendered) and not stills


# ── hook rules + validator ───────────────────────────────────────────

def test_hook_rules_in_style():
    r = hook_rules()
    assert r["first_word_by_s"] == 0.8 and r["max_words"] == 12
    assert "let me explain" in r["banned_phrases"] and "here is the thing" in r["filler_phrases"]


def test_draft5_hook_is_flagged():
    s = "Ten picometers—that is how AI reads your medical X-rays. Here is the thing. " + "word " * 120
    w = hook_warnings(s)
    assert any("fragment" in x for x in w)
    assert any('"that is how"' in x and '"here is the thing"' in x for x in w)


def test_plain_claim_hook_passes():
    s = "An AI can spot a broken wrist in 20 seconds. " + "word " * 120 + "That changes the job."
    assert hook_warnings(s) == []
    assert not INTRO_RE.search("in 20 seconds") and INTRO_RE.search("in sixty seconds")


def test_long_first_sentence_flagged():
    s = "Artificial intelligence systems are now able to look at a chest X-ray and find 5 things. " + "w " * 120
    assert any("sentence 1 has" in x for x in hook_warnings(s))


def test_filler_anywhere_and_curly_apostrophes():
    assert banned_phrases_in("Fine. Then. Here\u2019s the thing about it.", ["here's the thing"]) == ["here's the thing"]
    assert banned_phrases_in("the thingis", ["the thing is"]) == []
    assert first_sentence("One two. Three.") == "One two."


def test_enforce_hook_fixes_fragment(monkeypatch):
    body = " ".join(["word"] * 140)
    bad = {"script": "Ten picometers—that is how AI reads your X-rays. " + body}
    good = {"script": "An AI can read an X-ray 10 times faster than you blink. " + body}
    seen = {}
    monkeypatch.setattr(draft_mod, "_generate_json_response",
                        lambda prompt, provider: seen.setdefault("p", prompt) and good)
    out = draft_mod._enforce_hook(dict(bad), "p", None, research="10 times")
    assert out["script"] == good["script"]
    assert "no dash" in seen["p"] and "Here is the thing." in seen["p"]


def test_prompt_carries_plain_hook_rule(monkeypatch):
    from unittest.mock import patch
    with patch("verticals.draft.research_topic", return_value="r"), \
            patch("verticals.draft.call_llm", return_value=json.dumps({"script": "s", "broll_prompts": ["p"]})) as m:
        draft_mod.generate_draft("Test")
    prompt = m.call_args[0][0]
    assert "within 0.8 s" in prompt and "at most 12 words" in prompt and "fragment plus a dash" in prompt


def test_no_niche_hook_template_uses_banned_setup():
    import yaml
    rules = hook_rules()
    for p in Path(__file__).resolve().parent.parent.joinpath("niches").glob("*.yaml"):
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        for h in (data.get("script") or {}).get("hooks") or []:
            t = str(h.get("template", ""))
            assert not banned_phrases_in(t, rules["banned_phrases"] + rules["filler_phrases"]), (p.name, t)
            assert "that's how" not in t.lower()


# ── brand voice ──────────────────────────────────────────────────────

def test_style_voice_is_male_brand_voice():
    v = voice_settings()
    assert v["kokoro_voice"] == "am_michael" and float(v["kokoro_speed"]) == 1.08
    pace = load_style()["pace"]
    assert 150 <= pace["target_wpm"] <= 165


def test_style_voice_beats_workflow_default(monkeypatch):
    monkeypatch.setenv("KOKORO_VOICE", "af_heart")  # the workflow's hard-coded default
    monkeypatch.setenv("KOKORO_SPEED", "1.0")
    cfg = tts.kokoro_settings()
    assert cfg["voice"] == "am_michael" and cfg["speed"] == 1.08
    monkeypatch.setenv("DYPOL_VOICE", "bm_george")
    monkeypatch.setenv("DYPOL_VOICE_SPEED", "1.02")
    cfg = tts.kokoro_settings()
    assert cfg["voice"] == "bm_george" and cfg["speed"] == 1.02


def test_legacy_env_used_when_style_has_no_voice(monkeypatch):
    monkeypatch.setattr(tts, "_style_voice", lambda: {})
    monkeypatch.setenv("KOKORO_VOICE", "am_puck")
    assert tts.kokoro_settings()["voice"] == "am_puck"


def test_trim_lead_silence():
    import numpy as np
    sr = 24000
    audio = np.concatenate([np.zeros(sr), np.full(sr, 0.3, dtype="float32")]).astype("float32")
    out = tts.trim_lead_silence(audio, sr, 0.12)
    assert abs(len(out) / sr - 1.12) < 0.01
    silent = np.zeros(sr, dtype="float32")
    assert len(tts.trim_lead_silence(silent, sr)) == sr


# ── hook timing QC ───────────────────────────────────────────────────

def test_hook_timing(tmp_path):
    assert hook_timing([{"start": 0.3}])["ok"] is True
    assert hook_timing([])["first_word_s"] is None
    q = tmp_path / "qc.json"
    q.write_text(json.dumps({"status": "ok", "warnings": []}))
    t = add_hook_timing(q, [{"start": 1.1}], 0.8)
    data = json.loads(q.read_text())
    assert t["ok"] is False and data["status"] == "warn" and data["hook"]["first_word_s"] == 1.1
    assert any("first word" in w for w in data["warnings"])


def test_stage_timer_prints(capsys):
    from verticals.__main__ import _stage_timer
    lap = _stage_timer()
    lap("broll")
    assert "DYPOL_TIMING stage=broll seconds=" in capsys.readouterr().out


def test_narration_card_avoids_a_third_text_block(monkeypatch, tmp_path):
    rendered = []
    monkeypatch.setattr(assemble, "MEDIA_DIR", tmp_path)
    monkeypatch.setattr(assemble, "get_audio_duration", lambda p: 9.0)
    monkeypatch.setattr(assemble, "render_block",
                        lambda kind, data, out, d, **k: rendered.append(kind) or Path(out))
    monkeypatch.setattr(assemble, "_concat_copy", lambda parts, out: None)
    monkeypatch.setattr(assemble, "_recut", lambda *a, **k: None)
    monkeypatch.setattr(assemble, "prepare_audio", lambda *a, **k: tmp_path / "a.wav")
    monkeypatch.setattr(assemble, "run_cmd", lambda cmd, **k: None)
    monkeypatch.setattr(assemble, "shot_count", lambda sources, d: 3)
    monkeypatch.setattr(assemble, "plan_shots", lambda sources, d, durations=None: [
        {"source": 0, "duration": 3.0, "effect": "zoom_in", "variant": 0},
        {"source": None, "duration": 3.0, "effect": "zoom_in", "variant": 0},
        {"source": 1, "duration": 3.0, "effect": "zoom_in", "variant": 0}])
    monkeypatch.setenv("DYPOL_SFX_ENABLED", "0")
    monkeypatch.setenv("DYPOL_QC_ENABLED", "0")
    monkeypatch.setenv("DYPOL_LIVING_BG", "0")
    frames = []
    for i in range(2):
        f = tmp_path / f"broll_{i}.png"
        _png(f)
        frames.append(f)
    (tmp_path / "broll_manifest.json").write_text(json.dumps({"assets": [
        {"frame": f.name, "provider": "pixabay", "segment_index": i, "max_uses": 1, "clip": None}
        for i, f in enumerate(frames)]}))
    beats = [{"block": "kinetic_text", "data": {"words": ["fast", "scan"], "accent": "fast"}, "keywords": ["a", "b"]}
             for _ in range(2)]
    words = [{"word": w, "start": i * 0.5, "end": i * 0.5 + 0.4}
             for i, w in enumerate("Detectors catch radiation passing through tissue while models compare patterns".split())]
    assemble.assemble_video(frames, tmp_path / "vo.wav", tmp_path, "j", beats=beats, words=words)
    assert "diagram" in rendered
