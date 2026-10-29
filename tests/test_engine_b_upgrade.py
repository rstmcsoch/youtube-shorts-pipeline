"""Tests for the Engine B quality/speed upgrade: Kokoro TTS, captions, Pixabay
video tier, relevance gate, brand-free FLUX prompts, shot plan caps, audio mix."""

import json
import wave
from pathlib import Path

import pytest

import verticals.tts as tts
import verticals.visuals as visuals
import verticals.broll as broll
import verticals.music as music
from verticals.assemble import (
    plan_shots, build_sources, music_filter_graph, loudnorm_filter, _parse_loudnorm_json, shot_capacity,
)
from verticals.captions import group_caption_lines, _generate_ass, resolve_caption_font


# ───────────────────────── TTS ─────────────────────────

class TestKokoro:
    def test_speed_for_target_in_range_returns_none(self):
        assert tts.speed_for_target(60.0, 1.08) is None

    def test_speed_for_target_too_long_speeds_up(self):
        s = tts.speed_for_target(78.0, 1.08)
        assert s is not None and s > 1.08 and s <= 1.2

    def test_speed_for_target_too_short_slows_down(self):
        s = tts.speed_for_target(44.0, 1.08)
        assert s is not None and 0.95 <= s < 1.08

    def test_speed_for_target_refuses_big_changes(self):
        assert tts.speed_for_target(10.0, 1.08) is None  # script itself too short
        assert tts.speed_for_target(120.0, 1.08) is None

    def test_settings_defaults_and_clamp(self, monkeypatch):
        monkeypatch.delenv("KOKORO_VOICE", raising=False)
        monkeypatch.setenv("KOKORO_SPEED", "9")
        cfg = tts.kokoro_settings()
        assert cfg["voice"] == tts.KOKORO_DEFAULT_VOICE
        assert cfg["speed"] == 1.3
        monkeypatch.setenv("KOKORO_SPEED", "bad")
        assert tts.kokoro_settings()["speed"] == tts.KOKORO_DEFAULT_SPEED

    def test_write_wav_is_valid_pcm(self, tmp_path):
        import numpy as np
        out = tmp_path / "v.wav"
        tts._write_wav(np.zeros(24000, dtype="float32"), 24000, out)
        with wave.open(str(out)) as w:
            assert w.getframerate() == 24000 and w.getnchannels() == 1 and w.getnframes() == 24000

    def test_kokoro_failure_falls_back_to_edge(self, monkeypatch, tmp_path):
        def boom(*a, **k):
            raise RuntimeError("model missing")
        calls = {}

        def fake_edge(script, out_dir, lang, voice_override=""):
            calls["voice"] = voice_override
            return out_dir / "voiceover_en.mp3"
        monkeypatch.setattr(tts, "_generate_kokoro", boom)
        monkeypatch.setattr(tts, "_generate_edge_tts", fake_edge)
        p = tts.generate_voiceover("hello world", tmp_path, "en", provider="kokoro",
                                   voice_config={"voice_id": "en-US-GuyNeural"})
        assert p.name == "voiceover_en.mp3"
        assert calls["voice"] == "en-US-GuyNeural"

    def test_non_english_uses_edge(self, monkeypatch, tmp_path):
        used = {}
        monkeypatch.setattr(tts, "_generate_kokoro", lambda *a, **k: pytest.fail("kokoro used for hi"))
        monkeypatch.setattr(tts, "_generate_edge_tts",
                            lambda s, o, l, v="": used.setdefault("lang", l) and o / "x.mp3")
        tts.generate_voiceover("namaste", tmp_path, "hi", provider="kokoro")
        assert used["lang"] == "hi"

    def test_edge_ignores_non_edge_voice_ids(self, monkeypatch, tmp_path):
        seen = {}
        monkeypatch.setattr(tts, "_generate_edge_tts",
                            lambda s, o, l, v="": seen.setdefault("v", v) or o / "x.mp3")
        tts.generate_voiceover("hi", tmp_path, "en", provider="edge", voice_config={"voice_id": "af_heart"})
        assert seen["v"] == ""

    def test_checksum_mismatch_rejected(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KOKORO_MODEL_DIR", str(tmp_path))
        (tmp_path / "m.onnx").write_bytes(b"not a model")
        (tmp_path / tts.KOKORO_VOICES_FILE).write_bytes(b"x")
        monkeypatch.setitem(tts.KOKORO_SHA256, "m.onnx", "0" * 64)
        with pytest.raises(RuntimeError, match="checksum"):
            tts.ensure_kokoro_files("m.onnx")
        assert not (tmp_path / "m.onnx").exists()

    def test_implausible_duration_raises(self, monkeypatch, tmp_path):
        import numpy as np
        import sys
        import types

        class FakeKokoro:
            def __init__(self, *a):
                pass

            def create(self, text, voice, speed, lang):
                return np.zeros(24000, dtype="float32"), 24000  # 1 s for 60 words
        monkeypatch.setitem(sys.modules, "kokoro_onnx", types.SimpleNamespace(Kokoro=FakeKokoro))
        monkeypatch.setattr(tts, "ensure_kokoro_files", lambda m: (tmp_path / "a", tmp_path / "b"))
        with pytest.raises(RuntimeError, match="implausible"):
            tts._generate_kokoro(" ".join(["word"] * 60), tmp_path, "en")


# ───────────────────────── Captions ─────────────────────────

def _words(text, step=0.3):
    out, t = [], 0.0
    for w in text.split():
        out.append({"word": w, "start": round(t, 2), "end": round(t + step - 0.05, 2)})
        t += step
    return out


class TestCaptionLines:
    def test_lines_are_2_to_5_words(self):
        words = _words("This tiny chip just changed how phones handle AI. Here's the thing: it runs "
                       "a full language model offline with no cloud at all. That means faster answers.")
        lines = group_caption_lines(words, max_words=4)
        assert all(2 <= len(l) <= 5 for l in lines)
        assert sum(len(l) for l in lines) == len(words)

    def test_breaks_after_sentence_end(self):
        lines = group_caption_lines(_words("Big news today. Phones got faster now."), max_words=5)
        assert [w["word"] for w in lines[0]][-1] == "today."

    def test_max_words_clamped_to_five(self):
        lines = group_caption_lines(_words("a b c d e f g h i j k l"), max_words=9, max_chars=99)
        assert max(len(l) for l in lines) <= 5

    def test_ass_style(self, tmp_path):
        out = tmp_path / "c.ass"
        _generate_ass(_words("Defenders patch flaws faster than attackers now."), out,
                      highlight_color="#00FF88", font_family="Arial")
        text = out.read_text()
        style = [l for l in text.splitlines() if l.startswith("Style: Default")][0].split(",")
        assert style[7] == "-1"            # bold
        assert style[15] == "1"            # outline + shadow, not a box
        assert int(style[16]) >= 4 and int(style[17]) >= 2
        assert style[21] == str(int(1920 * 0.20))  # ~20% from bottom
        assert "\\fad(80," in text and ",80)}" in text
        assert "\\c&H88FF00&" in text      # active-word highlight kept
        assert style[1] in {"Montserrat", "Inter", "DejaVu Sans"}

    def test_font_chain(self):
        assert resolve_caption_font("Arial", {"montserrat", "dejavu sans"}) == "Montserrat"
        assert resolve_caption_font("Arial", {"inter", "dejavu sans"}) == "Inter"
        assert resolve_caption_font("", {"dejavu sans"}) == "DejaVu Sans"
        assert resolve_caption_font("Noto Sans CJK KR", {"noto sans cjk kr", "montserrat"}) == "Noto Sans CJK KR"
        assert resolve_caption_font("Arial", set()) == "DejaVu Sans"


# ───────────────────────── Relevance + FLUX ─────────────────────────

CYBER = {"type": "concept", "segment": "Defensive AI now patches flaws before attackers strike.",
         "entities": ["Google DeepMind"], "keywords": ["data visualization", "glowing nodes", "cybersecurity"]}


class TestRelevanceGate:
    def test_kitten_rejected_for_cyber_topic(self):
        kitten = {"provider": "pixabay", "title": "kitten, cat, pet, animal, cute", "width": 1080, "height": 1920}
        assert not visuals.stock_candidate_ok(kitten, CYBER)

    def test_requires_strong_tag_overlap(self):
        only_weak = {"title": "dark, blue, glowing, abstract", "width": 1080, "height": 1920}
        good = {"title": "cybersecurity, network, data", "width": 1080, "height": 1920}
        assert not visuals.stock_candidate_ok(only_weak, CYBER)
        assert visuals.stock_candidate_ok(good, CYBER)

    def test_off_topic_subject_even_with_overlap(self):
        hit = {"title": "cybersecurity, woman, portrait, laptop"}
        assert not visuals.stock_candidate_ok(hit, CYBER)

    def test_animals_allowed_when_segment_is_about_them(self):
        intent = {"segment": "Robot vacuums now map where your cat sleeps.",
                  "keywords": ["robot vacuum", "cat"], "entities": []}
        assert visuals.stock_candidate_ok({"title": "cat, robot, vacuum"}, intent)

    def test_select_visual_skips_kitten_and_uses_next_candidate(self, monkeypatch, tmp_path):
        kitten = {"provider": "pixabay", "title": "kitten cat glowing nodes data", "download_url": "k",
                  "width": 1080, "height": 1920, "asset_id": "1"}
        good = {"provider": "pixabay", "title": "cybersecurity data visualization network", "download_url": "g",
                "width": 1080, "height": 1920, "asset_id": "2"}
        monkeypatch.setattr(visuals, "_pixabay_candidates", lambda q: [kitten, good])
        got = []

        def fake_download(candidate, path):
            got.append(candidate["download_url"])
            return {**candidate, "sha256": candidate["asset_id"] * 8}
        monkeypatch.setattr(visuals, "_download_candidate", fake_download)
        meta, _ = visuals.select_visual(CYBER, tmp_path / "f.png", set())
        assert got == ["g"] and meta["asset_id"] == "2"

    def test_no_relevant_stock_falls_to_flux_with_clean_prompt(self, monkeypatch, tmp_path):
        kitten = {"provider": "pixabay", "title": "kitten cat", "download_url": "k", "width": 1080, "height": 1920}
        monkeypatch.setattr(visuals, "_pixabay_candidates", lambda q: [kitten])
        prompts = []

        def fake_flux(prompt, path):
            prompts.append(prompt)
            return {"provider": "cloudflare_workers_ai", "sha256": "abc"}
        monkeypatch.setattr(visuals, "_cloudflare_generate", fake_flux)
        meta, _ = visuals.select_visual(CYBER, tmp_path / "f.png", set())
        assert meta["provider"] == "cloudflare_workers_ai"
        assert "google" not in prompts[0].lower() and "deepmind" not in prompts[0].lower()


class TestFluxPrompt:
    def test_strips_entities_and_brands(self):
        p = visuals.flux_prompt({"entities": ["Google DeepMind", "Gemini 4 Carbon"],
                                 "keywords": ["Google server", "Nvidia GPU rack", "carbon hardware enclosure"]})
        low = p.lower()
        for banned in ("google", "deepmind", "gemini", "nvidia", "carbon"):
            assert banned not in low
        assert "no logos, no brand names" in low
        assert "server" in low and "gpu rack" in low

    def test_empty_after_strip_gets_generic_scene(self):
        p = visuals.flux_prompt({"entities": ["OpenAI"], "keywords": ["OpenAI"]})
        assert "openai" not in p.lower() and "generic" in p.lower()

    def test_variants_differ(self):
        a = visuals.flux_prompt(CYBER, 0)
        b = visuals.flux_prompt(CYBER, 1)
        assert a != b


# ───────────────────────── Pixabay video tier ─────────────────────────

VIDEO_HITS = {"hits": [
    {"id": 11, "pageURL": "https://pixabay.com/videos/id-11/", "tags": "kitten, cat", "duration": 12,
     "videos": {"medium": {"url": "https://cdn/k.mp4", "width": 1920, "height": 1080, "size": 1000}}},
    {"id": 12, "pageURL": "https://pixabay.com/videos/id-12/", "tags": "server, data center, network", "duration": 20,
     "videos": {"medium": {"url": "https://cdn/m.mp4", "width": 1920, "height": 1080, "size": 99 * 1024 * 1024},
                "small": {"url": "https://cdn/s.mp4", "width": 1280, "height": 720, "size": 5000}}},
    {"id": 13, "pageURL": "https://pixabay.com/videos/id-13/", "tags": "server room", "duration": 2,
     "videos": {"medium": {"url": "https://cdn/short.mp4", "width": 1920, "height": 1080, "size": 10}}},
]}
SERVER = {"type": "concept", "segment": "Data centers run the models.", "entities": ["Google DeepMind"],
          "keywords": ["server room", "Google data center", "dark"]}


class TestPixabayVideo:
    def test_short_brand_free_queries(self):
        qs = visuals.video_queries(SERVER)
        assert qs == ["server room", "data center"]
        assert all(len(q.split()) <= 3 for q in qs)

    def test_rendition_prefers_medium_but_respects_size(self):
        name, r = visuals._pick_rendition(VIDEO_HITS["hits"][1]["videos"])
        assert name == "small"

    def test_search_is_cached_24h(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PIXABAY_API_KEY", "k")
        monkeypatch.setenv("VERTICALS_HOME", str(tmp_path))
        calls = []

        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return VIDEO_HITS
        monkeypatch.setattr(visuals.requests, "get", lambda url, params=None, timeout=None: calls.append(url) or R())
        visuals._pixabay_video_search("server room")
        visuals._pixabay_video_search("Server Room")
        assert calls == ["https://pixabay.com/api/videos/"]

    def test_select_clip_skips_irrelevant_and_short(self, monkeypatch, tmp_path):
        monkeypatch.setattr(visuals, "_pixabay_video_search", lambda q: VIDEO_HITS)
        downloaded = []

        class Resp:
            def __init__(self, url):
                downloaded.append(url)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def raise_for_status(self):
                pass

            def iter_content(self, chunk_size=0):
                yield b"fake"
        monkeypatch.setattr(visuals.requests, "get", lambda url, stream=True, timeout=None: Resp(url))

        def fake_norm(raw, out, **k):
            Path(out).write_bytes(b"clip")
            return 4.0
        monkeypatch.setattr(visuals, "normalize_clip", fake_norm)
        meta = visuals.select_video_clip(SERVER, tmp_path / "clip_0.mp4", set())
        assert downloaded == ["https://cdn/s.mp4"]
        assert meta["provider"] == "pixabay_video" and meta["license"] == "Pixabay Content License"
        assert meta["source_url"] == "https://pixabay.com/videos/id-12/" and len(meta["sha256"]) == 64
        assert meta["duration_seconds"] <= 4.0 and meta["max_uses"] == 1

    def test_select_clip_never_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(visuals, "pixabay_video_candidates", lambda i: (_ for _ in ()).throw(RuntimeError()))
        assert visuals.select_video_clip(SERVER, tmp_path / "c.mp4") is None


# ───────────────────────── Shot plan ─────────────────────────

def _src(kind, provider, seg, path=None):
    return {"path": Path(path or f"{kind}_{seg}_{provider}"), "kind": kind, "provider": provider,
            "segment": seg, "max_uses": 1 if provider in broll.THIRD_PARTY_PROVIDERS else 2}


class TestShotPlan:
    def test_caps_and_variation(self):
        # The reviewed failure: few frames, one FLUX frame shown ~5 times.
        sources = [_src("still", "pixabay", 0), _src("still", "cloudflare_workers_ai", 1),
                   _src("still", "pixabay", 2), _src("still", "cloudflare_workers_ai", 3)]
        plan = plan_shots(sources, 56.0)
        counts = {}
        for i, s in enumerate(plan):
            if s["source"] is not None:
                counts[s["source"]] = counts.get(s["source"], 0) + 1
            if i and s["source"] is not None:
                assert s["source"] != plan[i - 1]["source"]
        assert all(c <= sources[k]["max_uses"] for k, c in counts.items())
        assert max(counts.values()) <= 2
        firsts = {}
        for s in plan:
            if s["source"] is None:
                continue
            if s["source"] in firsts:
                assert s["variant"] >= 1 and s["effect"] != firsts[s["source"]]
            else:
                firsts[s["source"]] = s["effect"]
                assert s["variant"] == 0

    def test_clip_first_and_once(self):
        sources = [_src("clip", "pixabay_video", 0), _src("still", "cloudflare_workers_ai", 0)]
        plan = plan_shots(sources, 16.0)
        assert plan[0]["source"] == 0
        assert sum(1 for s in plan if s["source"] == 0) == 1

    def test_gradient_only_when_exhausted(self):
        sources = [_src("still", "pixabay", i) for i in range(3)]
        plan = plan_shots(sources, 20.0)
        assert [s["source"] for s in plan].count(None) == len(plan) - 3

    def test_duration_is_exact(self):
        plan = plan_shots([_src("still", "gradient", 0)], 52.7)
        assert abs(sum(s["duration"] for s in plan) - 52.7) < 1e-6
        assert all(s["duration"] <= 4.0 for s in plan)

    def test_build_sources_reads_clip(self, tmp_path):
        (tmp_path / "clip_0.mp4").write_bytes(b"x")
        (tmp_path / "broll_manifest.json").write_text(json.dumps({"assets": [
            {"frame": "broll_0.png", "provider": "pixabay", "segment_index": 0,
             "clip": {"file": "clip_0.mp4", "provider": "pixabay_video", "duration_seconds": 4.0}},
            {"frame": "broll_extra_1.png", "provider": "cloudflare_workers_ai", "segment_index": 0, "max_uses": 2},
        ]}))
        srcs = build_sources([tmp_path / "broll_0.png"], tmp_path)
        assert [s["kind"] for s in srcs] == ["clip", "still", "still"]
        assert shot_capacity(srcs) == 4


class TestTopUp:
    def _manifest(self, tmp_path, n_ai=1, n_stock=3):
        rows = [{"frame": f"broll_{i}.png", "provider": "pixabay", "segment_index": i, "max_uses": 1,
                 "intent": {"keywords": ["server room"], "entities": ["Google"]}} for i in range(n_stock)]
        rows += [{"frame": f"broll_ai{i}.png", "provider": "cloudflare_workers_ai", "segment_index": i,
                  "max_uses": 2, "intent": {"keywords": ["circuit board"]}} for i in range(n_ai)]
        (tmp_path / "broll_manifest.json").write_text(json.dumps({"assets": rows}))

    def test_requests_more_flux_until_capacity(self, monkeypatch, tmp_path):
        self._manifest(tmp_path)  # capacity 3 + 2 = 5
        prompts = []

        def fake_flux(prompt, path):
            prompts.append(prompt)
            from PIL import Image
            Image.new("RGB", (100, 180)).save(path, format="PNG")
            return {"provider": "cloudflare_workers_ai", "source_url": "", "asset_id": "flux-1-schnell",
                    "license": "AI", "license_url": "", "retrieved_at": "", "sha256": ""}
        monkeypatch.setattr(visuals, "_cloudflare_generate", fake_flux)
        added = broll.top_up_ai_frames(tmp_path, 14)  # need 9 more → 5 frames
        assert added == 5
        assert all("google" not in p.lower() for p in prompts)
        data = json.loads((tmp_path / "broll_manifest.json").read_text())
        assert sum(1 for r in data["assets"] if r["provider"] == "cloudflare_workers_ai") == 6

    def test_never_more_than_10_flux_frames(self, monkeypatch, tmp_path):
        self._manifest(tmp_path, n_ai=9, n_stock=0)
        monkeypatch.setattr(visuals, "_cloudflare_generate", lambda p, path: (
            __import__("PIL.Image", fromlist=["Image"]).new("RGB", (10, 18)).save(path, format="PNG")
            or {"provider": "cloudflare_workers_ai", "sha256": ""}))
        assert broll.top_up_ai_frames(tmp_path, 40) == 1

    def test_stops_when_flux_unavailable(self, monkeypatch, tmp_path):
        self._manifest(tmp_path)
        monkeypatch.setattr(visuals, "_cloudflare_generate", lambda p, path: None)
        assert broll.top_up_ai_frames(tmp_path, 14) == 0


# ───────────────────────── Audio ─────────────────────────

class TestAudio:
    def test_loudnorm_targets(self):
        assert loudnorm_filter() == "loudnorm=I=-14:TP=-1.5:LRA=11"
        two = loudnorm_filter({"input_i": "-20", "input_tp": "-3", "input_lra": "5",
                               "input_thresh": "-30", "target_offset": "0.1"})
        assert "measured_I=-20" in two and "linear=true" in two

    def test_parse_loudnorm_json(self):
        err = 'junk\n{\n "input_i" : "-19.8",\n "input_tp" : "-2.1",\n "input_lra" : "4.0",\n' \
              ' "input_thresh" : "-30.0",\n "target_offset" : "0.2"\n}\n'
        assert _parse_loudnorm_json(err)["input_i"] == "-19.8"
        assert _parse_loudnorm_json('{"input_i" : "-inf"}') is None
        assert _parse_loudnorm_json("nothing") is None

    def test_music_graph_ducks_with_sidechain(self):
        g = music_filter_graph(60.0, -22)
        assert "sidechaincompress" in g and "volume=-22dB" in g and "[1:a]asplit=2" in g

    def test_music_disabled_by_default(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MUSIC_ENABLED", raising=False)
        assert music.select_and_prepare_music(tmp_path / "v.wav", tmp_path) == {}

    def test_only_verified_no_attribution_tracks(self, tmp_path):
        for f in ("ok.mp3", "attr.mp3", "unverified.mp3"):
            (tmp_path / f).write_bytes(b"x")
        (tmp_path / "tracks.json").write_text(json.dumps({"tracks": [
            {"file": "ok.mp3", "license": "Pixabay Content License", "verified": True, "attribution_required": False},
            {"file": "attr.mp3", "license": "CC BY 4.0", "verified": True, "attribution_required": True},
            {"file": "unverified.mp3", "license": "Pixabay Content License", "verified": False,
             "attribution_required": False},
            {"file": "missing.mp3", "license": "Pixabay Content License", "verified": True,
             "attribution_required": False},
        ]}))
        assert [Path(t["path"]).name for t in music.verified_tracks(tmp_path)] == ["ok.mp3"]

    def test_music_recorded_in_manifest(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MUSIC_ENABLED", "1")
        (tmp_path / "bed.mp3").write_bytes(b"x")
        (tmp_path / "tracks.json").write_text(json.dumps({"tracks": [
            {"file": "bed.mp3", "license": "Pixabay Content License", "verified": True,
             "attribution_required": False, "source_url": "https://pixabay.com/music/x"}]}))
        monkeypatch.setattr(music, "MUSIC_DIR", tmp_path)
        work = tmp_path / "w"
        work.mkdir()
        (work / "broll_manifest.json").write_text(json.dumps({"assets": []}))
        r = music.select_and_prepare_music(work / "v.wav", work, words=[{"word": "a", "start": 0, "end": 1}])
        assert -24 <= r["gain_db"] <= -20
        rec = json.loads((work / "broll_manifest.json").read_text())["audio"]["music"]
        assert rec["license"] == "Pixabay Content License" and rec["attribution_required"] is False
