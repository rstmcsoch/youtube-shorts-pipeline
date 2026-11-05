"""Console contract (5 Nov 2026): DYPOL_VOICE validation/blends and design-pattern fallback."""

import json

import numpy as np

from verticals import tts
from verticals.design_patterns import resolve_design_pattern


def test_plain_kokoro_id_is_valid():
    assert tts.validate_kokoro_voice("am_michael") == "am_michael"
    assert tts.validate_kokoro_voice(" BM_George ") == "bm_george"


def test_blend_is_parsed_and_normalised():
    assert tts.parse_kokoro_voice("am_michael:0.6,am_adam:0.4") == [("am_michael", 0.6), ("am_adam", 0.4)]
    assert tts.parse_kokoro_voice("am_michael:3,am_adam:1") == [("am_michael", 0.75), ("am_adam", 0.25)]
    assert tts.validate_kokoro_voice("am_michael:0.6, am_adam:0.4") == "am_michael:0.6,am_adam:0.4"


def test_invalid_voices_fall_back_to_am_michael(monkeypatch):
    lines = []
    monkeypatch.setattr(tts, "log", lines.append)
    for bad in ("not_a_voice", "am_michael:0.6,xx_nope:0.4", "am_michael:-1", "am_michael:abc",
                "am_michael:0.5,am_michael:0.5", "am_michael,am_adam", "a:1,b:1,c:1,d:1,e:1"):
        assert tts.validate_kokoro_voice(bad) == "am_michael", bad
    assert len(lines) == 7 and all("falling back to am_michael" in l for l in lines)


def test_kokoro_settings_validates_dypol_voice(monkeypatch):
    monkeypatch.setattr(tts, "log", lambda *_: None)
    monkeypatch.setenv("DYPOL_VOICE", "af_unknown")
    assert tts.kokoro_settings()["voice"] == "am_michael"
    monkeypatch.setenv("DYPOL_VOICE", "am_michael:0.6,am_adam:0.4")
    assert tts.kokoro_settings()["voice"] == "am_michael:0.6,am_adam:0.4"


def test_blend_voice_arg_is_weighted_style():
    class FakeEngine:
        styles = {"am_michael": np.ones((2, 3)), "am_adam": np.zeros((2, 3))}

        def get_voice_style(self, name):
            return self.styles[name]

    assert tts.kokoro_voice_arg(FakeEngine(), "am_michael") == "am_michael"
    arr = tts.kokoro_voice_arg(FakeEngine(), "am_michael:0.6,am_adam:0.4")
    assert np.allclose(arr, 0.6)


def test_design_pattern_fallback(tmp_path, monkeypatch):
    from verticals import design_patterns
    monkeypatch.setattr(design_patterns, "log", lambda *_: None)
    idx = json.loads(open(design_patterns.INDEX_PATH, encoding="utf-8").read())
    assert {p["id"]: p["status"] for p in idx} == {"elements": "ready", "characters": "coming_soon"}
    assert resolve_design_pattern("elements") == "elements"
    assert resolve_design_pattern("characters") == "elements"
    assert resolve_design_pattern("nope") == "elements"
    monkeypatch.delenv("DYPOL_PATTERN", raising=False)
    assert resolve_design_pattern() == "elements"
