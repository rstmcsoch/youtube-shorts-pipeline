"""Engine B: Kokoro narration pace is tuned into the 150-165 WPM band (draft job-mw0qn7ms,
3 Nov 2026, came out at 116 WPM at the fixed 1.08 speed). No network, no real model."""
import sys
import types

import numpy as np
import pytest

from verticals import tts

SR = 24000


def test_pace_band_reads_style_pace_block():
    assert tts.pace_band() == (150.0, 165.0, 158.0)


def test_pace_speed_in_band_is_none():
    assert tts.pace_speed(132, 132 / 158 * 60, 1.08) is None


def test_pace_speed_slow_render_speeds_up():
    s = tts.pace_speed(132, 70.5, 1.08)          # 112 WPM -> needs ~1.52, clamped
    assert s == tts.KOKORO_MAX_SPEED
    s2 = tts.pace_speed(132, 60.0, 1.08)         # 132 WPM -> 1.293
    assert 1.25 < s2 < 1.33


def test_pace_speed_fast_render_slows_down():
    s = tts.pace_speed(150, 50.0, 1.2)           # 180 WPM
    assert 1.0 < s < 1.1


def test_pace_speed_empty_inputs():
    assert tts.pace_speed(0, 10, 1.08) is None and tts.pace_speed(10, 0, 1.08) is None


def _fake_kokoro(monkeypatch, wpm_at_108: float, calls: list):
    class FakeKokoro:
        def __init__(self, *a, **k):
            pass

        def create(self, script, voice, speed, lang):
            calls.append(speed)
            words = len(script.split())
            seconds = words / (wpm_at_108 * speed / 1.08) * 60.0
            return np.full(int(seconds * SR), 0.2, dtype=np.float32), SR

    monkeypatch.setitem(sys.modules, "kokoro_onnx", types.SimpleNamespace(Kokoro=FakeKokoro))
    monkeypatch.setattr(tts, "ensure_kokoro_files", lambda *a, **k: ("m", "v"))
    monkeypatch.setattr(tts, "_write_wav", lambda *a, **k: None)
    for k in ("DYPOL_VOICE_SPEED", "KOKORO_SPEED", "DYPOL_VOICE"):
        monkeypatch.delenv(k, raising=False)


def test_slow_dense_script_is_rerendered_into_band(monkeypatch, tmp_path):
    calls = []
    _fake_kokoro(monkeypatch, 116.0, calls)
    script = " ".join(["word"] * 132)
    tts._generate_kokoro(script, tmp_path, "en")
    assert calls[0] == 1.08 and len(calls) == 2
    final_wpm = 116.0 * calls[-1] / 1.08
    assert final_wpm >= 145  # clamped at 1.4 -> ~150 WPM
    meta = (tmp_path / "voiceover_meta.json").read_text()
    assert str(calls[-1]) in meta


def test_in_band_render_is_not_rerendered(monkeypatch, tmp_path):
    calls = []
    _fake_kokoro(monkeypatch, 157.0, calls)
    tts._generate_kokoro(" ".join(["word"] * 140), tmp_path, "en")
    assert calls == [1.08]


def test_short_script_is_not_slowed_below_band(monkeypatch, tmp_path):
    calls = []
    _fake_kokoro(monkeypatch, 158.0, calls)
    tts._generate_kokoro(" ".join(["word"] * 100), tmp_path, "en")   # 38 s at 158 WPM
    assert calls == [1.08]
