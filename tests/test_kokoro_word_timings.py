"""Exact audio-visual sync: word timings from the timestamped Kokoro model's phoneme
durations (aligned by pronunciation), the Whisper fallback, and motion cues on word starts."""

import json
import sys
import types

import numpy as np
import pytest

import verticals.captions as captions
import verticals.tts as tts
from verticals.motion import cue_starts, motion_cards, spec_data, word_cues

PHON = {
    "In": "ɪn",
    "2024,": "tˈuː θˈaʊzənd twˈɛnti fˈɔːɹ,",
    "edge": "ˈɛdʒ",
    "cars": "kˈɑːɹz",
    "won.": "wˈʌn.",
}
SCRIPT = "In 2024, edge cars won."


def _stream(phonemes: str, step: float = 0.05, t0: float = 0.1):
    """(phoneme, start, end) per character, as kokoro-onnx reports them."""
    return [(c, round(t0 + i * step, 3), round(t0 + (i + 1) * step, 3)) for i, c in enumerate(phonemes)]


def _start_of(stream, index):
    return stream[index][1]


def test_number_maps_to_all_its_spoken_words():
    spoken = " ".join(PHON[w] for w in SCRIPT.split())
    stream = _stream(spoken)
    words = tts.map_phoneme_timings(SCRIPT, stream, PHON.__getitem__)
    assert [w["word"] for w in words] == SCRIPT.split()
    two = spoken.index("t")                       # "two thousand twenty four" all belongs to 2024
    four_end = spoken.index("ɹ,") + 1
    assert words[1]["start"] == pytest.approx(_start_of(stream, two))
    assert words[1]["end"] == pytest.approx(stream[four_end - 1][2])
    # a word that opens on a stress mark starts at the mark, not at the vowel after it
    assert words[2]["start"] == pytest.approx(_start_of(stream, spoken.index("ˈɛdʒ")))
    assert words[3]["start"] == pytest.approx(_start_of(stream, spoken.index("kˈɑ")))
    assert all(a["start"] <= b["start"] for a, b in zip(words, words[1:]))


def test_counting_spaces_would_be_wrong():
    """Spoken words != script words: 5 script words, 8 phoneme words."""
    spoken = " ".join(PHON[w] for w in SCRIPT.split())
    assert len(spoken.split()) == 8 and len(SCRIPT.split()) == 5
    words = tts.map_phoneme_timings(SCRIPT, _stream(spoken), PHON.__getitem__)
    assert words[-1]["word"] == "won." and words[-1]["start"] > words[1]["end"]


def test_unrelated_phonemes_fail_alignment():
    stream = _stream("zzzz qqqq xxxx")
    assert tts.map_phoneme_timings(SCRIPT, stream, PHON.__getitem__) is None
    assert tts.map_phoneme_timings("", stream, PHON.__getitem__) is None


def test_one_missed_word_is_spread_between_neighbours():
    phon = dict(PHON, cars="")    # phonemizer gave nothing for one word
    spoken = " ".join(PHON[w] for w in SCRIPT.split())
    words = tts.map_phoneme_timings(SCRIPT, _stream(spoken), phon.__getitem__)
    assert words is not None
    assert words[2]["end"] <= words[3]["start"] + 1e-6 <= words[4]["start"]


def _fake_engine(monkeypatch, tmp_path):
    class FakeKokoro:
        def __init__(self, *a, **k):
            pass

        def create(self, script, voice, speed, lang):
            return np.full(24000 * 6, 0.2, dtype=np.float32), 24000

    monkeypatch.setitem(sys.modules, "kokoro_onnx", types.SimpleNamespace(Kokoro=FakeKokoro))
    monkeypatch.setattr(tts, "ensure_kokoro_files", lambda *a, **k: (tmp_path / "m", tmp_path / "v"))
    monkeypatch.setattr(tts, "pace_speed", lambda *a, **k: None)
    monkeypatch.setattr(tts, "speed_for_target", lambda *a, **k: None)
    monkeypatch.setattr(tts, "short_padding_seconds", lambda *a, **k: 0.0)
    for k in ("DYPOL_VOICE_SPEED", "KOKORO_SPEED", "DYPOL_VOICE"):
        monkeypatch.delenv(k, raising=False)


SCRIPT12 = "In 2024, edge cars won. In 2024, edge cars won. In 2024, edge cars won."


def test_missing_timed_model_falls_back_to_whisper(monkeypatch, tmp_path):
    _fake_engine(monkeypatch, tmp_path)
    monkeypatch.setenv("KOKORO_TIMED_MODEL", str(tmp_path / "absent.onnx"))
    monkeypatch.delenv("KOKORO_TIMED_AUTO_DOWNLOAD", raising=False)
    lines = []
    monkeypatch.setattr(tts, "log", lines.append)
    out = tts._generate_kokoro(SCRIPT12, tmp_path, "en")
    assert out.exists() and not (tmp_path / tts.WORD_TIMINGS_FILE).exists()
    assert any("fall back to Whisper" in l for l in lines)
    whisper = []
    monkeypatch.setattr(captions, "_whisper_word_timestamps",
                        lambda *a, **k: whisper.append(1) or [])
    monkeypatch.setattr(captions, "log", lambda *a: None)
    from verticals import config
    monkeypatch.setattr(config, "run_cmd", lambda *a, **k: None)
    captions.generate_captions(out, tmp_path, "en", script=SCRIPT12)
    assert whisper == [1]


def test_failed_alignment_falls_back_to_whisper(monkeypatch, tmp_path):
    _fake_engine(monkeypatch, tmp_path)
    engine = types.SimpleNamespace(tokenizer=types.SimpleNamespace(phonemize=lambda w, lang: "ɑ"))
    monkeypatch.setattr(tts, "_timed_engine", lambda *a: engine)
    monkeypatch.setattr(tts, "kokoro_voice_arg", lambda e, v: v)
    monkeypatch.setattr(tts, "kokoro_create_timed",
                        lambda *a: (np.full(24000 * 6, 0.2, dtype=np.float32), 24000, _stream("zzzz qqqq")))
    lines = []
    monkeypatch.setattr(tts, "log", lines.append)
    tts._generate_kokoro(SCRIPT12, tmp_path, "en")
    assert not (tmp_path / tts.WORD_TIMINGS_FILE).exists()
    assert any("alignment failed" in l and "Whisper" in l for l in lines)


def test_timed_words_feed_captions_without_whisper(monkeypatch, tmp_path):
    _fake_engine(monkeypatch, tmp_path)
    phon = {w: PHON[w] for w in PHON}
    engine = types.SimpleNamespace(tokenizer=types.SimpleNamespace(phonemize=lambda w, lang: phon[w]))
    spoken = " ".join(PHON[w] for w in SCRIPT12.split())
    monkeypatch.setattr(tts, "_timed_engine", lambda *a: engine)
    monkeypatch.setattr(tts, "kokoro_voice_arg", lambda e, v: v)
    monkeypatch.setattr(tts, "kokoro_create_timed",
                        lambda *a: (np.full(24000 * 6, 0.2, dtype=np.float32), 24000, _stream(spoken, 0.03)))
    out = tts._generate_kokoro(SCRIPT12, tmp_path, "en")
    data = json.loads((tmp_path / tts.WORD_TIMINGS_FILE).read_text())
    assert data["audio"] == out.name and len(data["words"]) == 15
    monkeypatch.setattr(captions, "_whisper_word_timestamps", lambda *a, **k: pytest.fail("Whisper ran"))
    seen = {}
    monkeypatch.setattr(captions, "generate_dypol_ass",
                        lambda words, path, **k: seen.update(words=words, keywords=k.get("keywords")) or path)
    res = captions.generate_captions(out, tmp_path, "en", script=SCRIPT12, keywords=["edge"])
    assert [w["start"] for w in res["words"]] == [w["start"] for w in data["words"]]
    assert seen["keywords"] == ["edge"] and seen["words"] == res["words"]


def test_stale_timings_for_other_audio_are_ignored(tmp_path):
    tts.write_word_timings(tmp_path, "voiceover_en.wav", [{"word": "a", "start": 0, "end": 1}], "x")
    assert tts.load_word_timings(tmp_path / "voiceover_en.mp3") == []
    assert len(tts.load_word_timings(tmp_path / "voiceover_en.wav")) == 1


def test_generate_voiceover_clears_old_timings(monkeypatch, tmp_path):
    tts.write_word_timings(tmp_path, "voiceover_en.wav", [{"word": "a", "start": 0, "end": 1}], "x")
    monkeypatch.setattr(tts, "_generate_edge_tts", lambda s, o, l, v="": o / "voiceover_en.mp3")
    tts.generate_voiceover("hi", tmp_path, "en", provider="edge")
    assert not (tmp_path / tts.WORD_TIMINGS_FILE).exists()


def test_timed_model_sha_mismatch_is_rejected(monkeypatch, tmp_path):
    bad = tmp_path / "ts.onnx"
    bad.write_bytes(b"not the model")
    monkeypatch.setenv("KOKORO_TIMED_MODEL", str(bad))
    monkeypatch.setattr(tts, "log", lambda *a: None)
    assert tts.ensure_kokoro_timed_model() is None


# ── motion cues land on word starts ──
WORDS = [{"word": w, "start": s, "end": s + 0.3} for w, s in
         [("Banks", 10.0), ("score", 10.4), ("every", 10.9), ("swipe", 11.3), ("with", 12.0), ("risk", 12.6),
          ("models.", 13.1), ("It", 14.0), ("costs", 14.3), ("844", 14.8), ("million.", 15.4)]]


def test_kinetic_words_appear_on_their_word_start():
    specs = motion_cards("kinetic_text", {"words": ["swipe", "risk"], "accent": "risk"}, [3.0], 10.0, WORDS)
    data = spec_data(specs[0])
    assert data["cues"] == [pytest.approx(1.3), pytest.approx(2.6)]
    assert cue_starts(data, [0.1, 0.36]) == [pytest.approx(1.3), pytest.approx(2.6)]


def test_diagram_nodes_cue_on_word_starts_across_sub_shots():
    specs = motion_cards("diagram", {"nodes": ["Banks", "Score", "Risk models"], "title": ""}, [2.0, 2.0],
                         10.0, WORDS)
    assert all(spec_data(s)["cues"] == [0.0, pytest.approx(0.4), pytest.approx(2.6)] for s in specs)


def test_count_up_starts_counting_when_the_number_is_spoken():
    specs = motion_cards("count_up", {"value": 844, "label": "losses"}, [3.0], 13.5, WORDS)
    assert spec_data(specs[0])["cues"] == [pytest.approx(1.3)]


def test_unspoken_elements_keep_the_default_stagger_and_late_cues_are_capped():
    assert word_cues(["latency", "swipe"], WORDS, 10.0, 13.0) == [None, pytest.approx(1.3)]
    assert cue_starts({"cues": [None, 1.3]}, [0.1, 0.36]) == [0.1, 1.3]
    # a word 6 s in never holds the picture past the static limit
    assert cue_starts({"cues": [6.0]}, [0.1]) == [pytest.approx(2.8)]


def test_script_keywords_reach_captions():
    from verticals.__main__ import script_keywords
    draft = {"beats": [{"keywords": ["risk score", "Visa"]}, {"keywords": ["visa"]}],
             "visual_intents": [{"entities": ["Mastercard"]}], "youtube_tags": "fraud, AI"}
    assert script_keywords(draft) == ["risk score", "Visa", "Mastercard", "fraud", "AI"]
