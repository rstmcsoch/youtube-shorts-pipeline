"""Calmer captions (lower, key-word highlight only) and soft procedural SFX well under the voice."""

import re
import subprocess
import wave
from pathlib import Path

import pytest

from verticals import assemble, sfx
from verticals.captions import (
    DYPOL_CAPTION, SHORTS_BOTTOM_UI_FRAC, _sentences, caption_block_bottom, generate_dypol_ass, select_key_words,
)

REPO = Path(__file__).resolve().parent.parent
LIME = "\\c&H60F5C8&"
SCRIPT = ("In 2024, handwriting recognition hit 98 percent accuracy on messy cursive. "
          "Neural networks learn letter shapes from millions of examples. "
          "That is the new normal. Banks now read 3.5 million cheques a day with it!")


def _words(text, step=0.35):
    out, t = [], 0.0
    for w in text.split():
        out.append({"word": w, "start": t, "end": t + step * 0.8})
        t += step
    return out


# ───────────── caption position ─────────────

def test_caption_y_is_about_three_quarters_down():
    assert 0.73 <= DYPOL_CAPTION["center_y_frac"] <= 0.77


def test_caption_block_above_shorts_bottom_ui():
    safe = 1920 * (1 - SHORTS_BOTTOM_UI_FRAC)
    assert caption_block_bottom(1920) <= safe
    assert caption_block_bottom(1920) > 1920 * 0.70        # actually moved down from 0.63


def test_ass_uses_new_y(tmp_path):
    text = generate_dypol_ass(_words(SCRIPT), tmp_path / "c.ass").read_text()
    ys = {int(y) for y in re.findall(r"\\pos\(\d+,(\d+)\)", text)}
    assert ys == {round(1920 * DYPOL_CAPTION["center_y_frac"])}


# ───────────── key-word highlight ─────────────

def _is_num(w):
    return bool(re.search(r"\d", w)) or w.lower().strip(".,!?") in {"percent", "million", "billion", "thousand"}


def test_highlight_numbers_plus_at_most_one_term_per_sentence():
    words = _words(SCRIPT)
    picked = select_key_words(words)
    for sent in _sentences(words):
        terms = [i for i in sent if i in picked and not _is_num(words[i]["word"])]
        assert len(terms) <= 1
        for i in sent:
            if re.search(r"\d", words[i]["word"]):
                assert i in picked                       # every number / year / percentage


def test_percent_unit_follows_its_number():
    words = _words(SCRIPT)
    picked = select_key_words(words)
    i = [w["word"] for w in words].index("98")
    assert {i, i + 1} <= picked


def test_stopwords_never_key_terms():
    words = _words("This is what they would have done there. And then they were here again.")
    picked = select_key_words(words)
    assert all(words[i]["word"].lower().strip(".") not in {"this", "what", "they", "would", "have", "there", "were", "here", "again", "then"}
               for i in picked)


def test_script_keywords_take_priority():
    words = _words("Neural networks learn letter shapes from millions of examples.")
    picked = select_key_words(words, keywords=["shapes"])
    assert [words[i]["word"] for i in picked] == ["shapes"]


def test_ass_lime_count_matches_selection(tmp_path):
    words = _words(SCRIPT)
    text = generate_dypol_ass(words, tmp_path / "c.ass").read_text()
    assert text.count(LIME) == len(select_key_words(words))
    assert text.count(LIME) < len(words) / 2


# ───────────── SFX ─────────────

def test_sfx_offset_at_least_18_db():
    assert sfx.UNDER_VOICE_DB >= 18.0
    assert all(g <= 0 for g in sfx.TRANSITION_GAIN.values())


def test_rendered_sfx_peak_18db_under_voice_even_when_overlapping(tmp_path):
    np = pytest.importorskip("numpy")
    voice = tmp_path / "v.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=f=220:d=4:sample_rate=48000",
                    "-af", "volume=-3dB", str(voice)], check=True)
    ev = [{"t": 1.0, "at": 1.0, "clip": c, "kind": "transition", "gain_db": 0.0}
          for c in ("whoosh_01", "chime_01", "swell_01")]
    out = sfx.render_sfx_track(ev, 4.0, voice, tmp_path / "s.wav")
    with wave.open(str(out)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2") / 32767
    v = sfx._decode(voice)
    assert 20 * np.log10(np.abs(v).max()) - 20 * np.log10(np.abs(x).max()) >= 18.0


def test_no_kenney_pop_impact_click_files_referenced():
    src = (REPO / "verticals" / "sfx.py").read_text().lower()
    assert "kenney" not in src and ".ogg" not in src
    for c in sfx.load_manifest()["clips"]:
        assert c["category"] not in {"pop", "impact", "click", "squish"}
        assert not re.match(r"(pop|impact|click|squish)", c["id"])
        assert "file" not in c                            # synthesised, nothing vendored
    assert not list((REPO / "sfx").glob("*.ogg")) if (REPO / "sfx").exists() else True


def test_sfx_and_music_ducked_under_speech():
    g = assemble.music_filter_graph(45.0, -26, sfx=True)
    assert "[bed][sc]sidechaincompress" in g and "[3:a][sc2]sidechaincompress" in g
    assert "sidechaincompress" in assemble.sfx_only_filter_graph()


def test_loudnorm_true_peak_ceiling():
    assert assemble.LOUDNORM["TP"] <= -1.0
