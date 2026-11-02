"""Engine B piece 2: pacing, DYPOL karaoke captions, SFX, grade, QC."""

import json
import random
import re
import subprocess
import wave
from pathlib import Path

import pytest

from verticals import assemble, pacing, qc, sfx
from verticals.captions import (
    DYPOL_CAPTION, dypol_caption_chunks, generate_dypol_ass, use_dypol_captions,
)
from verticals.motion import code_typing_window

REPO = Path(__file__).resolve().parent.parent


def _words(text, start=0.1, step=0.36, pause_after_sentence=0.3):
    out, t = [], start
    for w in text.split():
        out.append({"word": w, "start": round(t, 3), "end": round(t + step * 0.85, 3)})
        t += step + (pause_after_sentence if w[-1] in ".!?" else 0.0)
    return out


SCRIPT = ("This model just beat every coding test, by forty percent. Here's why it matters. "
          "Run one command, and the agent ships the fix before you finish your coffee. "
          "Benchmarks moved twelve points in a month: that is the new normal for every lab.")


# ───────────────────────── pacing ─────────────────────────

class TestPacing:
    def test_limits_from_style_guide(self):
        assert pacing.pace_limits() == (2.0, 3.5)

    @pytest.mark.parametrize("duration", [12.0, 45.0, 58.3, 75.0])
    def test_every_segment_in_window_without_words(self, duration):
        segs = pacing.segments(pacing.cut_points(duration))
        assert abs(sum(segs) - duration) < 1e-6
        assert max(segs) <= 3.5 + 1e-6
        assert min(segs) >= 1.6

    def test_segments_with_words_respect_cap_and_prefer_clauses(self):
        words = _words(SCRIPT * 3)
        duration = words[-1]["end"] + 0.3
        cuts = pacing.cut_points(duration, words)
        segs = pacing.segments(cuts)
        assert max(segs) <= 3.5 + 1e-6 and min(segs) >= 1.6
        clause_times = [t for t, c in pacing._boundaries(words) if c]
        on_clause = sum(1 for c in cuts[1:-1] if any(abs(c - t) < 0.04 for t in clause_times))
        assert on_clause >= len(cuts[1:-1]) // 3

    def test_cuts_are_frame_quantised(self):
        cuts = pacing.cut_points(40.0, _words(SCRIPT * 2), fps=30)
        for c in cuts[1:-1]:
            assert abs(c * 30 - round(c * 30)) < 1e-6

    def test_short_video_is_one_segment(self):
        assert pacing.cut_points(3.0) == [0.0, 3.0]

    def test_group_cuts_keeps_order_and_count(self):
        segs = [2.5] * 20
        groups = pacing.group_cuts(segs, 13)
        assert len(groups) == 13
        assert [x for g in groups for x in g] == segs
        assert all(1 <= len(g) <= 2 for g in groups)

    @pytest.mark.parametrize("duration", [45.0, 60.0, 75.0])
    def test_shot_groups_same_shot_count_as_before(self, duration):
        groups = pacing.shot_groups(duration, None, 4.0)
        assert len(groups) == -(-int(duration * 1000) // 4000)
        assert pacing.max_static(groups) <= 3.5 + 1e-6
        assert abs(sum(map(sum, groups)) - duration) < 1e-6

    def test_plan_shots_uses_given_durations(self):
        sources = [{"path": Path(f"s{i}.png"), "kind": "still", "provider": "flux", "segment": i,
                    "max_uses": 2, "max_len": None} for i in range(6)]
        durs = [5.0, 2.8, 6.1, 3.0]
        plan = assemble.plan_shots(sources, sum(durs), durations=durs)
        assert [p["duration"] for p in plan] == durs

    def test_recut_filter_moves_alternate_subshots(self):
        graph, frames = assemble.recut_filter([2.4, 2.7, 2.0], assemble.PULL_BACK)
        assert frames == round(7.1 * 30)
        assert graph.count("trim=start_frame") == 3
        parts = graph.split(";")
        assert assemble.PULL_BACK not in parts[1] and assemble.PULL_BACK in parts[2]
        assert assemble.PULL_BACK not in parts[3]
        assert "concat=n=3:v=1:a=0" in graph

    def test_pull_back_keeps_motion_text_in_safe_band(self):
        # content box x 72-930, y 170-1080 scaled 0.92 about the frame centre
        s = 994 / 1080
        for x in (72, 930):
            assert 72 <= 540 + (x - 540) * s <= 1080 * 0.86
        assert 960 + (1080 - 960) * s < 1920 * 0.63

    def test_stills_split_into_subshots_with_new_moves(self, monkeypatch, tmp_path):
        calls, concats = [], []
        monkeypatch.setattr(assemble, "MEDIA_DIR", tmp_path)
        monkeypatch.setattr(assemble, "get_audio_duration", lambda p: 16.0)
        monkeypatch.setattr(assemble, "animate_frame",
                            lambda img, out, d, effect, variant=0: calls.append((Path(out).name, round(d, 3), effect, variant)))
        monkeypatch.setattr(assemble, "prepare_audio", lambda *a, **k: tmp_path / "a.wav")
        monkeypatch.setattr(assemble, "run_cmd", lambda cmd, **k: concats.append(cmd))
        monkeypatch.setenv("DYPOL_SFX_ENABLED", "0")
        monkeypatch.setenv("DYPOL_QC_ENABLED", "0")
        frames = []
        for i in range(4):
            f = tmp_path / f"broll_{i}.png"
            f.write_bytes(b"png")
            frames.append(f)
        assemble.assemble_video(frames, tmp_path / "vo.wav", tmp_path, "j", words=_words(SCRIPT)[:40])
        assert max(d for _, d, _, _ in calls) <= 3.5 + 1e-6
        subs = [c for c in calls if re.match(r"anim_\d+_\d+\.mp4", c[0])]
        assert subs, "long shots must be split"
        by_shot = {}
        for name, _, effect, variant in subs:
            by_shot.setdefault(name.split("_")[1], []).append((effect, variant))
        for moves in by_shot.values():
            assert len(set(moves)) == len(moves)


# ───────────────────────── captions ─────────────────────────

class TestDypolCaptions:
    def test_chunks_are_2_to_4_words_and_never_cross_sentences(self):
        chunks = dypol_caption_chunks(_words(SCRIPT))
        assert all(1 <= len(c) <= 4 for c in chunks)
        assert sum(1 for c in chunks if len(c) == 1) <= 1
        for c in chunks:
            for w in c[:-1]:
                assert w["word"][-1] not in ".!?"

    def test_lone_tail_word_borrows(self):
        chunks = dypol_caption_chunks(_words("That is the new normal."))
        assert [len(c) for c in chunks] == [3, 2]

    def test_ass_style_and_karaoke(self, tmp_path):
        out = generate_dypol_ass(_words(SCRIPT), tmp_path / "c.ass")
        text = out.read_text()
        style = [l for l in text.splitlines() if l.startswith("Style: Dypol")][0].split(",")
        assert style[1] == "Space Grotesk" and style[7] == "-1"
        assert style[15] == "1" and int(style[16]) >= 5 and int(style[17]) >= 3   # stroke + shadow
        assert "\\c&H60F5C8&" in text                                             # lime active word
        y = round(1920 * 0.63)
        assert f"\\an5\\pos({DYPOL_CAPTION['center_x']},{y})" in text
        n_words = len(_words(SCRIPT))
        assert text.count("Dialogue:") == n_words
        assert text.count("\\c&H60F5C8&") == n_words

    def test_position_clear_of_shorts_ui(self):
        cx, w = DYPOL_CAPTION["center_x"], DYPOL_CAPTION["max_width"]
        assert cx - w / 2 >= 72 and cx + w / 2 <= 1080 * 0.86
        line_h = DYPOL_CAPTION["size"]
        assert 1920 * DYPOL_CAPTION["center_y_frac"] + line_h <= 1920 * 0.78

    def test_long_chunk_is_set_smaller(self, tmp_path):
        words = _words("Internationalization standardization interoperability.")
        text = generate_dypol_ass(words, tmp_path / "c.ass").read_text()
        sizes = [int(s) for s in re.findall(r"\\fs(\d+)", text)]
        assert sizes and max(sizes) < DYPOL_CAPTION["size"]

    def test_language_switch(self, monkeypatch):
        assert use_dypol_captions("en") and use_dypol_captions("es")
        assert not use_dypol_captions("hi") and not use_dypol_captions("ja")
        monkeypatch.setenv("DYPOL_CAPTION_STYLE", "legacy")
        assert not use_dypol_captions("en")

    def test_ass_filter_points_at_vendored_fonts(self):
        f = assemble._ass_filter("/tmp/c.ass")
        assert f.startswith("ass=/tmp/c.ass:fontsdir=") and "style/fonts" in f
        assert (REPO / "style" / "fonts" / "SpaceGrotesk-Bold.ttf").exists()


# ───────────────────────── SFX ─────────────────────────

class TestSfx:
    def test_manifest_and_licences(self):
        m = sfx.load_manifest()
        clips = m["clips"]
        assert 10 <= len(clips) <= 15 and m["license"] == "CC0-1.0"
        cats = {c["category"] for c in clips}
        assert {"whoosh", "pop", "impact", "click", "riser"} <= cats
        lic = (REPO / "sfx" / "LICENSES.md").read_text()
        assert "CC0" in lic
        for c in clips:
            assert (REPO / "sfx" / c["file"]).exists()
            assert c["file"] in lic
            assert c["source"]["url"].startswith("https://kenney.nl/assets/")

    def test_plan_rates_variety_and_hook(self):
        duration = 60.0
        beats = [round(4.0 * k, 2) for k in range(1, 15)]
        ev = sfx.plan_sfx(duration, beats, sub_cuts=[b + 2.0 for b in beats], hook_t=0.1)
        assert ev[0]["kind"] == "hook" and ev[0]["clip"].startswith("impact")
        rate = sfx.transitions_per_minute(ev, duration)
        assert 6 <= rate <= 12
        trans = [e for e in ev if e["kind"] == "transition"]
        assert all(a["clip"] != b["clip"] for a, b in zip(trans, trans[1:]))
        assert {e["clip"].split("_")[0] for e in trans} == {"whoosh", "pop"}

    def test_floor_uses_subcuts_when_few_beats(self):
        ev = sfx.plan_sfx(60.0, [30.0], sub_cuts=[float(t) for t in range(3, 60, 3)])
        assert sfx.transitions_per_minute(ev, 60.0) >= 6

    def test_clicks_inside_typing_window_and_riser_lands(self):
        ev = sfx.plan_sfx(20.0, [5.0, 10.0, 15.0], typing=[(10.3, 12.0)], riser_at=10.0)
        clicks = [e for e in ev if e["kind"] == "typing"]
        assert clicks and all(10.3 <= e["t"] < 12.0 for e in clicks)
        riser = [e for e in ev if e["kind"] == "riser"][0]
        dur = next(c for c in sfx.load_manifest()["clips"] if c["id"] == riser["clip"])["duration_s"]
        assert abs(riser["t"] + dur - 10.0) < 0.01

    def test_whoosh_peak_lands_on_cut(self):
        ev = sfx.plan_sfx(30.0, [10.0, 20.0], hook_t=0.0)
        for e in ev:
            if e["kind"] == "transition":
                clip = next(c for c in sfx.load_manifest()["clips"] if c["id"] == e["clip"])
                assert abs(e["t"] + clip["peak_s"] - e["at"]) < 0.01

    def test_render_track_sits_under_voice(self, tmp_path):
        np = pytest.importorskip("numpy")
        voice = tmp_path / "v.wav"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=f=220:d=4:sample_rate=48000",
                        "-af", "volume=-6dB", str(voice)], check=True)
        ev = [{"t": 1.0, "at": 1.0, "clip": "whoosh_02", "kind": "transition", "gain_db": 0.0}]
        out = sfx.render_sfx_track(ev, 4.0, voice, tmp_path / "s.wav")
        with wave.open(str(out)) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2") / 32767
        v = sfx._decode(voice)
        diff = 20 * np.log10(np.abs(v).max()) - 20 * np.log10(np.abs(x).max())
        assert 11.0 <= diff <= 13.0
        assert len(x) == 4 * 48000

    def test_empty_plan_renders_nothing(self, tmp_path):
        assert sfx.render_sfx_track([], 5.0, tmp_path / "v.wav", tmp_path / "s.wav") is None

    def test_sfx_inputs_from_plan(self):
        sources = [{"kind": "motion", "block": "kinetic_text", "data": {}},
                   {"kind": "motion", "block": "count_up", "data": {}},
                   {"kind": "motion", "block": "code_panel", "data": {"lines": ["$ ls", "a b"], "language": "bash"}}]
        plan = [{"source": 0, "duration": 5.0, "cuts": [2.5, 2.5]},
                {"source": 1, "duration": 3.0, "cuts": [3.0]},
                {"source": 2, "duration": 3.0, "cuts": [3.0]}]
        info = assemble.sfx_inputs(plan, sources, [{"word": "Hi", "start": 0.2, "end": 0.4}])
        assert info["beat_changes"] == [5.0, 8.0] and info["sub_cuts"] == [2.5]
        assert info["hook_t"] == 0.2 and info["riser_at"] == 5.0
        t0, t1 = code_typing_window(sources[2]["data"], 3.0)
        assert info["typing"] == [(8.0 + t0, 8.0 + t1)]


# ───────────────────────── audio mix ─────────────────────────

class TestMix:
    def test_music_graph_with_sfx(self):
        g = assemble.music_filter_graph(60.0, -26, sfx=True)
        assert "[voice][ducked][3:a]amix=inputs=3" in g and "sidechaincompress" in g
        assert "[3:a]" not in assemble.music_filter_graph(60.0, -26)

    def test_prepare_audio_adds_sfx_input(self, monkeypatch, tmp_path):
        cmds = []
        monkeypatch.setattr(assemble, "run_cmd", lambda cmd, **k: cmds.append(cmd) or type("R", (), {"stderr": "", "stdout": ""})())
        music, fx = tmp_path / "m.mp3", tmp_path / "s.wav"
        music.write_bytes(b"x")
        fx.write_bytes(b"x")
        assemble.prepare_audio(tmp_path / "v.wav", tmp_path, 30.0, str(music), -26, sfx_path=str(fx))
        assert str(fx) in cmds[0] and "amix=inputs=3" in " ".join(cmds[0])
        assert any("loudnorm=I=-14" in " ".join(c) for c in cmds[1:])
        cmds.clear()
        assemble.prepare_audio(tmp_path / "v.wav", tmp_path, 30.0, None, -26, sfx_path=str(fx))
        assert "[0:a][1:a]amix=inputs=2" in " ".join(cmds[0])


# ───────────────────────── grade + QC ─────────────────────────

class TestGradeAndQc:
    def test_grade_filter_is_light(self):
        g = assemble.GRADE_FILTER
        assert "noise=alls=4:allf=t" in g and "vignette" in g and "eq=contrast=1.06" in g

    def test_parsers(self):
        log = """[Parsed_ebur128_0 @ 0x1] Summary:

  Integrated loudness:
    I:         -14.2 LUFS
    Threshold: -24.5 LUFS

  Loudness range:
    LRA:         5.1 LU

  True peak:
    Peak:       -1.6 dBFS
[silencedetect @ 0x2] silence_start: 12.5
[silencedetect @ 0x2] silence_end: 13.4 | silence_duration: 0.9
[blackdetect @ 0x3] black_start:0 black_end:0.6 black_duration:0.6
[freezedetect @ 0x4] lavfi.freezedetect.freeze_start: 20.1
[freezedetect @ 0x4] lavfi.freezedetect.freeze_duration: 4.2
[freezedetect @ 0x4] lavfi.freezedetect.freeze_end: 24.3
"""
        assert qc.parse_ebur128(log) == {"integrated_lufs": -14.2, "lra_lu": 5.1, "true_peak_dbfs": -1.6}
        assert qc.parse_intervals(log, "silence") == [{"start": 12.5, "end": 13.4, "duration": 0.9}]
        assert qc.parse_intervals(log, "black")[0]["duration"] == 0.6
        assert qc.parse_intervals(log, "freeze")[0] == {"start": 20.1, "end": 24.3, "duration": 4.2}
        rep = {"loudness": {"integrated_lufs": -17.0, "true_peak_dbfs": -0.5},
               "silences": qc.parse_intervals(log, "silence"), "black": [], "freezes": []}
        w = qc.qc_warnings(rep)
        assert len(w) == 3 and any("LUFS" in x for x in w)

    def test_run_qc_on_tiny_render_writes_json_and_sheet(self, tmp_path):
        video = tmp_path / "v.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=216x384:r=30:d=3",
                        "-f", "lavfi", "-i", "sine=f=440:d=3", "-shortest", "-c:v", "libx264", "-preset", "ultrafast",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", str(video)], check=True)
        rep = qc.run_qc(video, tmp_path / "qc.json", tmp_path / "sheet.png")
        assert (tmp_path / "qc.json").exists() and (tmp_path / "sheet.png").exists()
        data = json.loads((tmp_path / "qc.json").read_text())
        assert data["loudness"]["integrated_lufs"] < 0
        assert data["status"] in {"ok", "warn"} and data["freezes"] == []
        assert rep["contact_sheet"] == "sheet.png"

    def test_run_qc_never_raises(self, tmp_path):
        rep = qc.run_qc(tmp_path / "missing.mp4", tmp_path / "qc.json", tmp_path / "s.png")
        assert rep["status"] == "warn" and rep["errors"]
