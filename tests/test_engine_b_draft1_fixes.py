"""Fixes from the first real Engine B draft (job-mvz94hhv-5zshzk, 2 Nov 2026)."""

from pathlib import Path

from verticals import draft as draft_mod
from verticals.assemble import build_sources, plan_shots, shot_count
from verticals.beats import hook_warnings
from verticals.captions import DYPOL_CAPTION, _chunk_font_size, dypol_caption_chunks, fit_caption_chunks
from verticals.pacing import group_cuts
from verticals.sfx import TRANSITION_GAIN


def _sources_like_draft1():
    motion = [{"kind": "motion", "block": "diagram", "segment": s, "max_uses": 1} for s in (0, 1, 2, 3, 5, 6)]
    backups = [{"kind": "still", "segment": s, "max_uses": 1, "backup": True} for s in (0, 1, 2, 3, 5, 6)]
    clip3 = {"kind": "clip", "segment": 3, "max_uses": 1, "backup": True}
    stock = [{"kind": "still", "segment": s, "max_uses": 2} for s in (4, 7)]
    return motion + backups + [clip3] + stock


def test_shot_count_one_per_primary_source():
    src = _sources_like_draft1()
    assert shot_count(src, 45.27) == 8            # 6 motion + 2 stock beats
    assert shot_count(src, 90.0) == 12            # floor: no shot longer than ~8 s
    assert shot_count(src * 3, 45.27) == 12       # cap: the old duration / 4 s count


def test_planner_never_falls_back_to_backup_stills_at_the_tail():
    src = _sources_like_draft1()
    segs = [2.9, 2.4, 2.8, 2.9, 2.33, 2.6, 2.9, 2.83, 3.1, 2.67, 2.7, 2.43, 2.6, 3.1, 3.47, 1.77, 1.75]
    groups = group_cuts(segs, shot_count(src, sum(segs)))
    plan = plan_shots(src, sum(segs), durations=[sum(g) for g in groups])
    picked = [src[s["source"]] for s in plan]
    assert not any(p.get("backup") for p in picked)
    still_time = sum(s["duration"] for s, p in zip(plan, picked) if p["kind"] != "motion")
    assert still_time / sum(segs) < 0.35
    assert max(s["duration"] for s in plan) <= 8.0


def test_clip_in_a_motion_segment_is_a_backup(tmp_path):
    import json
    (tmp_path / "clip_0.mp4").write_bytes(b"x")
    (tmp_path / "broll_manifest.json").write_text(json.dumps({"assets": [
        {"frame": "broll_0.png", "segment_index": 0, "provider": "pixabay", "clip": {"file": "clip_0.mp4"}}]}))
    beats = [{"block": "kinetic_text", "data": {"words": ["a", "b"]}}]
    src = build_sources([tmp_path / "broll_0.png"], tmp_path, beats)
    clip = next(s for s in src if s["kind"] == "clip")
    assert clip.get("backup") is True


def test_group_cuts_by_time_keeps_shots_even():
    segs = [3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0]
    groups = group_cuts(segs, 4)
    assert len(groups) == 4 and sum(len(g) for g in groups) == len(segs)
    lens = [sum(g) for g in groups]
    assert max(lens) - min(lens) <= 3.0 + 1e-9


def _words(text, start=0.0, step=0.3):
    out, t = [], start
    for w in text.split():
        out.append({"word": w, "start": t, "end": t + step * 0.8})
        t += step
    return out


def test_fit_caption_chunks_splits_rather_than_shrinking():
    chunks = dypol_caption_chunks(_words("recognize individual letters. messy scribbles instantly."))
    fitted = fit_caption_chunks(chunks)
    cfg = DYPOL_CAPTION
    for c in fitted:
        fs = _chunk_font_size(" ".join(w["word"] for w in c), cfg["size"], cfg["max_width"], cfg["min_size"], cfg["outline"])
        assert len(c) == 1 or fs >= cfg["size"] - 16
    assert [w["word"] for c in fitted for w in c] == [w["word"] for c in chunks for w in c]


def test_hook_warns_on_setup_openers():
    bad = "Let me explain how AI reads your handwriting in sixty seconds because most people are wrong. " * 3
    assert "hook: script opens with an intro/greeting" in hook_warnings(bad)


def test_enforce_hook_takes_a_better_rewrite(monkeypatch):
    body = " ".join(["word"] * 140)
    bad = {"script": "Let me explain how AI reads handwriting in sixty seconds. " + body}
    good = {"script": "Handwriting AI now reads 98 percent of messy cursive correctly. " + body}
    monkeypatch.setattr(draft_mod, "_generate_json_response", lambda prompt, provider: good)
    assert draft_mod._enforce_hook(dict(bad), "p", None)["script"] == good["script"]
    monkeypatch.setattr(draft_mod, "_generate_json_response", lambda prompt, provider: bad)
    assert draft_mod._enforce_hook(dict(bad), "p", None)["script"] == bad["script"]


def test_transitions_sit_under_the_hook_impact():
    assert TRANSITION_GAIN["whoosh"] <= -5 and TRANSITION_GAIN["pop"] <= -5
    assert TRANSITION_GAIN["impact"] > TRANSITION_GAIN["whoosh"]


def test_no_niche_hook_template_is_setup():
    import yaml
    from verticals.beats import INTRO_RE
    for p in Path(__file__).resolve().parent.parent.joinpath("niches").glob("*.yaml"):
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        for h in (data.get("script") or {}).get("hooks") or []:
            assert not INTRO_RE.search(str(h.get("template", ""))), (p.name, h)
