"""Shot-list beats: closed enum, fallbacks, no-3-in-a-row, hook checks, assembly wiring."""

from verticals.beats import hook_warnings, normalize_beats, variety_ok
from verticals.style import block_kinds, hook_rules, max_same_run, load_style


def test_style_guide_values():
    style = load_style()
    assert style["pace"]["target_wpm"] == 158
    assert style["screen"]["max_static_s"] == 3.5
    assert style["screen"]["change_every_s"] == [2.0, 4.0]
    assert max_same_run() == 2
    assert hook_rules()["first_word_by_s"] == 0.8 and hook_rules()["number_by_s"] == 8.0
    assert set(block_kinds()) == {"kinetic_text", "count_up", "code_panel", "diagram", "stock_or_ai_still"}
    assert style["look"]["palette"]["background"] == "#0F141C"


def _kt(words):
    return {"block": "kinetic_text", "keywords": words, "data": {"words": words}}


def test_unknown_kind_uses_fallback():
    beats, warns = normalize_beats([
        {"block": "hologram", "fallback": "diagram", "keywords": ["a"], "data": {"nodes": ["A", "B"]}},
    ], n_segments=1)
    assert beats[0]["block"] == "diagram"
    assert warns


def test_bad_data_falls_back_and_hook_never_stock():
    beats, _ = normalize_beats([
        {"block": "count_up", "keywords": ["fast", "model"], "fallback": "stock_or_ai_still", "data": {"value": "lots"}},
        {"block": "stock_or_ai_still", "keywords": ["datacenter"]},
        {"block": "diagram", "keywords": ["a"], "data": {"nodes": ["A", "B"]}},
    ], n_segments=3)
    assert beats[0]["block"] == "kinetic_text"  # stock is not allowed in the hook
    assert beats[0]["data"]["words"] == ["fast", "model"]
    assert beats[1]["block"] == "stock_or_ai_still"


def test_no_three_in_a_row_is_auto_fixed():
    raw = [_kt(["one"]), _kt(["two"]), _kt(["three"]), _kt(["four"]), _kt(["five"]), _kt(["six"])]
    beats, warns = normalize_beats(raw, n_segments=6)
    assert variety_ok(beats)
    assert all(b["block"] in block_kinds() for b in beats)
    assert any("in a row" in w for w in warns)


def test_missing_beats_are_filled_from_prompts():
    beats, _ = normalize_beats(None, n_segments=8, prompts=["server racks glowing"] * 8)
    assert len(beats) == 8 and variety_ok(beats)
    assert beats[0]["block"] != "stock_or_ai_still"


def test_hook_warnings():
    good = "Seventy percent cheaper overnight. " + "word " * 120 + "That's the real story."
    assert hook_warnings(good) == []
    bad = "Hey everyone, welcome back to the channel today. " + "word " * 120 + "Subscribe for more."
    warns = hook_warnings(bad)
    assert any("number" in w for w in warns) and any("intro" in w for w in warns) and any("call-to-action" in w for w in warns)


def test_build_sources_adds_motion_and_demotes_still(tmp_path):
    from verticals.assemble import build_sources, plan_shots
    frames = []
    for i in range(4):
        f = tmp_path / f"broll_{i}.png"
        f.write_bytes(b"x")
        frames.append(f)
    beats = [{"block": "count_up", "data": {"value": 5, "label": "x"}},
             {"block": "stock_or_ai_still", "data": {}},
             {"block": "diagram", "data": {"nodes": ["A", "B"]}},
             {"block": "stock_or_ai_still", "data": {}}]
    srcs = build_sources(frames, tmp_path, beats)
    motion = [s for s in srcs if s["kind"] == "motion"]
    assert [m["block"] for m in motion] == ["count_up", "diagram"]
    assert all(s.get("backup") for s in srcs if s["kind"] == "still" and s["segment"] in (0, 2))
    plan = plan_shots(srcs, 16.0)
    first = srcs[plan[0]["source"]]
    assert first["kind"] == "motion" and first["block"] == "count_up"


def test_name_gate_scans_beat_text():
    from verticals.name_gate import find_names
    found = find_names({"beats": [{"block": "kinetic_text", "data": {"words": ["Elon", "Musk"]}}]})
    assert any(k.startswith("beats[") for k in found)
