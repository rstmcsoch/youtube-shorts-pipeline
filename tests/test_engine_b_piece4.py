"""Engine B piece 4: no static text holds, sourced on-screen number, catch beat,
kinetic_text <= 30%, motion takeaway as the final beat."""

import json
from unittest.mock import patch

import pytest

from verticals.beats import (enforce_sourced_numbers, kinetic_cap, normalize_beats, research_figures,
                             structure_problems, variety_ok)
from verticals.motion import longest_static, motion_cards
from verticals.pacing import pace_limits, shot_groups

MAX_STATIC = pace_limits()[1]  # 3.5 s from the style guide


def _words(text: str, start: float = 0.0, gap: float = 0.36):
    out, t = [], start
    for w in text.split():
        out.append({"word": w, "start": round(t, 3), "end": round(t + 0.33, 3)})
        t += gap
    return out


NARRATION = _words(
    "Banks score every card swipe in milliseconds. A model compares the purchase with your history, "
    "location and device, then flags anything that looks unusual. Each transaction gets a risk score, "
    "and only the riskiest ones are blocked or sent for review by a human analyst, while the rest pass.")

BLOCKS = {
    "kinetic_text": {"words": ["risk", "score"], "accent": "score"},
    "count_up": {"value": 844.8, "decimals": 1, "prefix": "", "suffix": "M", "label": "UK card fraud losses",
                 "source": "Source: Wikipedia, Credit card fraud"},
    "diagram": {"nodes": ["Swipe", "Risk score"], "title": "Scoring"},
    "code_panel": {"lines": ["$ score --txn 4411", "risk: 0.97", "action: block"], "language": "bash",
                   "title": "terminal"},
}


@pytest.mark.parametrize("kind", sorted(BLOCKS))
def test_no_rendered_segment_static_over_limit(kind):
    """A long motion beat (the 8-9 s holds of draft 2) never shows the same picture > 3.5 s."""
    groups = shot_groups(NARRATION[-1]["end"], NARRATION, 4.0, n_shots=1)  # one ~13 s beat
    cuts = groups[0]
    assert sum(cuts) > 8 and max(cuts) <= MAX_STATIC
    specs = motion_cards(kind, BLOCKS[kind], cuts, 0.0, NARRATION)
    assert longest_static(specs, sample_fps=5) <= MAX_STATIC


def test_static_detector_catches_the_old_single_card():
    """The old render (one kinetic card for the whole beat, only a slow push) is caught."""
    old = [{"kind": "kinetic_text", "data": BLOCKS["kinetic_text"], "duration": 8.7, "t_offset": 0.0, "total": 8.7}]
    assert longest_static(old, sample_fps=5) > MAX_STATIC


def test_kinetic_cards_follow_the_narration():
    cuts = [3.0, 3.0, 3.0]
    specs = motion_cards("kinetic_text", BLOCKS["kinetic_text"], cuts, 0.0, NARRATION)
    assert len(specs) == 3 and specs[0]["data"] == BLOCKS["kinetic_text"]
    texts = [tuple(s["data"]["words"]) for s in specs]
    assert len(set(texts)) == 3
    spoken = {w["word"].strip(".,").lower() for w in NARRATION}
    assert all(w.lower() in spoken for s in specs[1:] for w in s["data"]["words"])


def _kt(words, kw=None, role=None):
    b = {"block": "kinetic_text", "keywords": kw or words, "fallback": "diagram", "data": {"words": words}}
    if role:
        b["role"] = role
    return b


def test_kinetic_text_capped_at_30_percent():
    raw = [_kt([f"w{i}", "x"], kw=[f"Step {i}", "model", "score"]) for i in range(8)]
    beats, warns = normalize_beats(raw, n_segments=8)
    assert sum(b["block"] == "kinetic_text" for b in beats) <= kinetic_cap(8) == 2
    assert variety_ok(beats)
    assert beats[-1]["block"] != "stock_or_ai_still"
    assert any("30%" in w for w in warns)


def test_final_beat_is_never_a_still():
    raw = [{"block": "diagram", "keywords": ["a", "b"], "data": {"nodes": ["A", "B"]}},
           {"block": "code_panel", "keywords": ["c"], "data": {"lines": ["x = 1"], "language": "python"}},
           {"block": "stock_or_ai_still", "keywords": ["fraud", "future"], "fallback": "stock_or_ai_still"}]
    beats, _ = normalize_beats(raw, n_segments=3)
    assert beats[-1]["block"] in ("kinetic_text", "diagram", "count_up", "code_panel")
    assert beats[-1]["role"] == "takeaway" and beats[0]["role"] == "hook"


def test_at_most_two_stills():
    raw = [_kt(["hook", "now"])] + [{"block": "stock_or_ai_still", "keywords": [f"k{i}", "card", "bank"]}
                                    for i in range(6)] + [_kt(["the", "end"])]
    beats, _ = normalize_beats(raw, n_segments=8)
    assert sum(b["block"] == "stock_or_ai_still" for b in beats) <= 2


RESEARCH = ("[source: Wikipedia, Credit card fraud] In 2018, unauthorised financial fraud losses totalled "
            "£844.8 million in the United Kingdom. Banks prevented £1.66 billion.")


def test_research_figures_carry_their_source():
    figs = research_figures(RESEARCH)
    assert (844.8, "Wikipedia, Credit card fraud") in figs and any(v == 1.66 for v, _ in figs)
    assert research_figures("Topic: x\n(No live research available — script must stay general.)") == []


def test_sourced_count_up_is_kept_and_cited():
    beats, _ = normalize_beats([{"block": "count_up", "keywords": ["losses"],
                                 "data": {"value": 844.8, "suffix": "M", "label": "UK fraud losses"}},
                                {"block": "diagram", "keywords": ["a", "b"], "data": {"nodes": ["A", "B"]}}],
                               n_segments=2)
    beats, warns, ok = enforce_sourced_numbers(beats, RESEARCH)
    assert ok and beats[0]["block"] == "count_up"
    assert "Wikipedia" in beats[0]["data"]["source"]


def test_invented_figure_becomes_a_diagram_not_a_number():
    raw = [_kt(["fraud", "now"]),
           {"block": "count_up", "keywords": ["fraud", "detection", "accuracy"], "fallback": "diagram",
            "data": {"value": 99.9, "suffix": "%", "label": "accuracy"}},
           _kt(["the", "end"])]
    beats, _ = normalize_beats(raw, n_segments=3)
    beats, warns, ok = enforce_sourced_numbers(beats, RESEARCH)
    assert not ok and all(b["block"] != "count_up" for b in beats)
    assert any(b["block"] == "diagram" for b in beats)
    assert any("not found in research" in w for w in warns)


def test_no_research_numbers_means_a_diagram():
    raw = [_kt(["fraud", "now"], kw=["fraud", "now"]), _kt(["swipe", "score"], kw=["swipe", "score", "block"]),
           {"block": "code_panel", "keywords": ["c"], "data": {"lines": ["x = 1"], "language": "python"}},
           _kt(["the", "end"])]
    beats, _ = normalize_beats(raw, n_segments=4)
    beats, warns, ok = enforce_sourced_numbers(beats, "Topic: x\n(No live research available — stay general.)")
    assert not ok and any(b["block"] == "diagram" for b in beats)


def test_structure_needs_a_catch_beat():
    beats = [{"block": "kinetic_text", "role": "hook"}, {"block": "diagram", "role": "how"},
             {"block": "kinetic_text", "role": "takeaway"}]
    assert any("catch" in p for p in structure_problems("It scores every swipe.", beats))
    beats[1]["role"] = "catch"
    assert structure_problems("But it still flags honest buyers as false positives.", beats) == []


def test_final_motion_shot_replaces_a_trailing_still(tmp_path):
    from verticals.assemble import final_motion_shot
    sources = [{"kind": "motion", "block": "diagram", "segment": 0},
               {"kind": "still", "segment": 1, "path": tmp_path / "a.png"},
               {"kind": "motion", "block": "kinetic_text", "segment": 2}]
    plan = [{"source": 0, "duration": 4}, {"source": 2, "duration": 4}, {"source": 1, "duration": 4}]
    out = final_motion_shot(plan, sources)
    assert sources[out[-1]["source"]]["kind"] == "motion"


def _draft_json(script, beats):
    return json.dumps({"script": script, "broll_prompts": ["p"] * len(beats), "beats": beats,
                       "youtube_title": "T", "youtube_description": "D", "youtube_tags": "t"})


@patch("verticals.draft.research_topic", return_value=RESEARCH)
@patch("verticals.draft.call_llm")
def test_missing_number_and_catch_get_one_rewrite(mock_llm, _research):
    from verticals.draft import generate_draft
    body = " ".join(["word"] * 130)
    hook = "Fraud cost 844.8 million pounds in one year. "
    plain = [_kt(["fraud"], kw=["fraud", "cost"]), {"block": "diagram", "keywords": ["a", "b"],
                                                    "data": {"nodes": ["Swipe", "Score"]}},
             _kt(["the", "end"], kw=["end", "card"])]
    fixed = [{"block": "count_up", "role": "hook", "keywords": ["losses"],
              "data": {"value": 844.8, "suffix": "M", "label": "UK fraud losses"}},
             {"block": "diagram", "role": "catch", "keywords": ["a", "b"], "data": {"nodes": ["Swipe", "Score"]}},
             _kt(["the", "end"], kw=["end", "card"], role="takeaway")]
    mock_llm.side_effect = [_draft_json(hook + body, plain),
                            _draft_json(hook + body + " But it still flags honest buyers.", fixed)]
    draft = generate_draft("How AI detects credit card fraud")
    assert "STRUCTURE CORRECTION" in mock_llm.call_args_list[1][0][0]
    assert draft["beats"][0]["block"] == "count_up" and "Wikipedia" in draft["beats"][0]["data"]["source"]
    assert any(b["role"] == "catch" for b in draft["beats"])


@patch("verticals.draft.research_topic", return_value="research")
@patch("verticals.draft.call_llm")
def test_prompt_states_structure_number_and_variety_rules(mock_llm, _research):
    from verticals.draft import generate_draft
    mock_llm.return_value = json.dumps({"script": "s", "broll_prompts": ["p"]})
    generate_draft("Test")
    prompt = mock_llm.call_args_list[0][0][0]
    assert "STRUCTURE (in this order)" in prompt and "the catch" in prompt
    assert "ON-SCREEN NUMBER" in prompt and "NEVER invent" in prompt
    assert "kinetic_text for at most 2 beats" in prompt
