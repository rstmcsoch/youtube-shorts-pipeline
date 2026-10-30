"""Pre-render personal-names gate, false-positive fixes, and reviewer feedback."""
import json
import subprocess
import sys
from unittest.mock import patch

import pytest

from verticals.draft import (
    PersonalNamesError, REVIEWER_FEEDBACK_MAX_CHARS, generate_draft, reviewer_feedback_text,
)
from verticals.name_gate import find_names, flagged_phrases, scan_text


def _draft(**overrides):
    base = {
        "script": "Security teams drown in alerts every day.",
        "broll_prompts": ["Server room lights", "Hands on a keyboard"],
        "youtube_title": "Why AI Security Alerts Are Flawed",
        "youtube_description": "How AI triages security alerts.",
        "youtube_tags": "ai,security",
        "instagram_caption": "AI and security.",
        "tiktok_caption": "AI and security.",
        "thumbnail_prompt": "Dark security operations room",
    }
    base.update(overrides)
    return json.dumps(base)


# ---------------------------------------------------------------- false positives

@pytest.mark.parametrize("text", [
    "Shares rose on Wall Street after the Associated Press report.",
    "Anthropic's Massive Upgrade",
    "It topped the Artificial Analysis Index and the Artificial Analysis Intelligence Index.",
    "Gemini 4 Carbon Changes Cybersecurity Forever",
    "Morgan Stanley and Goldman Sachs raised targets; Wells Fargo said no.",
    "Charles Schwab clients and Dow Jones traders reacted.",
    "The Washington Post and The New York Times covered it; Fox News too.",
    "Victoria Falls power and the Jordan River basin.",
    "Researchers at Georgia Tech in Austin Texas",
    "Harley Davidson and Mercedes Benz unveiled EVs.",
    "Nvidia Grace Blackwell Superchip and the Vera Rubin Platform",
    "Florence Nightingale Hospital deployed an AI triage tool.",
    "Will Robots Take Jobs?",
    "Max Power Mode Explained",
    "Grace Period Ends For Old Chips",
    "Alexa Plus Arrives; Claude Code Gets Faster; Devin Writes Code",
    "The Dallas Fed and Chase Bank flagged fraud.",
    "Norton Coupon Codes",
    "Tom's Hardware tested the chip.",
])
def test_org_product_place_phrases_are_not_names(text):
    assert scan_text(text) == []


@pytest.mark.parametrize("text", [
    "Jane Doe explains how AI agents work.",
    "Alex Johnson wrote it.",
    "Marques Brownlee reviewed the phone.",
    "Linus Torvalds released a kernel.",
    "Carl Pei's Nothing Phone",
    "Will Smith stars in it.",
    "Mark Smith said so.",
    "Robert Kim leads the lab.",
    "Priya Sharma said the model is fast.",
    "Hunter Biden",
    "Sam Altman's OpenAI",
    "Lisa Su announced a chip.",
    "Jensen Huang said",
])
def test_real_person_names_still_block(text):
    assert scan_text(text)


def test_possessive_and_curly_apostrophe_names_block():
    assert scan_text("John Carmack\u2019s new engine")


def test_find_names_reports_fields_and_phrases():
    found = find_names({"youtube_title": "Jane Doe Explains AI", "script": "Fine text.",
                        "visual_intents": [{"entities": ["Alex Johnson"]}]})
    assert set(found) == {"youtube_title", "visual_intents[0].entities"}
    assert flagged_phrases(found) == ["Alex Johnson", "Jane Doe Explains"]


# ---------------------------------------------------------------- pre-render gate

@patch("verticals.draft.research_topic", return_value="research")
@patch("verticals.draft.call_llm")
def test_prerender_gate_regenerates_once_with_feedback(mock_llm, _research):
    mock_llm.side_effect = [
        _draft(youtube_title="Jane Doe Explains AI Alerts"),
        _draft(),
    ]
    draft = generate_draft("AI security alerts", niche="tech")
    assert draft["youtube_title"] == "Why AI Security Alerts Are Flawed"
    assert mock_llm.call_count == 2
    retry_prompt = mock_llm.call_args_list[1].args[0]
    assert "NAME CORRECTION" in retry_prompt and "Jane Doe" in retry_prompt.split("NAME CORRECTION")[1]


@patch("verticals.draft.research_topic", return_value="research")
@patch("verticals.draft.call_llm")
def test_prerender_gate_fails_fast_after_one_retry(mock_llm, _research, capsys):
    mock_llm.side_effect = [
        _draft(script="Jane Doe says alerts are noisy."),
        _draft(script="Jane Doe still says alerts are noisy."),
    ]
    with pytest.raises(PersonalNamesError):
        generate_draft("AI security alerts", niche="tech")
    assert mock_llm.call_count == 2
    out = capsys.readouterr().out
    assert "DYPOL_FAILURE_CATEGORY=content_gate" in out


@patch("verticals.draft.research_topic", return_value="research")
@patch("verticals.draft.call_llm")
def test_prerender_gate_passes_false_positive_phrases_without_retry(mock_llm, _research):
    mock_llm.side_effect = [_draft(
        script="Stocks on Wall Street jumped, the Associated Press said, after Anthropic's Massive launch "
               "topped the Artificial Analysis Index.",
        youtube_title="Anthropic's Massive Upgrade",
    )]
    draft = generate_draft("AI stocks", niche="tech")
    assert "Wall Street" in draft["script"]
    assert mock_llm.call_count == 1


@patch("verticals.draft.research_topic", return_value="research")
@patch("verticals.draft.call_llm")
def test_prerender_gate_can_be_disabled(mock_llm, _research, monkeypatch):
    monkeypatch.setenv("DYPOL_PRERENDER_NAME_GATE", "0")
    mock_llm.side_effect = [_draft(youtube_title="Jane Doe Explains AI")]
    assert generate_draft("x", niche="tech")["youtube_title"] == "Jane Doe Explains AI"


def test_cli_draft_exits_nonzero_on_blocked_names(tmp_path, monkeypatch):
    """`python -m verticals draft` stops with exit 3 before produce when names persist."""
    code = (
        "import sys, json\n"
        "from unittest.mock import patch\n"
        "bad = json.dumps({'script': 'Jane Doe says hi.', 'broll_prompts': ['a'], 'youtube_title': 'T'})\n"
        "with patch('verticals.draft.research_topic', return_value='r'), "
        "patch('verticals.draft.call_llm', return_value=bad):\n"
        "    sys.argv = ['verticals', 'draft', '--topic', 'x', '--niche', 'tech', '--provider', 'gemini']\n"
        "    from verticals.__main__ import main\n"
        "    main()\n"
    )
    env = dict(**__import__("os").environ, VERTICALS_HOME=str(tmp_path))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert result.returncode == 3, result.stdout + result.stderr
    assert "DYPOL_FAILURE_CATEGORY=content_gate" in result.stdout


# ---------------------------------------------------------------- reviewer feedback

def test_reviewer_feedback_is_capped_and_whitespace_normalised(monkeypatch):
    monkeypatch.setenv("REVIEWER_FEEDBACK", "  shorter   hook\n" + "x" * 5000)
    text = reviewer_feedback_text()
    assert text.startswith("shorter hook ")
    assert len(text) == REVIEWER_FEEDBACK_MAX_CHARS


@patch("verticals.draft.research_topic", return_value="research")
@patch("verticals.draft.call_llm")
def test_reviewer_feedback_is_injected_into_script_prompt(mock_llm, _research, monkeypatch):
    monkeypatch.setenv("REVIEWER_FEEDBACK", "Make the hook about alert fatigue.")
    mock_llm.side_effect = [_draft()]
    generate_draft("AI security alerts", niche="tech")
    prompt = mock_llm.call_args_list[0].args[0]
    assert "REVIEWER FEEDBACK" in prompt and "Make the hook about alert fatigue." in prompt


@patch("verticals.draft.research_topic", return_value="research")
@patch("verticals.draft.call_llm")
def test_no_feedback_block_without_feedback(mock_llm, _research, monkeypatch):
    monkeypatch.delenv("REVIEWER_FEEDBACK", raising=False)
    mock_llm.side_effect = [_draft()]
    generate_draft("AI security alerts", niche="tech")
    assert "REVIEWER FEEDBACK" not in mock_llm.call_args_list[0].args[0]


def test_post_render_cli_prints_failure_category(tmp_path):
    draft_path = tmp_path / "d.json"
    draft_path.write_text(json.dumps({"script": "Jane Doe says hi."}), encoding="utf-8")
    result = subprocess.run([sys.executable, "scripts/validate_no_personal_names.py", "--draft", str(draft_path)],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "DYPOL_FAILURE_CATEGORY=content_gate" in result.stdout


def test_reviewer_feedback_accepts_console_env_name(monkeypatch):
    monkeypatch.delenv("REVIEWER_FEEDBACK", raising=False)
    monkeypatch.setenv("DYPOL_REVIEWER_FEEDBACK", "Use a calmer tone.")
    assert reviewer_feedback_text() == "Use a calmer tone."
