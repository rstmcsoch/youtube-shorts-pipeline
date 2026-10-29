"""Tests for pipeline/draft.py — draft generation with mocked Claude API."""

import json
from unittest.mock import patch, MagicMock

from verticals.draft import generate_draft


class TestGenerateDraft:
    @patch("verticals.draft.research_topic")
    @patch("verticals.draft.call_llm")
    def test_basic_draft_generation(self, mock_llm, mock_research):
        mock_research.return_value = "Some research data about the topic."
        mock_llm.return_value = json.dumps({
            "script": "This is a test script about AI.",
            "broll_prompts": ["Prompt 1", "Prompt 2", "Prompt 3"],
            "youtube_title": "AI Revolution 2026",
            "youtube_description": "All about AI.",
            "youtube_tags": "ai,tech,2026",
            "instagram_caption": "AI is changing the world!",
            "thumbnail_prompt": "Futuristic AI image",
        })

        draft = generate_draft("AI is changing everything in 2026")

        assert draft["script"] == "This is a test script about AI."
        assert len(draft["broll_prompts"]) == 3
        assert draft["youtube_title"] == "AI Revolution 2026"
        assert draft["news"] == "AI is changing everything in 2026"
        assert draft["research"] == "Some research data about the topic."

    @patch("verticals.draft.research_topic")
    @patch("verticals.draft.call_llm")
    def test_handles_code_block_wrapper(self, mock_llm, mock_research):
        mock_research.return_value = "research"
        mock_llm.return_value = '```json\n{"script":"test","broll_prompts":["p1","p2","p3"],"youtube_title":"T","youtube_description":"D","youtube_tags":"t","instagram_caption":"C","thumbnail_prompt":"P"}\n```'

        draft = generate_draft("Test topic")
        assert draft["script"] == "test"

    @patch("verticals.draft.research_topic")
    @patch("verticals.draft.call_llm")
    def test_sanitizes_non_string_fields(self, mock_llm, mock_research):
        mock_research.return_value = "research"
        mock_llm.return_value = json.dumps({
            "script": 12345,  # non-string
            "broll_prompts": "not a list",  # non-list
            "youtube_title": "T",
            "youtube_description": "D",
            "youtube_tags": "t",
            "instagram_caption": "C",
            "thumbnail_prompt": "P",
        })

        draft = generate_draft("Test")
        assert isinstance(draft["script"], str)
        assert isinstance(draft["broll_prompts"], list)
        assert len(draft["broll_prompts"]) == 8  # fallback

    @patch("verticals.draft.research_topic")
    @patch("verticals.draft.call_llm")
    def test_includes_channel_context(self, mock_llm, mock_research):
        mock_research.return_value = "research"
        mock_llm.return_value = json.dumps({
            "script": "s", "broll_prompts": ["p1", "p2", "p3"],
            "youtube_title": "T", "youtube_description": "D",
            "youtube_tags": "t", "instagram_caption": "C",
            "thumbnail_prompt": "P",
        })

        draft = generate_draft("Test", channel_context="esports news channel")
        # Verify the channel context was passed to the LLM
        call_args = mock_llm.call_args[0][0]
        assert "esports news channel" in call_args

    @patch("verticals.draft.research_topic")
    @patch("verticals.draft.call_llm")
    def test_truncates_broll_prompts(self, mock_llm, mock_research):
        mock_research.return_value = "research"
        mock_llm.return_value = json.dumps({
            "script": "s",
            "broll_prompts": [f"p{i}" for i in range(1, 11)],  # too many
            "youtube_title": "T", "youtube_description": "D",
            "youtube_tags": "t", "instagram_caption": "C",
            "thumbnail_prompt": "P",
        })

        draft = generate_draft("Test")
        assert len(draft["broll_prompts"]) == 8  # truncated to 8


class TestHookAndLength:
    @patch("verticals.draft.research_topic")
    @patch("verticals.draft.call_llm")
    def test_prompt_requires_punchy_hook_and_length(self, mock_llm, mock_research):
        mock_research.return_value = "research"
        mock_llm.return_value = json.dumps({"script": "s", "broll_prompts": ["p"]})
        generate_draft("Test")
        prompt = mock_llm.call_args[0][0]
        assert "HOOK (first ~1.5 seconds)" in prompt
        assert "45 and 75 seconds" in prompt

    @patch("verticals.draft.research_topic")
    @patch("verticals.draft.call_llm")
    def test_short_script_gets_one_rewrite(self, mock_llm, mock_research):
        mock_research.return_value = "research"
        short = json.dumps({"script": " ".join(["word"] * 80), "broll_prompts": ["p"]})
        good = json.dumps({"script": " ".join(["word"] * 140), "broll_prompts": ["p"]})
        mock_llm.side_effect = [short, good]
        draft = generate_draft("Test")
        assert len(draft["script"].split()) == 140
        assert "LENGTH CORRECTION" in mock_llm.call_args[0][0]

    @patch("verticals.draft.research_topic")
    @patch("verticals.draft.call_llm")
    def test_failed_rewrite_keeps_original(self, mock_llm, mock_research):
        mock_research.return_value = "research"
        long = json.dumps({"script": " ".join(["word"] * 250), "broll_prompts": ["p"]})
        mock_llm.side_effect = [long, RuntimeError("quota")]
        draft = generate_draft("Test")
        assert len(draft["script"].split()) == 250

    def test_length_bounds(self):
        from verticals.draft import script_length_problem
        assert script_length_problem("word " * 140) == ""
        assert script_length_problem("word " * 90) == "short"
        assert script_length_problem("word " * 200) == "long"
        assert script_length_problem("tiny stub") == ""
