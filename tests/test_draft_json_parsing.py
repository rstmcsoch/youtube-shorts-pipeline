import json

import pytest

from verticals import draft


def test_extracts_json_object_from_fenced_response():
    payload = {"script": "AI tools say \"hello\".", "broll_prompts": ["frame"]}
    raw = "Here is the draft:\n```json\n" + json.dumps(payload) + "\n```\n"
    assert draft._extract_json_object(raw) == payload


def test_extracts_balanced_object_with_braces_inside_string():
    payload = {"script": "Show a code sample: {x: 1}", "broll_prompts": []}
    assert draft._extract_json_object("prefix " + json.dumps(payload) + " trailing") == payload


def test_rejects_truncated_json():
    with pytest.raises(json.JSONDecodeError):
        draft._extract_json_object('{"script": "truncated"')


def test_retries_once_after_invalid_json(monkeypatch):
    responses = iter(['{"script": "cut off"', '{"script": "complete", "broll_prompts": []}'])
    calls = []

    def fake_call_llm(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return next(responses)

    monkeypatch.setattr(draft, "call_llm", fake_call_llm)
    result = draft._generate_json_response("original prompt", "gemini")

    assert result["script"] == "complete"
    assert len(calls) == 2
    assert calls[0][1]["max_tokens"] == 8192
    assert calls[0][1]["response_mime_type"] == "application/json"
    assert calls[0][1]["timeout"] == 120
    assert "RETRY REQUIREMENT" in calls[1][0]


def test_fails_cleanly_after_second_invalid_response(monkeypatch):
    monkeypatch.setattr(draft, "call_llm", lambda *args, **kwargs: '{"script": "still cut off"')
    with pytest.raises(ValueError, match="invalid JSON after one retry"):
        draft._generate_json_response("original prompt", "gemini")
