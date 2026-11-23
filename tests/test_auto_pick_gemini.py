"""Blank-topic runs must not need Claude (23 Nov 2026)."""
from verticals.topics.base import TopicCandidate
from verticals.topics import engine as eng


def _cands():
    return [TopicCandidate(title="How phones sort photos by face", source="rss", trending_score=0.9),
            TopicCandidate(title="Other topic", source="rss", trending_score=0.5)]


def test_auto_pick_uses_call_llm(monkeypatch):
    import verticals.llm as llm
    monkeypatch.setattr(llm, "call_llm", lambda prompt, max_tokens=200, **k: '"How spam filters learn"\n')
    te = eng.TopicEngine.__new__(eng.TopicEngine)
    assert te.auto_pick(_cands()) == "How spam filters learn"


def test_auto_pick_falls_back_to_top_candidate(monkeypatch):
    import verticals.llm as llm
    def boom(*a, **k):
        raise RuntimeError("no key")
    monkeypatch.setattr(llm, "call_llm", boom)
    te = eng.TopicEngine.__new__(eng.TopicEngine)
    assert te.auto_pick(_cands()) == "How phones sort photos by face"


def test_no_claude_imports():
    src = open(eng.__file__).read()
    assert "get_claude_backend" not in src
