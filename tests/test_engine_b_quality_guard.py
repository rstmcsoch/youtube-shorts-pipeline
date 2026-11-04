"""Strong-model script, topic relevance, hook specifics and hard QC failures (Engine B, 4 Nov 2026).

Draft 6 (3 Nov): Gemini 3.8/3.6 returned 503, flash-lite wrote a history lesson about earbuds with no AI in it,
and every check only warned. Draft 7's hook ("Your phone automatically analyzes visual features...") had no
number. Now: weak models never write the script, an off-topic script is retried then fails as "script model
unavailable", the hook must carry a number or a surprising specific (one rewrite), and hard QC failures mark
the draft "needs fixes".
"""
import json

import pytest

from verticals import draft as dr
from verticals import llm
from verticals import topic_guard as tg
from verticals.__main__ import mark_quality
from verticals.beats import hook_specific, hook_warnings
from verticals.qc import hard_failures

EARBUDS = (
    "Your earbuds generate sound waves to cancel background noise 50 times a second. Active noise control works "
    "by adding a second sound designed specifically to cancel unwanted environmental sound. The concept dates "
    "back to the late 1930s. Later developmental work that began in the 1950s eventually resulted in aviation "
    "headsets. Modern earbuds use microphones, amplifiers, and digital signal processing to detect surrounding "
    "sound. The newer iteration incorporates the H2 chip and Bluetooth 5.3 connectivity.")
EARBUDS_AI = (EARBUDS + " Inside, a small neural network model was trained on thousands of hours of noise; the AI "
              "predicts the next sound wave and learns your ear's seal.")
PHOTOS_HOOK = "Your phone automatically analyzes visual features across your entire photo library."


# ------------------------------------------------------------------ hook: number or surprising specific

def test_hook_needs_a_number_or_a_surprising_specific():
    assert hook_specific(PHOTOS_HOOK + " Then it groups faces.") == ""
    assert any("no concrete number or surprising specific" in w for w in hook_warnings(PHOTOS_HOOK))
    assert hook_specific("Your earbuds flip sound waves 50,000 times a second.") == "number"
    assert hook_specific("Your phone sorts 3 billion faces a day.") == "number"
    assert hook_specific("An AI can spot a broken wrist faster than most doctors.") == "specific"
    assert hook_specific("Your earbuds react every millisecond.") == "specific"
    # a number later in the script does not rescue a vague hook
    assert hook_specific(PHOTOS_HOOK + " It checks 128 points.") == ""


def _script(first, words=130):
    body = ("The model maps each face to a vector, compares vectors, and clusters close ones together. " * 20).split()
    return first + " " + " ".join(body[: words - len(first.split())])


def test_vague_hook_gets_one_rewrite_with_the_specific_rule(monkeypatch):
    prompts = []
    better = _script("Google Photos sorts 4 billion photos a day by face.")

    def fake(prompt, provider):
        prompts.append(prompt)
        return {"script": better}

    monkeypatch.setattr(dr, "_generate_json_response", fake)
    out = dr._enforce_hook({"script": _script(PHOTOS_HOOK)}, "PROMPT", "gemini",
                           research="Google Photos sorts 4 billion photos a day [source: Google 2024]")
    assert out["script"] == better and len(prompts) == 1
    assert "MUST contain a concrete number" in prompts[0] and "surprising concrete specific" in prompts[0]


def test_hook_rule_is_in_the_prompt():
    import inspect
    src = inspect.getsource(dr.generate_draft)
    assert "CONTAINS a real number from the research written in digits" in src


# ------------------------------------------------------------------ topic relevance

def test_the_ai_free_earbuds_script_is_off_topic():
    assert tg.relevance("How noise-cancelling earbuds use AI", EARBUDS)["verdict"] == tg.OFF_TOPIC
    assert tg.relevance("How noise-cancelling earbuds use AI", EARBUDS_AI)["verdict"] == tg.ON_TOPIC


def test_borderline_asks_the_strong_model(monkeypatch):
    monkeypatch.setenv("DYPOL_TOPIC_CHECK", "1")
    asked = []
    monkeypatch.setattr(dr, "call_llm", lambda prompt, **k: asked.append(prompt) or '{"on_topic": false, "reason": "x"}')
    r = dr.script_relevance("How AI writes code from a prompt quickly",
                            "The AI model reads your prompt. The model predicts each token of code, trained on code.",
                            "gemini")
    assert asked and r["verdict"] == tg.OFF_TOPIC and r["method"] == "keywords+model"


# ------------------------------------------------------------------ strong models only

class _R:
    def __init__(self, status, text=""):
        self.status_code = status
        self.text = text

    def json(self):
        if self.status_code == 200:
            return {"candidates": [{"content": {"parts": [{"text": self.text}]}}]}
        return {"error": {"message": "This model is currently experiencing high demand"}}


@pytest.fixture
def gemini(monkeypatch):
    import requests
    import time
    calls = []

    def post(url, json=None, timeout=None, headers=None):
        model = url.split("/models/")[1].split(":")[0]
        calls.append(model)
        return _R(200, "weak answer") if model.endswith("lite") else _R(503)

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setattr(llm, "get_gemini_key", lambda: "k")
    monkeypatch.setenv("GEMINI_TEXT_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-3.6-flash,gemini-3.5-flash-lite")
    return calls


def test_strong_only_skips_the_lite_model(gemini):
    with llm.strong_models_only():
        with pytest.raises(llm.LLMUnavailableError):
            llm._call_gemini("p", 100)
    assert gemini == ["gemini-3.8-flash", "gemini-3.6-flash"]
    gemini.clear()
    assert llm._call_gemini("p", 100) == "weak answer"          # other tasks may still use it
    assert llm.LAST_MODEL == "gemini-3.5-flash-lite" and not llm.strong_only_active()


@pytest.fixture
def clock(monkeypatch):
    now = {"t": 0.0}
    monkeypatch.setattr(dr, "_clock", lambda: now["t"])
    monkeypatch.setattr(dr, "_sleep", lambda s: now.__setitem__("t", now["t"] + s + 45))  # an attempt ~45 s
    monkeypatch.setenv("DYPOL_TOPIC_CHECK", "1")
    monkeypatch.delenv("DYPOL_SCRIPT_BUDGET_S", raising=False)
    return now


def test_script_fails_clearly_when_strong_models_stay_down(clock):
    tries = []

    def one_draft(extra):
        tries.append(llm.strong_only_active())
        raise llm.LLMUnavailableError("Gemini API 503")

    with pytest.raises(dr.ScriptModelUnavailableError) as info:
        dr._strong_on_topic_draft(one_draft, "How noise-cancelling earbuds use AI", "gemini")
    assert str(info.value).startswith("script model unavailable")
    assert getattr(info.value, "failure_category") == "llm_unavailable"
    assert all(tries) and len(tries) >= 4 and 400 <= clock["t"] <= 600


def test_off_topic_script_is_retried_with_a_correction(clock, monkeypatch):
    monkeypatch.setattr(llm, "LAST_MODEL", "gemini-3.6-flash")
    extras = []
    scripts = [EARBUDS, EARBUDS_AI]

    def one_draft(extra):
        extras.append(extra)
        return {"script": scripts[len(extras) - 1]}

    out = dr._strong_on_topic_draft(one_draft, "How noise-cancelling earbuds use AI", "gemini")
    assert out["script"] == EARBUDS_AI and out["script_check"]["attempts"] == 2
    assert out["script_check"]["model"] == "gemini-3.6-flash"
    assert extras[0] == "" and "TOPIC CORRECTION" in extras[1]


def test_three_off_topic_scripts_fail_the_draft(clock):
    calls = []
    with pytest.raises(dr.ScriptModelUnavailableError) as info:
        dr._strong_on_topic_draft(lambda extra: calls.append(1) or {"script": EARBUDS},
                                  "How noise-cancelling earbuds use AI", "gemini")
    assert len(calls) == 3 and "off-topic" in str(info.value)


# ------------------------------------------------------------------ hard QC failures -> needs fixes

def test_run_qc_reports_a_low_resolution_render(tmp_path):
    import subprocess
    from verticals import qc
    video = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=216x384:r=30:d=2",
                    "-f", "lavfi", "-i", "sine=f=440:d=2", "-shortest", "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", str(video)], check=True)
    rep = qc.run_qc(video, tmp_path / "qc.json", tmp_path / "s.png")
    assert (rep["width"], rep["height"]) == (216, 384)
    assert rep["verdict"] == "needs_fixes" and rep["status"] == "warn"
    assert "resolution 216x384" in rep["hard_failures"][0]


def test_hard_failures():
    assert hard_failures({"width": 1080, "height": 1920, "freezes": []}) == []
    assert hard_failures({"width": 720, "height": 1280, "freezes": [{"start": 1.0, "duration": 3.4}]}) == []
    assert "resolution 1920x1080" in hard_failures({"width": 1920, "height": 1080})[0]
    assert "resolution 540x960" in hard_failures({"width": 540, "height": 960})[0]
    assert "static for 4.0 s" in hard_failures({"freezes": [{"start": 2.0, "duration": 4.0}]})[0]
    assert hard_failures({}, relevance={"verdict": "off_topic", "reasons": ["no AI"]})[0].startswith("off-topic")


def test_mark_quality_sets_needs_fixes(tmp_path):
    qc_json = tmp_path / "qc.json"
    qc_json.write_text(json.dumps({"width": 1080, "height": 1920,
                                   "freezes": [{"start": 20.0, "duration": 5.2}]}), encoding="utf-8")
    d = {"review_required": False, "script_check": {"model": "gemini-3.8-flash",
                                                    "relevance": {"verdict": "on_topic"}}}
    q = mark_quality(d, qc_json, "en")
    assert d["needs_fixes"] is True and d["review_required"] is True and q["verdict"] == "needs_fixes"
    assert q["hard_failures"] == ["en: picture static for 5.2 s at 20.00 s (limit 3.5 s)"]
    assert any(r.startswith("Needs fixes:") for r in d["review_reasons"])
    ok = {"script_check": {"relevance": {"verdict": "on_topic"}}}
    qc_json.write_text(json.dumps({"width": 1080, "height": 1920, "freezes": []}), encoding="utf-8")
    assert mark_quality(ok, qc_json)["verdict"] == "ok" and ok["needs_fixes"] is False


def test_handoff_record_and_auto_publish_guard_in_workflow():
    from pathlib import Path
    wf = (Path(__file__).resolve().parents[1] / ".github/workflows/verticals.yml").read_text(encoding="utf-8")
    assert '"needs_fixes": bool(draft.get("needs_fixes", False))' in wf
    assert "Block auto-publish of a draft that needs fixes" in wf
