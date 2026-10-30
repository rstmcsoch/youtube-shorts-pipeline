"""Speed changes in produce: parallel voiceover + b-roll, tail-silence padding."""
import json
import sys
import threading
import types
import wave
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from verticals import tts


# ---------------------------------------------------------------- tail padding

def test_short_padding_seconds():
    assert tts.short_padding_seconds(44.9) == pytest.approx(0.35)
    assert tts.short_padding_seconds(43.6) == pytest.approx(1.65)
    assert tts.short_padding_seconds(43.4) == 0.0      # too far below: re-render instead
    assert tts.short_padding_seconds(45.0) == 0.0
    assert tts.short_padding_seconds(60.0) == 0.0
    assert tts.short_padding_seconds(0.0) == 0.0


class _FakeKokoro:
    calls = []
    seconds = 44.9

    def __init__(self, *_a, **_k):
        pass

    def create(self, script, voice, speed, lang):
        _FakeKokoro.calls.append(speed)
        rate = 1000
        return np.full(int(_FakeKokoro.seconds * rate), 0.1, dtype="float32"), rate


@pytest.fixture
def fake_kokoro(monkeypatch, tmp_path):
    _FakeKokoro.calls = []
    monkeypatch.setitem(sys.modules, "kokoro_onnx", types.SimpleNamespace(Kokoro=_FakeKokoro))
    monkeypatch.setattr(tts, "ensure_kokoro_files", lambda *_a, **_k: (tmp_path / "m", tmp_path / "v"))
    return _FakeKokoro


def _wav_seconds(path):
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def test_kokoro_just_short_is_padded_not_rerendered(fake_kokoro, tmp_path):
    fake_kokoro.seconds = 44.9
    script = " ".join(["word"] * 110)
    out = tts._generate_kokoro(script, tmp_path, "en")
    assert len(fake_kokoro.calls) == 1
    assert _wav_seconds(out) >= 45.0


def test_kokoro_far_short_still_rerenders(fake_kokoro, tmp_path):
    fake_kokoro.seconds = 43.0
    script = " ".join(["word"] * 100)
    tts._generate_kokoro(script, tmp_path, "en")
    assert len(fake_kokoro.calls) == 2
    assert fake_kokoro.calls[1] < fake_kokoro.calls[0]


# ---------------------------------------------------------------- parallel produce

def _produce_env(monkeypatch, tmp_path, record):
    import verticals.__main__ as cli
    import verticals.broll as broll
    import verticals.captions as captions
    import verticals.music as music
    import verticals.assemble as assemble

    media = tmp_path / "media"
    monkeypatch.setattr(cli, "MEDIA_DIR", media)
    started = threading.Event()

    def fake_broll(prompts, work_dir, **_k):
        record["broll_thread"] = threading.current_thread().name
        # Voiceover must already be running while b-roll is gathered.
        record["overlap"] = started.wait(timeout=5)
        (Path(work_dir) / "broll_manifest.json").write_text("{}", encoding="utf-8")
        return [Path(work_dir) / "f0.png"]

    def fake_vo(script, work_dir, lang, provider=None, voice_config=None):
        record["vo_thread"] = threading.current_thread().name
        started.set()
        path = Path(work_dir) / f"voiceover_{lang}.wav"
        path.write_bytes(b"x")
        return path

    monkeypatch.setattr(broll, "generate_broll", fake_broll)
    monkeypatch.setattr(broll, "top_up_ai_frames", lambda *a, **k: False)
    monkeypatch.setattr(broll, "judge_contact_sheet", lambda *a, **k: None)
    monkeypatch.setattr(tts, "generate_voiceover", fake_vo)
    monkeypatch.setattr(assemble, "get_audio_duration", lambda p: 50.0)
    monkeypatch.setattr(captions, "generate_captions", lambda *a, **k: {"srt_path": "", "ass_path": ""})
    monkeypatch.setattr(music, "select_and_prepare_music", lambda *a, **k: {})
    monkeypatch.setattr(assemble, "assemble_video",
                        lambda **k: Path(k["out_dir"]) / "video.mp4")
    draft_path = tmp_path / "d.json"
    draft_path.write_text(json.dumps({"job_id": "j1", "script": "hello world", "niche": "tech",
                                      "broll_prompts": ["a"], "review_required": True}), encoding="utf-8")
    return cli, Namespace(draft=str(draft_path), lang="en", voice="kokoro", script=None, force=False)


def test_voiceover_runs_in_parallel_with_broll(monkeypatch, tmp_path):
    record = {}
    cli, args = _produce_env(monkeypatch, tmp_path, record)
    monkeypatch.setattr(cli, "_apply_visual_manifest", lambda *a, **k: None)
    video = cli.cmd_produce(args)
    assert video.name == "video.mp4"
    assert record["overlap"] is True
    assert record["vo_thread"].startswith("voiceover")
    assert record["broll_thread"] == threading.current_thread().name


def test_parallel_produce_can_be_disabled(monkeypatch, tmp_path):
    record = {}
    monkeypatch.setenv("DYPOL_PARALLEL_PRODUCE", "0")
    cli, args = _produce_env(monkeypatch, tmp_path, record)
    monkeypatch.setattr(cli, "_apply_visual_manifest", lambda *a, **k: None)
    cli.cmd_produce(args)
    assert record["overlap"] is False          # voiceover only starts after b-roll
    assert record["vo_thread"] == threading.current_thread().name


# ---------------------------------------------------------------- parallel shot encodes

def test_assemble_renders_shots_concurrently_and_keeps_order(monkeypatch, tmp_path):
    import time as _time
    import verticals.assemble as assemble

    monkeypatch.setenv("DYPOL_ASSEMBLE_WORKERS", "4")
    monkeypatch.setattr(assemble, "MEDIA_DIR", tmp_path)
    monkeypatch.setattr(assemble, "get_audio_duration", lambda p: 16.0)
    frames = []
    for i in range(4):
        f = tmp_path / f"broll_{i}.png"
        f.write_bytes(b"png")
        frames.append(f)
    active, peak = [0], [0]
    lock = threading.Lock()

    def fake_animate(img, out, duration, effect, variant=0):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        _time.sleep(0.2 if "0" in Path(img).name else 0.05)   # first shot finishes last
        Path(out).write_bytes(b"mp4")
        with lock:
            active[0] -= 1

    commands = []
    monkeypatch.setattr(assemble, "animate_frame", fake_animate)
    monkeypatch.setattr(assemble, "prepare_audio", lambda *a, **k: tmp_path / "audio_final.wav")
    monkeypatch.setattr(assemble, "run_cmd", lambda cmd, **k: commands.append(cmd))
    out = assemble.assemble_video(frames, tmp_path / "vo.wav", tmp_path, "j1")
    assert out.name == "verticals_j1_en.mp4"
    assert peak[0] > 1
    lines = (tmp_path / "concat.txt").read_text().splitlines()
    assert lines == [f"file '{tmp_path / f'anim_{i}.mp4'}'" for i in range(len(lines))]


def test_assemble_workers_env(monkeypatch):
    import verticals.assemble as assemble
    monkeypatch.setenv("DYPOL_ASSEMBLE_WORKERS", "2")
    assert assemble.assemble_workers() == 2
    monkeypatch.setenv("DYPOL_ASSEMBLE_WORKERS", "junk")
    assert 1 <= assemble.assemble_workers() <= 4
