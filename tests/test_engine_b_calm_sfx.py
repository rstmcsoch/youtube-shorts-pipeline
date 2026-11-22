"""Calm SFX: no noise sweeps, loudness-leveled 24 LU under the voice, DYPOL_SFX=off switch."""

import json
import re
import subprocess
import wave
from pathlib import Path

import pytest

from verticals import sfx

np = pytest.importorskip("numpy")
REPO = Path(__file__).resolve().parents[1]
BANNED = "who" + "osh"


def _speechy_voice(path: Path, seconds: float = 8.0, level: float = 0.3) -> Path:
    """Synthetic voice: harmonic 'vowels' with a ~4 Hz syllable envelope plus breathy noise."""
    rng = np.random.default_rng(3)
    t = np.arange(int(seconds * sfx.SR)) / sfx.SR
    f0 = 140 + 25 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sfx.SR
    x = sum((1.0 / k) * np.sin(k * phase) for k in range(1, 12))
    x += 0.15 * rng.standard_normal(t.size)
    env = np.clip(np.sin(2 * np.pi * 4.0 * t), 0, None) ** 0.6
    env *= (np.sin(2 * np.pi * 0.25 * t) > -0.7)          # pauses between phrases
    x = x * env
    x = level * x / np.max(np.abs(x))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sfx.SR)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    return path


def _ebur128_lufs(path: Path) -> float:
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    return float(re.findall(r"I:\s+(-?[\d.]+) LUFS", err)[-1])


@pytest.fixture(autouse=True)
def _sfx_on(monkeypatch):
    monkeypatch.delenv("DYPOL_SFX", raising=False)
    monkeypatch.delenv("DYPOL_SFX_ENABLED", raising=False)


# ───────────── palette ─────────────

def test_no_banned_names_reachable():
    assert BANNED not in (REPO / "verticals" / "sfx.py").read_text().lower()
    clips = sfx.load_manifest()["clips"]
    assert {c["category"] for c in clips} <= {"chime", "swell"}
    assert all(BANNED not in json.dumps(c).lower() for c in clips)
    assert BANNED not in sfx.TRANSITION_GAIN and BANNED not in sfx.clips_by_category()
    for i in (1, 2, 3):
        with pytest.raises(KeyError):
            sfx._clip(f"{BANNED}_0{i}")
    with pytest.raises(ValueError):
        sfx.synth_clip({"id": "x", "duration_s": 0.5, "peak_s": 0.1, "synth": {"kind": BANNED, "seed": 1}})
    for seed in range(25):
        beats = [round(2.5 * k + 0.3 * (seed % 3), 2) for k in range(1, 24)]
        ev = sfx.plan_sfx(60.0, beats, sub_cuts=[b + 1.1 for b in beats], hook_t=0.2,
                          riser_at=beats[4], seed=seed)
        assert ev and all(BANNED not in e["clip"] for e in ev)
        assert all(e["clip"].split("_")[0] in {"chime", "swell"} for e in ev)


def test_chimes_are_soft_low_and_slow_attack():
    for c in sfx.load_manifest()["clips"]:
        x = sfx.synth_clip(c)
        assert np.argmax(np.abs(x)) / sfx.SR >= sfx.MIN_ATTACK_S          # attack >= 20 ms
        if c["category"] == "chime":
            assert c["synth"]["f0"] <= 1500.0
            spec = np.abs(np.fft.rfft(x))
            assert np.fft.rfftfreq(x.size, 1 / sfx.SR)[np.argmax(spec)] <= 1500.0


def test_at_most_one_effect_per_section():
    beats = [3.0, 4.0, 4.5, 9.0, 15.0, 15.2, 22.0, 30.0, 41.0, 50.0]
    subs = [float(t) / 2 for t in range(1, 120)]
    ev = sfx.plan_sfx(60.0, beats, sub_cuts=subs, hook_t=0.1, riser_at=22.0)
    bounds = [0.0] + beats
    secs = [sfx._section_of(e["at"], bounds) for e in ev]
    assert len(secs) == len(set(secs))
    assert sum(e["kind"] == "riser" for e in ev) == 1


# ───────────── loudness leveling ─────────────

def test_sfx_lufs_gap_at_least_22_db_on_synthetic_voice(tmp_path):
    voice = _speechy_voice(tmp_path / "voice.wav")
    beats = [1.5, 3.5, 6.0]
    ev = sfx.plan_sfx(8.0, beats, hook_t=0.0, riser_at=6.0)
    ev = [dict(e, gain_db=0.0) for e in ev]                 # worst case: no relative trim
    out = sfx.render_sfx_track(ev, 8.0, voice, tmp_path / "sfx.wav")
    assert out is not None
    with wave.open(str(out)) as w:
        track = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2") / 32767
    v_lufs = sfx.integrated_lufs(sfx._decode(voice))
    assert sfx.integrated_lufs(track) <= v_lufs - 22.0
    # each effect individually
    for e in ev:
        clip = sfx._clip(e["clip"])
        i = int(e["t"] * sfx.SR)
        seg = track[i:i + int(clip["duration_s"] * sfx.SR)]
        assert v_lufs - sfx.integrated_lufs(seg) >= 22.0
    # cross-check with ffmpeg's independent BS.1770 meter
    assert _ebur128_lufs(voice) - _ebur128_lufs(out) >= 22.0


def test_level_is_loudness_not_peak(tmp_path):
    voice = _speechy_voice(tmp_path / "v.wav")
    ev = [{"t": 1.0, "at": 1.0, "clip": c, "kind": "transition", "gain_db": 0.0} for c in ("chime_01",)]
    ev2 = [{"t": 1.0, "at": 1.0, "clip": "swell_01", "kind": "transition", "gain_db": 0.0}]
    v_lufs = sfx.integrated_lufs(sfx._decode(voice))
    for events, name in ((ev, "a.wav"), (ev2, "b.wav")):
        out = sfx.render_sfx_track(events, 4.0, voice, tmp_path / name)
        with wave.open(str(out)) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2") / 32767
        assert abs((v_lufs - sfx.integrated_lufs(x)) - 24.0) < 0.6


def test_numpy_meter_matches_ffmpeg(tmp_path):
    voice = _speechy_voice(tmp_path / "v.wav")
    assert abs(sfx.integrated_lufs(sfx._decode(voice)) - _ebur128_lufs(voice)) < 1.0


# ───────────── off switch ─────────────

@pytest.mark.parametrize("env", [("DYPOL_SFX", "off"), ("DYPOL_SFX", "OFF"), ("DYPOL_SFX_ENABLED", "0"),
                                 ("DYPOL_SFX_ENABLED", "false")])
def test_off_switch_disables_everything(monkeypatch, tmp_path, env):
    voice = _speechy_voice(tmp_path / "v.wav", seconds=3.0)
    monkeypatch.setenv(*env)
    assert sfx.enabled() is False
    assert sfx.plan_sfx(30.0, [10.0, 20.0], hook_t=0.0, riser_at=20.0) == []
    ev = [{"t": 1.0, "at": 1.0, "clip": "chime_01", "kind": "transition", "gain_db": 0.0}]
    assert sfx.render_sfx_track(ev, 3.0, voice, tmp_path / "s.wav") is None
    assert not (tmp_path / "s.wav").exists()


def test_on_by_default_and_explicit_on(monkeypatch):
    assert sfx.enabled() is True
    monkeypatch.setenv("DYPOL_SFX", "on")
    assert sfx.enabled() is True and sfx.plan_sfx(30.0, [10.0, 20.0])
