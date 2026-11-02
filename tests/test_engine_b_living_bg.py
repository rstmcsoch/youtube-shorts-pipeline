"""Owner feedback (2 Nov): dark living backgrounds, flowing elements, no long text-only runs."""

import pytest
from PIL import Image

from verticals import motion
from verticals.backdrop import Backdrop, mean_luma
from verticals.beats import normalize_beats, text_run_ok
from verticals.icons import ICONS, icon_sprite, pick_icon
from verticals.motion import render_frame

from tests.test_engine_b_piece4 import BLOCKS


def _diff(a: Image.Image, b: Image.Image) -> float:
    sa = list(a.convert("L").resize((54, 96), Image.BOX).getdata())
    sb = list(b.convert("L").resize((54, 96), Image.BOX).getdata())
    return sum(abs(x - y) for x, y in zip(sa, sb)) / len(sa)


@pytest.fixture()
def white_still(tmp_path):
    p = tmp_path / "white.png"
    Image.new("RGB", (640, 960), (255, 255, 255)).save(p)
    return p


@pytest.mark.parametrize("kind", sorted(BLOCKS))
@pytest.mark.parametrize("bg", ["none", "procedural", "white_footage"])
def test_background_mean_luma_below_0_3(kind, bg, white_still):
    """Never a white background: slate template, procedural layer, even a pure-white still as footage."""
    backdrop = {"none": None, "procedural": Backdrop(None, seed=1, duration=6.0),
                "white_footage": Backdrop(white_still, seed=2, duration=6.0)}[bg]
    try:
        motion._LOCAL.bg = backdrop.frame(2.0) if backdrop else None
        img = render_frame(kind, BLOCKS[kind], 2.0, 6.0)
    finally:
        motion._LOCAL.bg = None
    assert mean_luma(img) < 0.3
    if backdrop:
        assert mean_luma(backdrop.frame(2.0)) < 0.3


@pytest.mark.parametrize("source", ["procedural", "footage"])
def test_background_is_alive(source, white_still):
    pattern = white_still.with_name("pattern.png")
    img = Image.new("RGB", (640, 960))
    img.putdata([((x * 7) % 256, (y * 3) % 256, ((x + y) * 5) % 256) for y in range(960) for x in range(640)])
    img.save(pattern)
    bd = Backdrop(pattern if source == "footage" else None, seed=3, duration=8.0)
    assert bd.kind == source
    assert _diff(bd.frame(1.0), bd.frame(3.0)) > 0.3


@pytest.mark.parametrize("kind", sorted(BLOCKS))
def test_blocks_keep_moving_after_they_land(kind):
    """Not a pop-in then static: late in a long block, frames 0.5 s apart still differ."""
    a = render_frame(kind, BLOCKS[kind], 6.0, 8.0)
    b = render_frame(kind, BLOCKS[kind], 6.5, 8.0)
    assert _diff(a, b) > 0.05


def test_icons_render_and_are_picked_by_topic():
    for name in ICONS:
        spr = icon_sprite(name, 96, (76, 201, 240))
        assert spr.size == (96, 96) and spr.getchannel("A").getextrema()[1] > 0
    assert pick_icon("unlock your phone") == "phone"
    assert pick_icon(["credit card fraud"]) == "card"
    assert pick_icon("plain words here") == "chip"


def _b(kind, kw):
    data = {"kinetic_text": {"words": kw[:2]}, "code_panel": {"lines": ["x = 1"], "language": "python"},
            "diagram": {"nodes": ["A", "B"]}}[kind]
    return {"block": kind, "keywords": kw, "data": data}


def test_at_most_two_text_dominant_blocks_in_a_row():
    raw = [_b("diagram", ["face", "scan"]), _b("kinetic_text", ["depth", "map"]),
           _b("code_panel", ["match", "score"]), _b("kinetic_text", ["neural", "net", "chip"]),
           _b("code_panel", ["unlock", "phone"]), _b("diagram", ["end", "card"])]
    beats, warns = normalize_beats(raw, n_segments=6)
    assert text_run_ok(beats)
    assert any("text-dominant" in w for w in warns)


def test_motion_beats_get_footage_backdrops(tmp_path):
    from verticals.assemble import backdrop_sources
    stills = []
    for i in range(3):
        p = tmp_path / f"broll_{i}.png"
        Image.new("RGB", (64, 64), (40, 40, 40)).save(p)
        stills.append(p)
    sources = [{"kind": "motion", "block": "kinetic_text", "segment": 0},
               {"kind": "motion", "block": "diagram", "segment": 1},
               {"kind": "motion", "block": "count_up", "segment": 2},
               {"kind": "still", "segment": 1, "path": stills[1], "provider": "pixabay", "backup": True},
               {"kind": "still", "segment": 2, "path": stills[2], "provider": "gradient", "backup": True}]
    plan = [{"source": 0, "duration": 4}, {"source": 1, "duration": 4}, {"source": 2, "duration": 4}]
    out = backdrop_sources(plan, sources)
    assert out[1] == stills[1] and 2 not in out  # segment 1 gets its own still; gradients are never footage
    sources.append({"kind": "still", "segment": 0, "path": stills[0], "provider": "cloudflare_workers_ai",
                    "backup": True})
    out = backdrop_sources(plan, sources)
    assert out[0] == stills[0] and out[1] == stills[1]  # the hook gets footage too


def test_gradient_only_stock_beat_becomes_a_motion_card(tmp_path):
    import json
    from verticals.assemble import build_sources
    for i in range(2):
        (tmp_path / f"broll_{i}.png").write_bytes(b"x")
    (tmp_path / "broll_manifest.json").write_text(json.dumps({"assets": [
        {"frame": "broll_0.png", "segment_index": 0, "provider": "pixabay"},
        {"frame": "broll_1.png", "segment_index": 1, "provider": "gradient"}]}))
    beats = [{"block": "stock_or_ai_still", "keywords": ["phone"]},
             {"block": "stock_or_ai_still", "keywords": ["sensor", "blocked"]}]
    srcs = build_sources([tmp_path / "broll_0.png", tmp_path / "broll_1.png"], tmp_path, beats)
    motion = [s for s in srcs if s["kind"] == "motion"]
    assert [(m["block"], m["segment"]) for m in motion] == [("diagram", 1)]
    assert all(s.get("backup") for s in srcs if s["kind"] == "still" and s["segment"] == 1)


def test_year_count_up_has_no_thousands_comma():
    from verticals.motion import _fmt_number, _is_year
    assert _is_year({"value": 1895.0}) and _fmt_number(1895.0, 0, True) == "1895"
    assert not _is_year({"value": 1895.0, "suffix": "M"})


def test_gradient_still_is_only_a_backup(tmp_path):
    import json
    from verticals.assemble import build_sources, shot_count
    (tmp_path / "broll_0.png").write_bytes(b"x")
    (tmp_path / "clip_0.mp4").write_bytes(b"x")
    (tmp_path / "broll_manifest.json").write_text(json.dumps({"assets": [
        {"frame": "broll_0.png", "segment_index": 0, "provider": "gradient",
         "clip": {"file": "clip_0.mp4", "provider": "pixabay_video", "duration_seconds": 6}}]}))
    srcs = build_sources([tmp_path / "broll_0.png"], tmp_path, [{"block": "stock_or_ai_still", "keywords": ["x"]}])
    assert [s["kind"] for s in srcs if not s.get("backup")] == ["clip"]
    assert shot_count(srcs, 6.0) == 1
