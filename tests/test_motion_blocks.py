"""Motion renderers: render a 2 s sample of each block and check it with ffprobe."""

import json
import shutil
import subprocess

import pytest

from verticals.motion import MOTION_KINDS, render_block, render_frame
from verticals.style import color, font_path, load_style, STYLE_DIR

SAMPLES = {
    "kinetic_text": {"words": ["10x", "cheaper", "overnight"], "accent": "cheaper"},
    "count_up": {"value": 94.5, "decimals": 1, "suffix": "%", "label": "accuracy on the benchmark",
                 "source": "Source: vendor announcement, Nov 2026"},
    "code_panel": {"language": "bash", "title": "terminal",
                   "lines": ["$ pip install fastmodel", "Successfully installed", "$ fastmodel run", "latency: 42 ms"]},
    "diagram": {"title": "How it works", "nodes": ["Your prompt", "Router", "Small model", "Answer"]},
}

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                                  reason="ffmpeg/ffprobe not installed")


def _probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,pix_fmt,codec_name:format=duration",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    return data["streams"][0], float(data["format"]["duration"])


@needs_ffmpeg
@pytest.mark.parametrize("kind", MOTION_KINDS)
def test_render_two_second_sample(kind, tmp_path):
    out = render_block(kind, SAMPLES[kind], tmp_path / f"{kind}.mp4", 2.0)
    stream, duration = _probe(out)
    assert abs(duration - 2.0) < 0.1
    assert (stream["width"], stream["height"]) == (1080, 1920)
    assert stream["r_frame_rate"] == "30/1"
    assert stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"


@pytest.mark.parametrize("kind", MOTION_KINDS)
def test_frames_move_and_stay_in_safe_area(kind):
    first = render_frame(kind, SAMPLES[kind], 0.2, 4.0)
    later = render_frame(kind, SAMPLES[kind], 1.5, 4.0)
    end = render_frame(kind, SAMPLES[kind], 3.9, 4.0)
    assert first.size == (1080, 1920)
    assert first.tobytes() != later.tobytes() != end.tobytes()  # never static
    bg = color("background")
    # Nothing drawn in the Shorts UI zones: right 14% and bottom 22% (diagram grid excepted).
    if kind != "diagram":
        right = end.crop((int(1080 * 0.86) + 2, 0, 1080, 1920))
        bottom = end.crop((0, int(1920 * 0.78), 1080, 1920))
        assert right.getcolors(1) == [(right.width * right.height, bg)]
        assert bottom.getcolors(1) == [(bottom.width * bottom.height, bg)]


def test_long_words_shrink_to_fit():
    data = {"words": ["Interoperability", "everywhere"], "accent": "everywhere"}
    img = render_frame("kinetic_text", data, 3.0, 3.5)
    right = img.crop((940, 0, 1080, 1920))
    assert right.getcolors(1) == [(right.width * right.height, color("background"))]


def test_style_fonts_and_licences_are_vendored():
    look = load_style()["look"]
    for role in look["fonts"]:
        assert font_path(role).exists()
    for lic in look["licences"]:
        assert (STYLE_DIR / lic).exists()
        assert "Open Font License" in (STYLE_DIR / lic).read_text(encoding="utf-8")
