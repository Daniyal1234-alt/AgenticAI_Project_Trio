"""Phase 3 — image generation + filters + composition (offline)."""

import os

import pytest
from PIL import Image

from phase1_story.agent import run_phase1
from phase2_audio.pipeline import run_phase2, realign_scene_durations
from phase3_video.image_gen import apply_filter, generate_scene_image
from phase3_video.pipeline import run_phase3


def test_generate_scene_image_writes_png(tmp_path):
    story, _ = run_phase1(prompt="A cat in a library", num_scenes=2)
    out = tmp_path / "scene.png"
    path = generate_scene_image(story.scenes[0], story, str(out))
    assert os.path.isfile(path)
    img = Image.open(path)
    assert img.size == (1280, 720)


def test_apply_filter_changes_pixels(tmp_path):
    story, _ = run_phase1(prompt="A cat in a library", num_scenes=2)
    src = tmp_path / "src.png"
    generate_scene_image(story.scenes[0], story, str(src))
    dst = tmp_path / "dst.png"
    apply_filter(str(src), "darken", str(dst))

    a = list(Image.open(src).convert("RGB").getdata())
    b = list(Image.open(dst).convert("RGB").getdata())
    # darken should reduce mean brightness across all pixels
    avg_a = sum(sum(p) for p in a) / (len(a) * 3)
    avg_b = sum(sum(p) for p in b) / (len(b) * 3)
    assert avg_b < avg_a


@pytest.mark.asyncio
async def test_run_phase3_produces_final_video(tmp_path):
    story, _ = run_phase1(prompt="A cat in a library", num_scenes=2)
    manifest = await run_phase2(story, str(tmp_path))
    story = realign_scene_durations(story, manifest)
    result = await run_phase3(story, manifest, str(tmp_path))
    assert "scene_videos" in result
    assert len(result["scene_videos"]) == 2
    final = os.path.join(str(tmp_path), result["final_video"])
    # final_output.mp4 is always created (might be empty if neither MoviePy nor ffmpeg
    # is on the test machine — but the file must exist as a placeholder).
    assert os.path.exists(final)
