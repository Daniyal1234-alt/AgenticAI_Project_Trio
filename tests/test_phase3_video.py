"""Phase 3 — image generation + filters + composition (offline)."""

import os
from pathlib import Path

import pytest
from PIL import Image

from phase1_story.agent import run_phase1
from phase2_audio.pipeline import realign_scene_durations, run_phase2
from phase3_video import animation, lipsync, svd
from phase3_video.image_gen import (
    apply_filter,
    generate_scene_image,
    generate_speaker_image,
)
from phase3_video.pipeline import run_phase3


def test_generate_scene_image_writes_png(tmp_path):
    story, _ = run_phase1(prompt="A cat in a library", num_scenes=2)
    out = tmp_path / "scene.png"
    path = generate_scene_image(story.scenes[0], story, str(out))
    assert os.path.isfile(path)
    img = Image.open(path)
    assert img.size == (1280, 720)


def test_generate_speaker_image_writes_png(tmp_path):
    story, _ = run_phase1(prompt="A scribe in a forgotten archive", num_scenes=2)
    out = tmp_path / "speaker.png"
    path = generate_speaker_image(story.scenes[0], story.characters[0].name, story, str(out))
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
    avg_a = sum(sum(p) for p in a) / (len(a) * 3)
    avg_b = sum(sum(p) for p in b) / (len(b) * 3)
    assert avg_b < avg_a


def test_ken_burns_produces_clip(tmp_path):
    story, _ = run_phase1(prompt="A robot tutor", num_scenes=2)
    img = tmp_path / "still.png"
    generate_scene_image(story.scenes[0], story, str(img))
    out = tmp_path / "kb.mp4"
    animation.ken_burns(str(img), 2.0, str(out), motion="zoom_in")
    # Either real MP4 (ffmpeg present) or empty placeholder (no ffmpeg available).
    assert out.exists()
    if animation.ffmpeg_exe():
        assert out.stat().st_size > 1000


def test_svd_passthrough_produces_clip(tmp_path, monkeypatch):
    monkeypatch.delenv("KAGGLE_ENDPOINT", raising=False)
    story, _ = run_phase1(prompt="A diver explores a wreck", num_scenes=2)
    img = tmp_path / "still.png"
    generate_scene_image(story.scenes[0], story, str(img))
    out = tmp_path / "svd.mp4"
    svd.generate_clip(str(img), 2.0, str(out))
    assert out.exists()
    if animation.ffmpeg_exe():
        assert out.stat().st_size > 1000


@pytest.mark.asyncio
async def test_lipsync_passthrough_produces_clip(tmp_path, monkeypatch):
    monkeypatch.delenv("KAGGLE_ENDPOINT", raising=False)
    story, _ = run_phase1(prompt="A submarine crew faces a storm", num_scenes=2)
    manifest = await run_phase2(story, str(tmp_path))
    # Find the first dialogue audio segment.
    audio_seg = next(s for s in manifest.segments if s.kind == "dialogue")
    audio_path = tmp_path / audio_seg.audio_file

    speaker_img = tmp_path / "speaker.png"
    generate_speaker_image(story.scenes[0], audio_seg.character or "Narrator", story, str(speaker_img))

    out = tmp_path / "lip.mp4"
    lipsync.lipsync_line(str(speaker_img), str(audio_path), str(out))
    assert out.exists()
    if animation.ffmpeg_exe():
        assert out.stat().st_size > 1000


@pytest.mark.asyncio
async def test_run_phase3_produces_final_video(tmp_path):
    story, _ = run_phase1(prompt="A cat in a library", num_scenes=2)
    manifest = await run_phase2(story, str(tmp_path))
    story = realign_scene_durations(story, manifest)
    result = await run_phase3(story, manifest, str(tmp_path))
    assert "scene_videos" in result
    assert len(result["scene_videos"]) == 2
    final = os.path.join(str(tmp_path), result["final_video"])
    assert os.path.exists(final)


@pytest.mark.asyncio
async def test_run_phase3_creates_per_line_clips(tmp_path):
    story, _ = run_phase1(prompt="A small town in winter", num_scenes=2)
    manifest = await run_phase2(story, str(tmp_path))
    story = realign_scene_durations(story, manifest)
    await run_phase3(story, manifest, str(tmp_path))
    # First scene should have a per-line clip on disk.
    clips_dir = Path(tmp_path) / "clips"
    assert clips_dir.exists()
    line_clips = list(clips_dir.glob("scene01_line*.mp4"))
    assert line_clips, "expected at least one per-line clip for scene 1"
