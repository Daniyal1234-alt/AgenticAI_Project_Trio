"""Phase 2 — audio generation (TTS fallback path + BGM + manifest)."""

import os
import tempfile

import pytest

from phase1_story.agent import run_phase1
from phase2_audio.bgm import pick_bgm
from phase2_audio.pipeline import realign_scene_durations, run_phase2
from phase2_audio.tts import estimate_ms, voice_for
from schemas.pipeline import Character, TimingManifest


def test_voice_for_resolves_by_style_and_role():
    c = Character(name="N", role="narrator", voice_style="warm")
    assert voice_for(c).startswith("en-")
    c2 = Character(name="X", role="antagonist", voice_style="")
    assert voice_for(c2).startswith("en-")


def test_estimate_ms_monotonic_in_text_length():
    short = estimate_ms("Hello.")
    long = estimate_ms("This is a much longer sentence with many more words to speak.")
    assert long > short


def test_pick_bgm_generates_file_when_no_asset(tmp_path):
    out = tmp_path / "bgm.wav"
    result = pick_bgm("hopeful", 3.0, str(out))
    assert os.path.isfile(result)
    assert os.path.getsize(result) > 1000  # real WAV bytes


@pytest.mark.asyncio
async def test_run_phase2_produces_manifest(tmp_path):
    story, _ = run_phase1(prompt="A submarine crew faces a storm", num_scenes=2)
    manifest = await run_phase2(story, str(tmp_path))
    assert isinstance(manifest, TimingManifest)
    assert manifest.total_duration_ms > 0
    # Each scene contributes at least one BGM segment.
    for sc in story.scenes:
        scene_segs = [s for s in manifest.segments if s.scene_id == sc.scene_number]
        assert any(s.kind == "bgm" for s in scene_segs)
        # Audio files referenced exist on disk.
        for s in scene_segs:
            assert os.path.isfile(os.path.join(str(tmp_path), s.audio_file))


@pytest.mark.asyncio
async def test_realign_scene_durations_reflects_manifest(tmp_path):
    story, _ = run_phase1(prompt="A small town in winter", num_scenes=2)
    manifest = await run_phase2(story, str(tmp_path))
    realigned = realign_scene_durations(story, manifest)
    for sc in realigned.scenes:
        assert sc.duration_seconds >= 2.0
