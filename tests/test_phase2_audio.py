"""Phase 2 — audio generation (TTS fallback path + BGM + manifest)."""

import json
import os
import tempfile

import pytest

from phase1_story.agent import run_phase1
from phase2_audio.bgm import pick_bgm
from phase2_audio.pipeline import realign_scene_durations, run_phase2
from phase2_audio.tts import estimate_ms, prosody_for, synthesize, voice_for
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


def test_prosody_for_returns_distinct_rates_per_emotion():
    assert prosody_for("urgent")["rate"] != prosody_for("calm")["rate"]
    assert prosody_for("melancholy")["pitch"] != prosody_for("joyful")["pitch"]
    # Unknown emotions fall through to the no-op default
    assert prosody_for("alien") == {"rate": "+0%", "pitch": "+0Hz"}
    # Empty / None inputs are also neutralised
    assert prosody_for("") == {"rate": "+0%", "pitch": "+0Hz"}


@pytest.mark.asyncio
async def test_synthesize_accepts_prosody_kwargs(tmp_path):
    out = tmp_path / "line.mp3"
    ms = await synthesize(
        "Hello, world.", "en-US-AriaNeural", str(out),
        rate="+10%", pitch="-5Hz",
    )
    assert ms > 0
    # File exists either as MP3 (network ok) or .wav fallback (offline).
    assert out.exists() or (tmp_path / "line.wav").exists()


@pytest.mark.asyncio
async def test_run_phase2_writes_timing_manifest_json(tmp_path):
    story, _ = run_phase1(prompt="A diver explores a wreck", num_scenes=2)
    await run_phase2(story, str(tmp_path))
    manifest_file = tmp_path / "timing_manifest.json"
    assert manifest_file.exists(), "timing_manifest.json must be written to disk"
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert data["segments"], "manifest segments[] must be populated"
    required = {"scene_id", "audio_file", "start_ms", "end_ms"}
    for seg in data["segments"]:
        assert required <= seg.keys(), f"missing fields in segment: {required - seg.keys()}"
    assert data["total_duration_ms"] > 0


@pytest.mark.asyncio
async def test_run_phase2_uses_handoff_emotions_when_available(tmp_path):
    # Run Phase 1 *with* project_dir so phase2_audio_handoff.json is written.
    story, _ = run_phase1(
        prompt="A scientist hears a warning in the data",
        num_scenes=2,
        project_dir=tmp_path,
    )
    handoff = tmp_path / "phase2_audio_handoff.json"
    assert handoff.exists(), "Phase 1 should have written the handoff"
    manifest = await run_phase2(story, str(tmp_path))
    # Existing invariants must still hold even with handoff loaded.
    assert manifest.total_duration_ms > 0
    assert any(seg.kind == "bgm" for seg in manifest.segments)
    assert any(seg.kind == "dialogue" for seg in manifest.segments)
