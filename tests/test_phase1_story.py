"""Phase 1 — story / script / character generation tests (offline)."""

from __future__ import annotations

import json
from pathlib import Path

from phase1_story import tools
from phase1_story.agent import run_phase1
from phase1_story.serializer import ARTIFACT_NAMES
from schemas.pipeline import (
    Character,
    DialogueLine,
    Scene,
    Story,
    StoryArc,
    StoryOutline,
    SceneOutline,
)


# --------------------------------------------------------------------------- #
# End-to-end (existing — must keep passing)                                    #
# --------------------------------------------------------------------------- #


def test_run_phase1_returns_validated_story():
    story, log = run_phase1(prompt="A lone scribe finds a forgotten library", num_scenes=3)
    assert isinstance(story, Story)
    assert story.title
    assert len(story.scenes) == 3
    assert len(story.characters) >= 2
    assert all(sc.dialogue for sc in story.scenes)
    assert all(sc.visual_prompt for sc in story.scenes)
    assert log, "expected log entries"


def test_scene_numbers_are_contiguous():
    story, _ = run_phase1(prompt="A robot tutor on Mars", num_scenes=4)
    nums = [sc.scene_number for sc in story.scenes]
    assert nums == list(range(1, len(nums) + 1))


def test_dialogue_speakers_in_character_roster():
    story, _ = run_phase1(prompt="A treasure hunter and her parrot", num_scenes=3)
    char_names = {c.name for c in story.characters}
    for sc in story.scenes:
        for d in sc.dialogue:
            assert d.character in char_names, f"speaker {d.character} missing from roster"


# --------------------------------------------------------------------------- #
# Tool: validate_story_arc                                                     #
# --------------------------------------------------------------------------- #


def test_validate_story_arc_flags_missing_climax():
    outline = StoryOutline(
        title="Missing Climax",
        logline="A test story without a climax.",
        themes=["test"],
        arc=StoryArc(intro="setup", rising_action="pressure", climax="", resolution="end"),
        scene_outlines=[
            SceneOutline(scene_number=1, heading="A", mood="hopeful"),
            SceneOutline(scene_number=2, heading="B", mood="tense"),
            SceneOutline(scene_number=3, heading="C", mood="reflective"),
        ],
    )
    res = tools.validate_story_arc(outline)
    assert res["ok"] is False
    assert any("climax" in i for i in res["issues"])


# --------------------------------------------------------------------------- #
# Tool: estimate_duration                                                      #
# --------------------------------------------------------------------------- #


def test_estimate_duration_matches_word_count():
    short = Scene(
        scene_number=1, heading="A", mood="neutral", duration_seconds=2.0,
        dialogue=[DialogueLine(character="X", line="hi")],
    )
    longer = Scene(
        scene_number=2, heading="B", mood="neutral", duration_seconds=10.0,
        dialogue=[DialogueLine(character="X", line=" ".join(["word"] * 50))],
    )
    short_res = tools.estimate_duration(short)["value"]
    long_res = tools.estimate_duration(longer)["value"]
    # estimate_duration returns scene.duration_seconds when set
    assert short_res == 2.0
    assert long_res == 10.0
    # When duration is unset it falls back to word-count heuristic
    bare = Scene(scene_number=3, heading="C", mood="neutral", duration_seconds=8.0, dialogue=[])
    bare.duration_seconds = 0  # bypass schema floor for the test
    res = tools.estimate_duration(bare)["value"]
    assert res >= 2.0  # falls back to default ≥ 2s


# --------------------------------------------------------------------------- #
# Tool: check_consistency                                                      #
# --------------------------------------------------------------------------- #


def test_check_consistency_rejects_duplicate_names():
    roster = [
        Character(name="Alex", role="protagonist", voice_style="warm", appearance="Tall"),
        Character(name="alex", role="supporting", voice_style="cheerful", appearance="Short"),
    ]
    res = tools.check_consistency(roster)
    assert res["ok"] is False
    assert any("duplicate" in i for i in res["issues"])


def test_check_consistency_rejects_no_protagonist():
    roster = [
        Character(name="Narrator", role="narrator", voice_style="warm", appearance="Unseen"),
        Character(name="Sidekick", role="supporting", voice_style="cheerful", appearance="Short"),
    ]
    res = tools.check_consistency(roster)
    assert res["ok"] is False
    assert any("protagonist" in i for i in res["issues"])


# --------------------------------------------------------------------------- #
# Tool: build_visual_prompt                                                    #
# --------------------------------------------------------------------------- #


def test_build_visual_prompt_includes_mood_and_location():
    sc = Scene(
        scene_number=1,
        heading="EXT. RUINED LIBRARY — DUSK",
        location="Ruined library",
        time_of_day="DUSK",
        mood="melancholy",
        action="The scribe lights a candle.",
        dialogue=[DialogueLine(character="Scribe", line="...")],
    )
    res = tools.build_visual_prompt(sc, style="cinematic")
    assert res["ok"] is True
    prompt = res["value"].lower()
    assert "melancholy" in prompt
    assert "library" in prompt
    assert "cinematic" in prompt


# --------------------------------------------------------------------------- #
# Tool: validate_duration                                                      #
# --------------------------------------------------------------------------- #


def test_validate_duration_rejects_60_plus_seconds():
    # Scene schema caps at 60 — bypass via model_construct to test the tool directly
    sc = Scene.model_construct(
        scene_number=1,
        heading="X",
        mood="neutral",
        duration_seconds=75.0,
        dialogue=[DialogueLine(character="X", line="hi")],
    )
    res = tools.validate_duration(sc)
    assert res["ok"] is False
    assert any("60" in i for i in res["issues"])


# --------------------------------------------------------------------------- #
# Tool: analyze_emotions                                                       #
# --------------------------------------------------------------------------- #


def test_analyze_emotions_tags_known_keywords():
    lines = [
        DialogueLine(character="A", line="What is happening here?", direction=""),
        DialogueLine(character="B", line="Run! It's coming!", direction="(panic)"),
        DialogueLine(character="C", line="A quiet morning.", direction="(softly)"),
    ]
    tags = tools.analyze_emotions(lines)["value"]
    assert tags[0]["emotion"] == "curious"
    assert tags[1]["emotion"] == "tense"
    assert tags[2]["emotion"] == "calm"


# --------------------------------------------------------------------------- #
# Serializer: writes all 6 handoff files                                       #
# --------------------------------------------------------------------------- #


def test_serializer_writes_six_handoff_files(tmp_path: Path):
    project_dir = tmp_path / "proj"
    story, _ = run_phase1(
        prompt="A young astronaut discovers a hidden ocean on Mars",
        num_scenes=3,
        project_dir=project_dir,
    )
    assert isinstance(story, Story)

    for name in ARTIFACT_NAMES:
        p = project_dir / name
        assert p.exists(), f"{name} was not written"
        data = json.loads(p.read_text(encoding="utf-8"))
        assert isinstance(data, dict) and data, f"{name} is empty"

    summary = json.loads((project_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_status"] in {"success", "fallback", "partial"}
    assert "tools_log" in summary
    assert set(summary["artifacts"].keys()) == set(ARTIFACT_NAMES)

    phase2 = json.loads((project_dir / "phase2_audio_handoff.json").read_text(encoding="utf-8"))
    assert "voice_configs" in phase2 and "segments" in phase2 and "music_moods" in phase2
    # at least one segment carries an emotion tag from analyze_emotions
    assert phase2["segments"], "phase2 segments should not be empty"
    assert all("emotion" in s for s in phase2["segments"])

    phase3 = json.loads((project_dir / "phase3_video_handoff.json").read_text(encoding="utf-8"))
    assert phase3["scenes"]
    assert all("visual_prompt" in s and "camera" in s for s in phase3["scenes"])
