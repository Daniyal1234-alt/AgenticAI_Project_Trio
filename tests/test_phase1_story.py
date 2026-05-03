"""Phase 1 — story/script/character generation tests (offline)."""

from phase1_story.agent import run_phase1
from schemas.pipeline import Story


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
