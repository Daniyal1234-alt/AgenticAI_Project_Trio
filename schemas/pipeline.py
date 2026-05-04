"""
Shared pipeline schema — the central JSON state object passed between phases.

Phase 1 produces: Story (story + scenes[] + characters[])
Phase 2 produces: TimingManifest (audio segments per scene/line) and writes audio files
Phase 3 produces: SceneVideo[] and final_output.mp4 path
Phase 4 reads/writes: ProjectState (the umbrella object)
Phase 5 mutates:    ProjectState across versions (StateManager snapshots)
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Phase 1 — Story / Script / Characters                                        #
# --------------------------------------------------------------------------- #


class Character(BaseModel):
    """A character in the story with a stable identity across scenes."""

    model_config = ConfigDict(extra="ignore")

    name: str
    role: str = Field(description="protagonist | antagonist | narrator | supporting")
    voice_style: str = Field(
        default="neutral",
        description="Hint for the TTS voice (warm / gravelly / whispered / cheerful etc.)",
    )
    voice_id: Optional[str] = Field(
        default=None,
        description="Resolved Edge-TTS voice locale (e.g. en-US-GuyNeural). Phase 2 fills this in.",
    )
    appearance: str = Field(default="", description="Short visual description — used by Phase 3.")
    personality: str = Field(default="")


class DialogueLine(BaseModel):
    """A single spoken line inside a scene."""

    model_config = ConfigDict(extra="ignore")

    character: str
    line: str
    direction: str = Field(default="", description="Stage direction, e.g. (whispered)")


class Scene(BaseModel):
    """A scene unit — what Phase 2 voices and Phase 3 visualises."""

    model_config = ConfigDict(extra="ignore")

    scene_number: int
    heading: str = Field(description="Like INT. SHIP'S BRIDGE — NIGHT")
    location: str = ""
    time_of_day: str = "DAY"
    mood: str = Field(default="neutral", description="One word: tense, hopeful, melancholy…")
    duration_seconds: float = Field(default=8.0, ge=2.0, le=60.0)
    characters: list[str] = Field(default_factory=list)
    action: str = ""
    visual_prompt: str = Field(
        default="",
        description="Prompt fed to the image generator in Phase 3.",
    )
    dialogue: list[DialogueLine] = Field(default_factory=list)


class Story(BaseModel):
    """Phase 1 output — the creative skeleton everything else hangs off."""

    model_config = ConfigDict(extra="ignore")

    title: str
    logline: str = ""
    style: str = Field(default="cinematic", description="cinematic / cartoon / noir / anime…")
    scenes: list[Scene] = Field(default_factory=list)
    characters: list[Character] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Phase 1 — internal handoff types between the three agents                    #
# --------------------------------------------------------------------------- #


class SceneOutline(BaseModel):
    """High-level beat — produced by the Story agent, consumed by the Script agent."""

    model_config = ConfigDict(extra="ignore")

    scene_number: int
    heading: str
    mood: str = "neutral"
    target_duration_s: float = Field(default=8.0, ge=2.0, le=60.0)
    summary: str = Field(default="", description="One-line description of what happens.")


class StoryArc(BaseModel):
    """Three-act skeleton — output of the Story agent's structural reasoning."""

    model_config = ConfigDict(extra="ignore")

    intro: str = ""
    rising_action: str = ""
    climax: str = ""
    resolution: str = ""


class StoryOutline(BaseModel):
    """Story agent output — title, themes, arc, scene beats. No characters yet."""

    model_config = ConfigDict(extra="ignore")

    title: str
    logline: str = ""
    style: str = "cinematic"
    themes: list[str] = Field(default_factory=list)
    arc: StoryArc = Field(default_factory=StoryArc)
    scene_outlines: list[SceneOutline] = Field(default_factory=list)


class CharacterRoster(BaseModel):
    """Character agent output — the cast for this story."""

    model_config = ConfigDict(extra="ignore")

    characters: list[Character] = Field(default_factory=list)


class ScriptOutput(BaseModel):
    """Script agent output — fully fleshed scenes (dialogue, visuals, timing)."""

    model_config = ConfigDict(extra="ignore")

    scenes: list[Scene] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Phase 2 — Audio                                                              #
# --------------------------------------------------------------------------- #


class AudioSegment(BaseModel):
    """One synthesised audio file inside a scene's timeline."""

    model_config = ConfigDict(extra="ignore")

    scene_id: int
    kind: Literal["dialogue", "bgm"]
    character: Optional[str] = None
    audio_file: str = Field(description="Path relative to project version folder.")
    start_ms: int
    end_ms: int
    text: str = ""


class TimingManifest(BaseModel):
    """Phase 2 output — assembly instructions for Phase 3."""

    model_config = ConfigDict(extra="ignore")

    segments: list[AudioSegment] = Field(default_factory=list)
    total_duration_ms: int = 0
    bgm_track: Optional[str] = Field(default=None, description="Path to the master BGM file, if any.")


# --------------------------------------------------------------------------- #
# Phase 3 — Video                                                              #
# --------------------------------------------------------------------------- #


class SceneVideo(BaseModel):
    """Per-scene visual asset metadata."""

    model_config = ConfigDict(extra="ignore")

    scene_id: int
    image_file: str
    duration_seconds: float
    animation: Literal["zoom_in", "zoom_out", "pan_left", "pan_right", "static"] = "zoom_in"


# --------------------------------------------------------------------------- #
# Umbrella project state                                                       #
# --------------------------------------------------------------------------- #


PipelinePhase = Literal["created", "phase1", "phase2", "phase3", "complete", "edited"]


class ProjectState(BaseModel):
    """
    The single source of truth that flows between phases and is snapshotted
    by the StateManager (Phase 5) on every change.
    """

    model_config = ConfigDict(extra="ignore")

    project_id: str
    version: int = 1
    user_prompt: str = ""
    style_hint: str = "cinematic"
    phase: PipelinePhase = "created"
    story: Optional[Story] = None
    timing: Optional[TimingManifest] = None
    scene_videos: list[SceneVideo] = Field(default_factory=list)
    final_video: Optional[str] = Field(
        default=None,
        description="Path (relative to project folder) of final_output.mp4",
    )
    edit_log: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
