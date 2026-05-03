"""Edit-agent schemas — what the intent classifier returns and how versions are recorded."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


EditTarget = Literal["audio", "video_frame", "video", "script"]


class EditIntent(BaseModel):
    """
    Structured output from the LangGraph edit-intent classifier.

    The free-text query "make scene 2 darker" becomes:
        EditIntent(intent="recolor_scene", target="video_frame",
                   scope="scene:2", parameters={"aesthetic": "darker"})
    """

    model_config = ConfigDict(extra="ignore")

    intent: str = Field(description="Snake-case action label.")
    target: EditTarget
    scope: str = Field(
        default="all",
        description='Selector — "scene:2", "character:Narrator", "all"…',
    )
    parameters: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    explanation: str = Field(default="")


class VersionEntry(BaseModel):
    """A single snapshot in the project's version history."""

    model_config = ConfigDict(extra="ignore")

    version: int
    created_at: str
    summary: str
    snapshot_dir: str
    edit_intent: Optional[EditIntent] = None
