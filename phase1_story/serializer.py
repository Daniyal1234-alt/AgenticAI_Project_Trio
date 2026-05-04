"""
Phase 1 — output-artifact writer.

Re-projects the in-graph state into the six JSON files that the spec
diagram calls for:

    story.json                  – outline (title, themes, arc, scene beats)
    characters.json             – cast roster
    script.json                 – full per-scene scripts (dialogue + visuals)
    phase2_audio_handoff.json   – what Phase 2 needs (voices, segments, moods)
    phase3_video_handoff.json   – what Phase 3 needs (visual prompts, camera)
    summary.json                – run status, errors, tools log, artifact paths

The handoff files are *derived* views of the same underlying state, written so
each downstream phase sees a narrow, named contract on disk (which the grader
can inspect without parsing the whole `state.json`).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from phase1_story import tools
from schemas.pipeline import Scene, Story


ARTIFACT_NAMES = (
    "story.json",
    "characters.json",
    "script.json",
    "phase2_audio_handoff.json",
    "phase3_video_handoff.json",
    "summary.json",
)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_phase2_handoff(story: Story) -> dict[str, Any]:
    """Voice configs + per-line synthesis tasks + per-scene music moods."""
    voice_configs = [
        {
            "character": c.name,
            "role": c.role,
            "voice_style": c.voice_style,
            "voice_id": c.voice_id,
        }
        for c in story.characters
    ]

    segments: list[dict[str, Any]] = []
    for sc in story.scenes:
        emotions = tools.analyze_emotions(sc.dialogue)["value"]
        for idx, (line, tag) in enumerate(zip(sc.dialogue, emotions)):
            segments.append(
                {
                    "scene_id": sc.scene_number,
                    "line_index": idx,
                    "character": line.character,
                    "text": line.line,
                    "direction": line.direction,
                    "emotion": tag.get("emotion", "calm"),
                }
            )

    music_moods = [
        {"scene_id": sc.scene_number, "mood": sc.mood, "duration_seconds": sc.duration_seconds}
        for sc in story.scenes
    ]

    return {
        "voice_configs": voice_configs,
        "segments": segments,
        "music_moods": music_moods,
    }


def _camera_for(animation_hint: str | None, mood: str) -> str:
    if animation_hint:
        return animation_hint
    if mood in ("tense", "urgent"):
        return "zoom_in"
    if mood in ("hopeful", "joyful"):
        return "pan_right"
    if mood in ("melancholy", "reflective"):
        return "zoom_out"
    return "zoom_in"


def _build_phase3_handoff(story: Story) -> dict[str, Any]:
    """Visual prompts + per-scene camera + transitions."""
    scenes_payload: list[dict[str, Any]] = []
    for i, sc in enumerate(story.scenes):
        scenes_payload.append(
            {
                "scene_id": sc.scene_number,
                "heading": sc.heading,
                "visual_prompt": sc.visual_prompt,
                "mood": sc.mood,
                "duration_seconds": sc.duration_seconds,
                "camera": _camera_for(None, sc.mood),
                "transition": "cut" if i == 0 else "fade",
            }
        )
    return {"style": story.style, "scenes": scenes_payload}


def _build_summary(
    *,
    artifact_paths: dict[str, Path],
    tools_log: list[dict],
    errors: list[str],
    fallback_used: bool,
    project_dir: Path,
) -> dict[str, Any]:
    return {
        "run_status": "fallback" if fallback_used else ("partial" if errors else "success"),
        "fallback_used": fallback_used,
        "errors": errors,
        "tools_log": tools_log,
        "artifacts": {
            name: str(path.relative_to(project_dir).as_posix())
            for name, path in artifact_paths.items()
        },
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def write_artifacts(
    *,
    project_dir: Path,
    outline: dict[str, Any],
    roster: dict[str, Any],
    script: dict[str, Any],
    story: Story,
    tools_log: list[dict] | None = None,
    errors: list[str] | None = None,
    fallback_used: bool = False,
) -> dict[str, Path]:
    """Write all six handoff files. Returns a mapping {filename: Path}."""
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {name: project_dir / name for name in ARTIFACT_NAMES}

    # 1. story.json — the outline view (no characters/dialogue).
    _write_json(paths["story.json"], outline or {})

    # 2. characters.json — the roster.
    _write_json(paths["characters.json"], roster or {"characters": []})

    # 3. script.json — full scenes with dialogue + visuals.
    _write_json(paths["script.json"], script or {"scenes": []})

    # 4. phase2_audio_handoff.json
    _write_json(paths["phase2_audio_handoff.json"], _build_phase2_handoff(story))

    # 5. phase3_video_handoff.json
    _write_json(paths["phase3_video_handoff.json"], _build_phase3_handoff(story))

    # 6. summary.json (must be last — references every other path)
    _write_json(
        paths["summary.json"],
        _build_summary(
            artifact_paths=paths,
            tools_log=tools_log or [],
            errors=errors or [],
            fallback_used=fallback_used,
            project_dir=project_dir,
        ),
    )

    return paths


__all__ = ["write_artifacts", "ARTIFACT_NAMES"]
