"""
Edit executor — turns an EditIntent into actual changes on disk.

For each (target, intent) we know which phases need to re-run on what scope:
    audio/change_voice_tone    → Phase 2 for affected scenes
    audio/change_bgm           → Phase 2 (BGM only) for affected scenes
    video_frame/apply_filter   → apply PIL filter to affected scene images, recomposite
    video_frame/regenerate_*   → re-render affected images, recomposite
    video/adjust_speed         → recomposite with speed factor
    video/{add,remove}_subtitle→ recomposite with subtitles flag
    script/regenerate_script   → Phase 1 → 2 → 3 from scratch (cascades)

`apply_edit(state, intent, ...)` is the one entry the web/API uses.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
from typing import Awaitable, Callable, Optional

from phase1_story.agent import run_phase1
from phase2_audio.pipeline import realign_scene_durations, run_phase2
from phase3_video.image_gen import apply_filter
from phase3_video.pipeline import run_phase3
from schemas.edit import EditIntent
from schemas.pipeline import ProjectState, Story, TimingManifest


ProgressCb = Optional[Callable[[str], Awaitable[None] | None]]


async def _emit(cb: ProgressCb, msg: str) -> None:
    if cb is None:
        return
    res = cb(msg)
    if asyncio.iscoroutine(res):
        await res


def _scope_to_scene_ids(intent: EditIntent, story: Story) -> set[int]:
    """Resolve `scene:N` / `character:Name` / `all` into a set of scene IDs."""
    scope = intent.scope or "all"
    if scope == "all":
        return {sc.scene_number for sc in story.scenes}
    if scope.startswith("scene:"):
        try:
            return {int(scope.split(":", 1)[1])}
        except ValueError:
            return set()
    if scope.startswith("character:"):
        name = scope.split(":", 1)[1].strip().lower()
        return {sc.scene_number for sc in story.scenes if any(c.lower() == name for c in sc.characters)}
    return {sc.scene_number for sc in story.scenes}


def _apply_voice_change(intent: EditIntent, story: Story) -> None:
    """Mutate character voice_style (Phase 2 will re-resolve voice_id)."""
    scope = intent.scope or "all"
    new_tone = intent.parameters.get("tone") or "warm"
    if scope.startswith("character:"):
        target = scope.split(":", 1)[1].strip().lower()
        for c in story.characters:
            if c.name.lower() == target:
                c.voice_style = new_tone
                c.voice_id = None  # force re-resolve
    else:
        for c in story.characters:
            c.voice_style = new_tone
            c.voice_id = None


async def apply_edit(
    state: ProjectState,
    intent: EditIntent,
    project_dir: str,
    *,
    progress: ProgressCb = None,
) -> ProjectState:
    """
    Mutate the live project (`current/` directory) according to the intent.

    Caller is responsible for snapshotting before calling this and saving state
    afterwards. We never touch history — that's StateManager's job.
    """
    target = intent.target

    # ----- script (cascade everything) -----
    if target == "script":
        await _emit(progress, "[Edit] Regenerating story (Phase 1 → 2 → 3)")
        prompt = state.user_prompt or (state.story.logline if state.story else "")
        num_scenes = len(state.story.scenes) if state.story else 3
        new_story, _ = run_phase1(prompt=prompt, num_scenes=num_scenes, style=state.style_hint)
        state.story = new_story
        manifest = await run_phase2(new_story, project_dir, progress=progress)
        state.timing = manifest
        new_story = realign_scene_durations(new_story, manifest)
        state.story = new_story
        result = await run_phase3(
            new_story, manifest, project_dir,
            regenerate=True, subtitles=True, speed=1.0,
            progress=progress,
        )
        state.scene_videos = result["scene_videos"]
        state.final_video = result["final_video"]
        state.phase = "complete"
        return state

    if state.story is None:
        raise RuntimeError("Cannot edit a project that has no story yet")

    affected = _scope_to_scene_ids(intent, state.story)

    # ----- audio edits -----
    if target == "audio":
        if intent.intent in ("change_voice_tone",):
            _apply_voice_change(intent, state.story)
        # For BGM edits we just reshape mood (or keep) and re-run Phase 2.
        elif intent.intent == "change_bgm":
            mood = intent.parameters.get("mood")
            if mood:
                for sc in state.story.scenes:
                    if sc.scene_number in affected:
                        sc.mood = mood
        await _emit(progress, "[Edit] Re-running Phase 2 (audio)")
        manifest = await run_phase2(state.story, project_dir, progress=progress)
        state.timing = manifest
        state.story = realign_scene_durations(state.story, manifest)
        # Recomposite to pick up new audio. scope_filter=set() → reuse images.
        await _emit(progress, "[Edit] Recompositing video")
        result = await run_phase3(
            state.story, manifest, project_dir,
            regenerate=False, scope_filter=set(),
            subtitles=True, speed=1.0,
            progress=progress,
        )
        state.scene_videos = result["scene_videos"]
        state.final_video = result["final_video"]
        state.phase = "edited"
        return state

    # ----- video_frame edits -----
    if target == "video_frame":
        manifest = state.timing or TimingManifest()
        if intent.intent == "apply_filter":
            filt = intent.parameters.get("filter", "darken")
            await _emit(progress, f"[Edit] Applying filter '{filt}' to scenes {sorted(affected)}")
            images_dir = os.path.join(project_dir, "images")
            for sid in sorted(affected):
                path = os.path.join(images_dir, f"scene{sid:02d}.png")
                if os.path.isfile(path):
                    await asyncio.to_thread(apply_filter, path, filt, path)
            await _emit(progress, "[Edit] Recompositing")
            # scope_filter=set() means "regenerate nothing — reuse all images on disk"
            result = await run_phase3(
                state.story, manifest, project_dir,
                regenerate=False, scope_filter=set(),
                subtitles=True, speed=1.0,
                progress=progress,
            )
            state.scene_videos = result["scene_videos"]
            state.final_video = result["final_video"]
            state.phase = "edited"
            return state

        # regenerate_character_frames / regenerate_scene_image
        await _emit(progress, f"[Edit] Re-rendering scenes {sorted(affected)}")
        result = await run_phase3(
            state.story, manifest, project_dir,
            regenerate=True, scope_filter=affected,
            subtitles=True, speed=1.0,
            progress=progress,
        )
        state.scene_videos = result["scene_videos"]
        state.final_video = result["final_video"]
        state.phase = "edited"
        return state

    # ----- video (recomposition only) -----
    if target == "video":
        manifest = state.timing or TimingManifest()
        speed = float(intent.parameters.get("speed", 1.0)) if intent.intent == "adjust_speed" else 1.0
        subtitles = intent.intent != "remove_subtitle"
        await _emit(
            progress,
            f"[Edit] Recompositing — speed={speed} subtitles={subtitles}",
        )
        # scope_filter=set() → reuse all images on disk, only recomposite the MP4.
        result = await run_phase3(
            state.story, manifest, project_dir,
            regenerate=False, scope_filter=set(),
            subtitles=subtitles, speed=speed,
            progress=progress,
        )
        state.scene_videos = result["scene_videos"]
        state.final_video = result["final_video"]
        state.phase = "edited"
        return state

    # Unknown target — no-op so the caller can still surface the issue.
    await _emit(progress, f"[Edit] Unknown target '{target}' — no changes applied")
    return state
