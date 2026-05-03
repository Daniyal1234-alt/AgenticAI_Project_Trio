"""
Pipeline orchestrator — wires Phases 1 → 2 → 3 together for the web layer.

The CLI uses this directly; the FastAPI app uses the streaming variant
that emits per-phase progress messages over a WebSocket.

Public:
    PROJECTS_ROOT                  → outputs/projects (single shared root)
    new_project_id()               → fresh ULID-ish id
    await generate_full_pipeline(prompt, *, project_id, num_scenes, style,
                                 progress) -> ProjectState
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import uuid
from typing import Awaitable, Callable, Optional

from phase1_story.agent import run_phase1
from phase2_audio.pipeline import realign_scene_durations, run_phase2
from phase3_video.pipeline import run_phase3
from phase5_edit.state_manager import StateManager
from schemas.pipeline import ProjectState


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECTS_ROOT = os.path.join(HERE, "outputs", "projects")
os.makedirs(PROJECTS_ROOT, exist_ok=True)


ProgressCb = Optional[Callable[[str], Awaitable[None] | None]]


async def _emit(cb: ProgressCb, msg: str) -> None:
    if cb is None:
        return
    res = cb(msg)
    if asyncio.iscoroutine(res):
        await res


def new_project_id() -> str:
    """ULID-ish project id: timestamp-prefixed for sortability."""
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


async def generate_full_pipeline(
    prompt: str,
    *,
    project_id: Optional[str] = None,
    num_scenes: int = 3,
    style: str = "cinematic",
    progress: ProgressCb = None,
) -> ProjectState:
    """Run all three creative phases and snapshot v1."""
    project_id = project_id or new_project_id()
    sm = StateManager(PROJECTS_ROOT, project_id)
    project_dir = sm.current_dir

    state = ProjectState(
        project_id=project_id,
        version=1,
        user_prompt=prompt,
        style_hint=style,
        phase="created",
    )
    state.notes.append(f"Starting pipeline: {prompt[:80]}")
    sm.save_state(state)

    # ── Phase 1 ────────────────────────────────────────────────────
    await _emit(progress, "──── PHASE 1: Story & Script ────")
    story, log = run_phase1(prompt=prompt, num_scenes=num_scenes, style=style)
    for entry in log:
        await _emit(progress, entry)
    state.story = story
    state.phase = "phase1"
    sm.save_state(state)

    # ── Phase 2 ────────────────────────────────────────────────────
    await _emit(progress, "──── PHASE 2: Audio ────")
    manifest = await run_phase2(story, project_dir, progress=progress)
    state.timing = manifest
    state.story = realign_scene_durations(story, manifest)
    state.phase = "phase2"
    sm.save_state(state)

    # ── Phase 3 ────────────────────────────────────────────────────
    await _emit(progress, "──── PHASE 3: Video ────")
    result = await run_phase3(
        state.story, manifest, project_dir,
        regenerate=False, subtitles=True, speed=1.0,
        progress=progress,
    )
    state.scene_videos = result["scene_videos"]
    state.final_video = result["final_video"]
    state.phase = "complete"
    sm.save_state(state)

    # Snapshot v1.
    sm.snapshot(summary="Initial generation")
    await _emit(progress, "──── DONE — v1 snapshotted ────")
    return state
