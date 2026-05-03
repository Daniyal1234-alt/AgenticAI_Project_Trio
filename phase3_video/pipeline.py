"""
Phase 3 orchestrator.

For each scene: generate (or load) an image, then composite all scenes +
manifest into a final MP4.

Public:
    await run_phase3(story, manifest, project_dir, *, regenerate=False,
                     subtitles=True, speed=1.0, scope_filter=None) -> dict
"""

from __future__ import annotations

import asyncio
import os
from typing import Awaitable, Callable, Optional

from phase3_video.compositor import compose
from phase3_video.image_gen import generate_scene_image
from schemas.pipeline import Scene, SceneVideo, Story, TimingManifest


ProgressCb = Optional[Callable[[str], Awaitable[None] | None]]


async def _emit(cb: ProgressCb, msg: str) -> None:
    if cb is None:
        return
    res = cb(msg)
    if asyncio.iscoroutine(res):
        await res


async def run_phase3(
    story: Story,
    manifest: TimingManifest,
    project_dir: str,
    *,
    regenerate: bool = False,
    subtitles: bool = True,
    speed: float = 1.0,
    scope_filter: Optional[set[int]] = None,
    progress: ProgressCb = None,
) -> dict:
    """
    Generate per-scene images and composite the final MP4.

    `scope_filter`: if provided, only those scene_ids will be re-rendered
    (used by Phase 5 partial re-runs). Existing images for other scenes are
    re-used.
    """
    images_dir = os.path.join(project_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    image_map: dict[int, str] = {}
    scene_videos: list[SceneVideo] = []

    for scene in story.scenes:
        img_path = os.path.join(images_dir, f"scene{scene.scene_number:02d}.png")
        in_scope = scope_filter is None or scene.scene_number in scope_filter
        if regenerate or in_scope or not os.path.isfile(img_path):
            await _emit(progress, f"[Phase3] Rendering scene {scene.scene_number} image")
            await asyncio.to_thread(generate_scene_image, scene, story, img_path)
        else:
            await _emit(progress, f"[Phase3] Reusing cached image for scene {scene.scene_number}")

        image_map[scene.scene_number] = img_path
        scene_videos.append(
            SceneVideo(
                scene_id=scene.scene_number,
                image_file=os.path.relpath(img_path, project_dir).replace("\\", "/"),
                duration_seconds=scene.duration_seconds,
                animation="zoom_in",
            )
        )

    await _emit(progress, "[Phase3] Compositing final video — this can take a moment")
    final_path = await asyncio.to_thread(
        compose, story, manifest, image_map, project_dir,
        subtitles=subtitles, speed=speed,
    )
    rel_final = os.path.relpath(final_path, project_dir).replace("\\", "/")
    await _emit(progress, f"[Phase3] Done — {rel_final}")

    return {
        "scene_videos": scene_videos,
        "final_video": rel_final,
    }
