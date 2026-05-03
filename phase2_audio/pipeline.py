"""
Phase 2 orchestrator.

Walks the Story scene-by-scene, line-by-line:
  - synthesises each dialogue line via edge-tts (per-character voice)
  - computes start_ms / end_ms per segment so Phase 3 can sync to the timeline
  - generates a per-scene BGM file and registers it as a scene-level segment
  - writes everything under {project_dir}/audio/

Returns a fully populated `TimingManifest`.
"""

from __future__ import annotations

import asyncio
import os
from typing import Awaitable, Callable, Optional

from phase2_audio.bgm import pick_bgm
from phase2_audio.tts import estimate_ms, synthesize, voice_for
from schemas.pipeline import AudioSegment, Story, TimingManifest


ProgressCb = Optional[Callable[[str], Awaitable[None] | None]]


async def _emit(cb: ProgressCb, msg: str) -> None:
    if cb is None:
        return
    res = cb(msg)
    if asyncio.iscoroutine(res):
        await res


async def run_phase2(
    story: Story,
    project_dir: str,
    *,
    progress: ProgressCb = None,
) -> TimingManifest:
    audio_dir = os.path.join(project_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    # Resolve voices for every character once.
    voices = {c.name: voice_for(c) for c in story.characters}
    for c in story.characters:
        c.voice_id = voices.get(c.name, "en-US-JennyNeural")
    await _emit(progress, f"[Phase2] Voices resolved: {voices}")

    segments: list[AudioSegment] = []
    cursor_ms = 0

    for scene in story.scenes:
        scene_start_ms = cursor_ms
        await _emit(progress, f"[Phase2] Scene {scene.scene_number} — synthesising dialogue")

        # Dialogue lines, sequential within a scene (so timing is monotonic).
        for idx, line in enumerate(scene.dialogue, start=1):
            voice = voices.get(line.character) or voice_for({"voice_style": "neutral", "role": ""})
            fname = f"scene{scene.scene_number:02d}_line{idx:02d}.mp3"
            fpath = os.path.join(audio_dir, fname)
            duration_ms = await synthesize(line.line, voice, fpath)
            # synthesize() may have written .wav fallback — pick what's actually there.
            written = fpath if os.path.exists(fpath) else os.path.splitext(fpath)[0] + ".wav"
            segments.append(
                AudioSegment(
                    scene_id=scene.scene_number,
                    kind="dialogue",
                    character=line.character,
                    audio_file=os.path.relpath(written, project_dir).replace("\\", "/"),
                    start_ms=cursor_ms,
                    end_ms=cursor_ms + duration_ms,
                    text=line.line,
                )
            )
            cursor_ms += duration_ms

        # Pad each scene up to its declared duration so visuals and audio stay in sync.
        scene_target_ms = int(scene.duration_seconds * 1000)
        scene_actual_ms = cursor_ms - scene_start_ms
        if scene_actual_ms < scene_target_ms:
            cursor_ms = scene_start_ms + scene_target_ms

        # Per-scene BGM.
        bgm_name = f"scene{scene.scene_number:02d}_bgm.wav"
        bgm_path = os.path.join(audio_dir, bgm_name)
        scene_dur_s = (cursor_ms - scene_start_ms) / 1000.0
        chosen = pick_bgm(scene.mood, scene_dur_s, bgm_path)
        segments.append(
            AudioSegment(
                scene_id=scene.scene_number,
                kind="bgm",
                character=None,
                audio_file=os.path.relpath(chosen, project_dir).replace("\\", "/"),
                start_ms=scene_start_ms,
                end_ms=cursor_ms,
                text=f"BGM ({scene.mood})",
            )
        )
        await _emit(progress, f"[Phase2] Scene {scene.scene_number} BGM = {os.path.basename(chosen)}")

    manifest = TimingManifest(
        segments=segments,
        total_duration_ms=cursor_ms,
        bgm_track=None,  # per-scene BGMs only — Phase 3 mixes them
    )
    await _emit(progress, f"[Phase2] Done — {len(segments)} segments, {cursor_ms / 1000:.1f}s total")
    return manifest


def realign_scene_durations(story: Story, manifest: TimingManifest) -> Story:
    """
    Update each scene's `duration_seconds` from the manifest so Phase 3
    knows exactly how long each scene's image must hold on screen.
    """
    by_scene: dict[int, list[AudioSegment]] = {}
    for s in manifest.segments:
        by_scene.setdefault(s.scene_id, []).append(s)
    for scene in story.scenes:
        segs = by_scene.get(scene.scene_number, [])
        if segs:
            start = min(s.start_ms for s in segs)
            end = max(s.end_ms for s in segs)
            scene.duration_seconds = max(2.0, (end - start) / 1000.0)
    return story
