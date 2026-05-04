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
import json
import os
from pathlib import Path
from typing import Awaitable, Callable, Optional

from phase1_story import tools as phase1_tools
from phase2_audio.bgm import pick_bgm
from phase2_audio.tts import estimate_ms, prosody_for, synthesize, voice_for
from schemas.pipeline import AudioSegment, Story, TimingManifest


ProgressCb = Optional[Callable[[str], Awaitable[None] | None]]


async def _emit(cb: ProgressCb, msg: str) -> None:
    if cb is None:
        return
    res = cb(msg)
    if asyncio.iscoroutine(res):
        await res


def _load_emotion_map(project_dir: str) -> dict[tuple[int, int], str]:
    """
    Read Phase 1's `phase2_audio_handoff.json` (if present) and build a
    {(scene_id, line_index): emotion} lookup. Returns {} if the file is
    missing — callers fall back to recomputing emotions inline so tests
    that skip the handoff still work.
    """
    path = Path(project_dir) / "phase2_audio_handoff.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[tuple[int, int], str] = {}
    for seg in data.get("segments", []) or []:
        try:
            key = (int(seg["scene_id"]), int(seg["line_index"]))
        except (KeyError, TypeError, ValueError):
            continue
        out[key] = str(seg.get("emotion") or "calm")
    return out


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

    # Prefer Phase 1's persisted emotion tags; fall back to recomputing them.
    emotion_map = _load_emotion_map(project_dir)
    if emotion_map:
        await _emit(progress, f"[Phase2] Loaded {len(emotion_map)} emotion tags from handoff")

    segments: list[AudioSegment] = []
    cursor_ms = 0

    for scene in story.scenes:
        scene_start_ms = cursor_ms
        await _emit(progress, f"[Phase2] Scene {scene.scene_number} — synthesising dialogue")

        # Dialogue lines, sequential within a scene (so timing is monotonic).
        for idx, line in enumerate(scene.dialogue, start=1):
            voice = voices.get(line.character) or voice_for({"voice_style": "neutral", "role": ""})
            emotion = emotion_map.get((scene.scene_number, idx - 1))
            if emotion is None:
                tagged = phase1_tools.analyze_emotions([line])["value"]
                emotion = tagged[0]["emotion"] if tagged else "calm"
            pros = prosody_for(emotion)
            fname = f"scene{scene.scene_number:02d}_line{idx:02d}.mp3"
            fpath = os.path.join(audio_dir, fname)
            duration_ms = await synthesize(
                line.line, voice, fpath, rate=pros["rate"], pitch=pros["pitch"],
            )
            # synthesize() may have written a .wav fallback when edge-tts failed.
            # Pick the MP3 only if it exists AND has real bytes (>200 — empty/
            # truncated streams from a dropped connection look like 0-byte files).
            wav_fallback = os.path.splitext(fpath)[0] + ".wav"
            if os.path.exists(fpath) and os.path.getsize(fpath) > 200:
                written = fpath
            else:
                written = wav_fallback
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

    # Standalone manifest file — the spec calls this out by name.
    manifest_path = Path(project_dir) / "timing_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    await _emit(progress, f"[Phase2] Wrote {manifest_path.name}")

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
