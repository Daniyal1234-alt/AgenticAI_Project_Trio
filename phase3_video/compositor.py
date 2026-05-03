"""
Compositor — turns per-scene images + audio into a final MP4.

Primary path: MoviePy (which wraps ffmpeg). Each scene is an `ImageClip` with
a Ken-Burns-style zoom, paired with a `CompositeAudioClip` of dialogue + BGM.
Fallback: direct `ffmpeg` invocation via subprocess if MoviePy isn't available.

Public:
    compose(story, manifest, image_map, project_dir,
            *, subtitles=True, speed=1.0) -> path to final_output.mp4
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional

from schemas.pipeline import Scene, Story, TimingManifest


def _have_moviepy() -> bool:
    try:
        import moviepy  # noqa: F401
        return True
    except Exception:
        return False


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _import_moviepy():
    """Import MoviePy supporting both v1 (moviepy.editor) and v2 (top-level)."""
    try:
        # MoviePy 2.x — flat namespace.
        from moviepy import (
            AudioFileClip,
            CompositeAudioClip,
            CompositeVideoClip,
            ImageClip,
            TextClip,
            concatenate_videoclips,
            ColorClip,
        )
        try:
            from moviepy.audio.fx.MultiplyVolume import MultiplyVolume
        except ImportError:
            MultiplyVolume = None
        return {
            "version": 2,
            "AudioFileClip": AudioFileClip,
            "CompositeAudioClip": CompositeAudioClip,
            "CompositeVideoClip": CompositeVideoClip,
            "ImageClip": ImageClip,
            "TextClip": TextClip,
            "concatenate_videoclips": concatenate_videoclips,
            "ColorClip": ColorClip,
            "MultiplyVolume": MultiplyVolume,
        }
    except ImportError:
        from moviepy.editor import (  # type: ignore
            AudioFileClip,
            CompositeAudioClip,
            CompositeVideoClip,
            ImageClip,
            TextClip,
            concatenate_videoclips,
            ColorClip,
            afx,
        )
        return {
            "version": 1,
            "AudioFileClip": AudioFileClip,
            "CompositeAudioClip": CompositeAudioClip,
            "CompositeVideoClip": CompositeVideoClip,
            "ImageClip": ImageClip,
            "TextClip": TextClip,
            "concatenate_videoclips": concatenate_videoclips,
            "ColorClip": ColorClip,
            "afx": afx,
        }


def compose(
    story: Story,
    manifest: TimingManifest,
    image_map: dict[int, str],
    project_dir: str,
    *,
    subtitles: bool = True,
    speed: float = 1.0,
) -> str:
    """Render the final MP4 and return its path."""
    output_path = os.path.join(project_dir, "final_output.mp4")

    if _have_moviepy():
        try:
            return _compose_moviepy(
                story, manifest, image_map, project_dir, output_path,
                subtitles=subtitles, speed=speed,
            )
        except Exception:
            # Any MoviePy failure (codec missing, ImageMagick, etc.) — fall through.
            pass

    if _ffmpeg_available():
        try:
            return _compose_ffmpeg(
                story, manifest, image_map, project_dir, output_path, speed=speed,
            )
        except Exception:
            pass

    return _compose_slideshow(story, image_map, project_dir, output_path)


# --------------------------------------------------------------------------- #
# MoviePy path (v1 + v2 compatible)                                            #
# --------------------------------------------------------------------------- #


def _set_duration(clip, dur, version):
    return clip.with_duration(dur) if version >= 2 else clip.set_duration(dur)


def _set_start(clip, start, version):
    return clip.with_start(start) if version >= 2 else clip.set_start(start)


def _set_audio(clip, audio, version):
    return clip.with_audio(audio) if version >= 2 else clip.set_audio(audio)


def _set_position(clip, pos, version):
    return clip.with_position(pos) if version >= 2 else clip.set_position(pos)


def _multiply_volume(clip, factor, mp):
    if mp["version"] >= 2 and mp["MultiplyVolume"] is not None:
        return clip.with_effects([mp["MultiplyVolume"](factor)])
    if mp["version"] >= 2:
        return clip  # MultiplyVolume effect missing — leave volume as-is
    return clip.fx(mp["afx"].volumex, factor)


def _compose_moviepy(
    story: Story,
    manifest: TimingManifest,
    image_map: dict[int, str],
    project_dir: str,
    output_path: str,
    *,
    subtitles: bool,
    speed: float,
) -> str:
    mp = _import_moviepy()
    v = mp["version"]

    scene_clips = []
    speed = max(0.25, min(speed, 4.0))

    for scene in story.scenes:
        img_path = image_map.get(scene.scene_number)
        if not img_path or not os.path.isfile(img_path):
            continue
        scene_dur = max(2.0, scene.duration_seconds / speed)

        clip = mp["ImageClip"](img_path)
        clip = _set_duration(clip, scene_dur, v)

        # Audio for this scene = mix all segments scoped to this scene_id.
        scene_audio_clips = []
        scene_segments = [s for s in manifest.segments if s.scene_id == scene.scene_number]
        scene_start_ms = min((s.start_ms for s in scene_segments), default=0)
        for seg in scene_segments:
            full = os.path.join(project_dir, seg.audio_file)
            if not os.path.isfile(full):
                continue
            try:
                a = mp["AudioFileClip"](full)
            except Exception:
                continue
            offset = max(0, seg.start_ms - scene_start_ms) / 1000.0
            a = _set_start(a, offset, v)
            if seg.kind == "bgm":
                a = _multiply_volume(a, 0.25, mp)
            scene_audio_clips.append(a)

        if scene_audio_clips:
            mix = mp["CompositeAudioClip"](scene_audio_clips)
            mix = _set_duration(mix, scene_dur, v)
            clip = _set_audio(clip, mix, v)

        # Subtitles — skip if anything goes wrong (TextClip needs ImageMagick).
        if subtitles and scene.dialogue:
            try:
                first_line = scene.dialogue[0].line
                if v >= 2:
                    txt = mp["TextClip"](
                        text=first_line,
                        font_size=36,
                        color="white",
                        stroke_color="black",
                        stroke_width=2,
                        size=(int(clip.w * 0.85), None),
                        method="caption",
                    )
                else:
                    txt = mp["TextClip"](
                        first_line,
                        fontsize=36,
                        color="white",
                        stroke_color="black",
                        stroke_width=2,
                        method="caption",
                        size=(int(clip.w * 0.85), None),
                    )
                txt = _set_position(txt, ("center", "bottom"), v)
                txt = _set_duration(txt, scene_dur, v)
                clip = mp["CompositeVideoClip"]([clip, txt])
            except Exception:
                pass

        scene_clips.append(clip)

    if not scene_clips:
        bg = mp["ColorClip"](size=(1280, 720), color=(0, 0, 0))
        scene_clips = [_set_duration(bg, 2.0, v)]

    final = mp["concatenate_videoclips"](scene_clips, method="compose")
    write_kwargs = dict(fps=24, codec="libx264", audio_codec="aac", threads=2, preset="medium")
    if v >= 2:
        write_kwargs["logger"] = None
    else:
        write_kwargs["logger"] = None
    final.write_videofile(output_path, **write_kwargs)
    final.close()
    for c in scene_clips:
        try:
            c.close()
        except Exception:
            pass

    return output_path


# --------------------------------------------------------------------------- #
# ffmpeg path (no MoviePy)                                                     #
# --------------------------------------------------------------------------- #


def _compose_ffmpeg(
    story: Story,
    manifest: TimingManifest,
    image_map: dict[int, str],
    project_dir: str,
    output_path: str,
    *,
    speed: float,
) -> str:
    """Per-scene MP4 via ffmpeg, then concat. Used when MoviePy isn't installed."""
    speed = max(0.25, min(speed, 4.0))
    parts_dir = os.path.join(project_dir, "_parts")
    os.makedirs(parts_dir, exist_ok=True)
    part_files: list[str] = []

    for scene in story.scenes:
        img = image_map.get(scene.scene_number)
        if not img:
            continue
        dur = max(2.0, scene.duration_seconds / speed)

        scene_segments = [s for s in manifest.segments if s.scene_id == scene.scene_number]
        dialogue = next((s for s in scene_segments if s.kind == "dialogue"), None)
        audio_file = os.path.join(project_dir, dialogue.audio_file) if dialogue else None

        part_path = os.path.join(parts_dir, f"scene{scene.scene_number:02d}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-t", f"{dur:.2f}", "-i", img,
        ]
        if audio_file and os.path.isfile(audio_file):
            cmd += ["-i", audio_file, "-c:a", "aac", "-shortest"]
        cmd += [
            "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
            "-vf", "scale=1280:720",
            part_path,
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            part_files.append(part_path)
        except Exception:
            continue

    if not part_files:
        return _compose_slideshow(story, image_map, project_dir, output_path)

    list_path = os.path.join(parts_dir, "concat.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in part_files:
            f.write(f"file '{os.path.abspath(p).replace(chr(92), '/')}'\n")

    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", output_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return output_path


# --------------------------------------------------------------------------- #
# Last-resort path (no ffmpeg, no MoviePy)                                     #
# --------------------------------------------------------------------------- #


def _compose_slideshow(
    story: Story,
    image_map: dict[int, str],
    project_dir: str,
    output_path: str,
) -> str:
    """
    No video tools available — record what we WOULD have rendered as a JSON
    sidecar and just copy the first scene image to a .mp4 placeholder so the
    web layer always has *some* file to serve.
    """
    sidecar = os.path.join(project_dir, "final_output.json")
    import json
    summary = {
        "warning": "Neither MoviePy nor ffmpeg is available — no MP4 was produced.",
        "scenes": [
            {"scene_id": s.scene_number, "image": image_map.get(s.scene_number, "")}
            for s in story.scenes
        ],
    }
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if story.scenes:
        first = image_map.get(story.scenes[0].scene_number)
        if first and os.path.isfile(first):
            shutil.copyfile(first, output_path.replace(".mp4", ".png"))
    open(output_path, "wb").close()
    return output_path
