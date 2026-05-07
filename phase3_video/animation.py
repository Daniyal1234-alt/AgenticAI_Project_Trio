"""
Ken Burns animations via FFmpeg's `zoompan` filter.

A pure compositing layer — no model dependency. Takes a still image, returns
a short MP4 with a single, deterministic motion (zoom_in / zoom_out / pan_left
/ pan_right / static). Used for:

  - Pre/post-dialogue silence in a scene (panning slowly over the establishing
    image).
  - The `svd.generate_clip` passthrough fallback (when the remote endpoint
    is unreachable, we still emit a valid moving clip).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


WIDTH, HEIGHT = 1280, 720
FPS = 24


def ffmpeg_exe() -> Optional[str]:
    """
    Return a path to an ffmpeg binary. Prefers the one bundled by
    `imageio-ffmpeg` (always present in our deps) over a system install.
    """
    try:
        import imageio_ffmpeg  # type: ignore
        path = imageio_ffmpeg.get_ffmpeg_exe()
        if path and os.path.isfile(path):
            return path
    except Exception:
        pass
    return shutil.which("ffmpeg")


# zoompan expressions. `d` (duration in frames) is filled in per call.
# `s` and `fps` go on the zoompan filter so the output is a uniform 1280x720@24.
_MOTIONS = {
    "zoom_in":   "zoompan=z='min(zoom+0.0015,1.4)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={d}:s={W}x{H}:fps={fps}",
    "zoom_out":  "zoompan=z='if(eq(on,0),1.4,max(zoom-0.0015,1.0))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={d}:s={W}x{H}:fps={fps}",
    "pan_left":  "zoompan=z='1.2':x='(iw-iw/zoom)*(1-on/{d})':y='ih/2-(ih/zoom/2)':d={d}:s={W}x{H}:fps={fps}",
    "pan_right": "zoompan=z='1.2':x='(iw-iw/zoom)*(on/{d})':y='ih/2-(ih/zoom/2)':d={d}:s={W}x{H}:fps={fps}",
}


def ken_burns(
    image_path: str | Path,
    duration_s: float,
    out_path: str | Path,
    *,
    motion: str = "zoom_in",
) -> str:
    """
    Render a single Ken-Burns clip from a still image.

    Returns the output path. Falls back to a no-motion loop if FFmpeg is
    missing OR the requested motion is unknown.
    """
    image_path = str(image_path)
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    duration_s = max(0.5, float(duration_s))
    n_frames = max(1, int(duration_s * FPS))

    ffmpeg = ffmpeg_exe()
    if not ffmpeg:
        # No ffmpeg at all — write a tiny placeholder so callers don't crash.
        Path(out_path).write_bytes(b"")
        return out_path

    motion = (motion or "static").lower()
    expr = _MOTIONS.get(motion)

    # We mux a silent audio track (lavfi anullsrc, mono 48 kHz) into every
    # ken-burns output so the clip can be concatenated alongside dialogue
    # clips that have audio — without uniform stream layout, the concat
    # filter / demuxer drops the inconsistent input.
    silent_audio = [
        "-f", "lavfi", "-t", f"{duration_s:.3f}",
        "-i", "anullsrc=channel_layout=mono:sample_rate=48000",
    ]
    audio_codec = ["-c:a", "aac", "-shortest"]

    if expr is None:
        # Static loop — no zoompan, just resize-and-pad to 1280x720.
        vf = (
            f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1"
        )
        cmd = [
            ffmpeg, "-y",
            "-loop", "1", "-t", f"{duration_s:.3f}", "-i", image_path,
            *silent_audio,
            "-vf", vf,
            "-r", str(FPS),
            "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
            *audio_codec,
            out_path,
        ]
    else:
        # Pad to 1280x720 first so zoompan output is consistent regardless
        # of source aspect ratio.
        prefilter = (
            f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
        )
        vf = prefilter + expr.format(d=n_frames, W=WIDTH, H=HEIGHT, fps=FPS)
        cmd = [
            ffmpeg, "-y",
            "-loop", "1", "-t", f"{duration_s:.3f}", "-i", image_path,
            *silent_audio,
            "-vf", vf,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            *audio_codec,
            out_path,
        ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
    except Exception:
        # Best-effort: if zoompan failed (older ffmpeg, expression quirk),
        # retry without motion so the pipeline still produces a clip.
        if motion != "static":
            return ken_burns(image_path, duration_s, out_path, motion="static")
        Path(out_path).write_bytes(b"")
    return out_path


__all__ = ["ken_burns", "ffmpeg_exe", "WIDTH", "HEIGHT", "FPS"]
