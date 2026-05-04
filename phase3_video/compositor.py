"""
Final-MP4 compositor.

Two public entry points:

    compose_scene(scene_id, line_clips, line_texts, bgm_path, out_path, ...)
        Stitches the per-line lip-sync clips for a single scene, mixes BGM
        under at -12 dB, and (optionally) burns subtitles via FFmpeg
        `drawtext`. Output is a per-scene MP4.

    concat_with_transitions(scene_clips, out_path, ...)
        Concatenates per-scene MP4s with FFmpeg `xfade` crossfades. Falls
        back to a hard-cut concat when the chain is too short or anything
        in the xfade graph fails.

We deliberately use FFmpeg directly (not MoviePy) for all video ops in
this rebuild — the old MoviePy `TextClip` path silently dropped subtitles
on machines without ImageMagick. `drawtext` is a libfreetype filter that
ships with every modern FFmpeg build, so subs always render.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable, Optional

from phase3_video.animation import FPS, HEIGHT, WIDTH, ffmpeg_exe


# --------------------------------------------------------------------------- #
# Per-scene composition                                                       #
# --------------------------------------------------------------------------- #


def compose_scene(
    scene_id: int,
    line_clips: list[str],
    line_texts: list[str],
    bgm_path: Optional[str],
    out_path: str,
    *,
    subtitles: bool = True,
    speed: float = 1.0,
) -> str:
    """
    Stitch the per-line lip-sync clips for one scene into a single MP4 with
    BGM mixed under at -12 dB, and (optionally) subtitles burnt in.

    Returns the output path. On any FFmpeg failure, returns the path to
    whatever was last successfully written so the pipeline keeps going.
    """
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_exe()

    line_clips = [str(p) for p in line_clips if p and Path(p).exists() and Path(p).stat().st_size > 1000]
    if not ffmpeg or not line_clips:
        # Nothing playable — emit an empty file so downstream concat skips us.
        Path(out_path).write_bytes(b"")
        return out_path

    # Stage 1 — concatenate the per-line clips into a single dialogue track.
    parts_dir = Path(out_path).parent / "_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    dialogue_clip = str(parts_dir / f"scene{scene_id:02d}_dialogue.mp4")
    if not _concat_clips(ffmpeg, line_clips, dialogue_clip):
        return out_path

    current = dialogue_clip

    # Stage 2 — mix BGM under at -12 dB (≈ 0.25 linear), looped if shorter.
    if bgm_path and Path(bgm_path).exists() and Path(bgm_path).stat().st_size > 1000:
        with_bgm = str(parts_dir / f"scene{scene_id:02d}_bgm.mp4")
        if _mix_bgm(ffmpeg, current, bgm_path, with_bgm, gain=0.25):
            current = with_bgm

    # Stage 3 — burn subtitles via drawtext.
    if subtitles and line_texts:
        subbed = str(parts_dir / f"scene{scene_id:02d}_subs.mp4")
        if _burn_subtitles(ffmpeg, current, line_clips, line_texts, subbed):
            current = subbed

    # Stage 4 — speed adjust (Phase 5 "speed up scene") if requested.
    if abs(speed - 1.0) > 1e-3:
        sped = str(parts_dir / f"scene{scene_id:02d}_speed.mp4")
        if _adjust_speed(ffmpeg, current, sped, speed=speed):
            current = sped

    # Move the final result into place.
    if current != out_path:
        try:
            if Path(out_path).exists():
                Path(out_path).unlink()
            os.replace(current, out_path)
        except Exception:
            # Best-effort copy.
            Path(out_path).write_bytes(Path(current).read_bytes())

    return out_path


def _concat_clips(ffmpeg: str, clips: list[str], out_path: str) -> bool:
    """Concatenate clips via the ffmpeg concat demuxer (assumes uniform codec)."""
    if len(clips) == 1:
        try:
            Path(out_path).write_bytes(Path(clips[0]).read_bytes())
            return True
        except Exception:
            return False

    list_path = str(Path(out_path).with_suffix(".txt"))
    try:
        with open(list_path, "w", encoding="utf-8") as f:
            for c in clips:
                # ffmpeg's concat demuxer needs forward slashes and quoted paths.
                p = os.path.abspath(c).replace("\\", "/").replace("'", r"'\''")
                f.write(f"file '{p}'\n")
    except Exception:
        return False

    cmd = [
        ffmpeg, "-y", "-f", "concat", "-safe", "0",
        "-i", list_path,
        # Re-encode rather than `-c copy` — line clips may differ slightly
        # in fps/timebase from passthrough vs Wav2Lip output.
        "-vf", f"scale={WIDTH}:{HEIGHT},setsar=1",
        "-r", str(FPS),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=180)
        return Path(out_path).exists() and Path(out_path).stat().st_size > 1000
    except Exception:
        return False
    finally:
        try:
            os.unlink(list_path)
        except Exception:
            pass


def _mix_bgm(ffmpeg: str, dialogue_path: str, bgm_path: str, out_path: str, *, gain: float = 0.25) -> bool:
    """
    Layer BGM under the dialogue track at the given linear gain.
    BGM is looped to dialogue length and trimmed.
    """
    cmd = [
        ffmpeg, "-y",
        "-i", dialogue_path,
        "-stream_loop", "-1", "-i", bgm_path,
        "-filter_complex",
        f"[1:a]volume={gain:.3f}[bgm];"
        f"[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=0[mix]",
        "-map", "0:v",
        "-map", "[mix]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=120)
        return Path(out_path).exists() and Path(out_path).stat().st_size > 1000
    except Exception:
        return False


_DRAWTEXT_ESCAPE = str.maketrans({
    "\\": r"\\",
    ":": r"\:",
    "'": r"\'",
    "%": r"\%",
})


def _font_path() -> Optional[str]:
    candidates = [
        os.environ.get("PHASE3_FONT"),
        "C:/Windows/Fonts/seguisb.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p.replace("\\", "/")
    return None


def _probe_clip_duration(ffmpeg: str, path: str) -> Optional[float]:
    """Best-effort clip duration probe via ffprobe (shipped with ffmpeg)."""
    ffprobe = (ffmpeg or "").replace("ffmpeg.exe", "ffprobe.exe").replace("ffmpeg", "ffprobe")
    if not ffprobe or not os.path.isfile(ffprobe):
        return None
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            check=True, capture_output=True, timeout=10,
        )
        return float(out.stdout.decode().strip())
    except Exception:
        return None


def _burn_subtitles(
    ffmpeg: str,
    current: str,
    line_clips: list[str],
    line_texts: list[str],
    out_path: str,
) -> bool:
    """
    Burn subtitles via drawtext. One filter per line, gated by `enable=`
    so each subtitle appears only during its own line.
    """
    font = _font_path()
    durations: list[float] = []
    for c in line_clips:
        d = _probe_clip_duration(ffmpeg, c)
        if d is None or d <= 0:
            return False
        durations.append(d)

    cumulative = 0.0
    filters: list[str] = []
    for i, (text, dur) in enumerate(zip(line_texts, durations)):
        start = cumulative
        end = cumulative + dur
        cumulative = end
        clean = (text or "").translate(_DRAWTEXT_ESCAPE)
        if not clean.strip():
            continue
        parts = [
            f"drawtext=text='{clean}'",
            "fontsize=34",
            "fontcolor=white",
            "bordercolor=black@0.7",
            "borderw=3",
            "box=1",
            "boxcolor=black@0.45",
            "boxborderw=12",
            "x=(w-text_w)/2",
            "y=h-110",
            f"enable='between(t,{start:.3f},{end:.3f})'",
        ]
        if font:
            parts.insert(1, f"fontfile='{font}'")
        filters.append(":".join(parts))

    if not filters:
        return False

    vf = ",".join(filters)
    cmd = [
        ffmpeg, "-y", "-i", current,
        "-vf", vf,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-c:a", "copy",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=180)
        return Path(out_path).exists() and Path(out_path).stat().st_size > 1000
    except Exception:
        return False


def _adjust_speed(ffmpeg: str, src: str, dst: str, *, speed: float) -> bool:
    """Speed up / slow down both video and audio."""
    speed = max(0.25, min(4.0, float(speed)))
    # Audio atempo is clamped to [0.5, 2.0] per filter — chain if needed.
    a_filters = []
    remaining = speed
    while remaining > 2.0:
        a_filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        a_filters.append("atempo=0.5")
        remaining /= 0.5
    a_filters.append(f"atempo={remaining:.6f}")
    a_chain = ",".join(a_filters)
    cmd = [
        ffmpeg, "-y", "-i", src,
        "-filter_complex",
        f"[0:v]setpts=PTS/{speed:.6f}[v];[0:a]{a_chain}[a]",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        dst,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=120)
        return Path(dst).exists() and Path(dst).stat().st_size > 1000
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Final concat with crossfade transitions                                     #
# --------------------------------------------------------------------------- #


def concat_with_transitions(
    scene_clips: Iterable[str],
    out_path: str,
    *,
    transition_duration: float = 0.5,
) -> str:
    """
    Concatenate per-scene MP4s into a final video with `xfade` crossfade
    transitions between them. Falls back to a hard-cut concat (or a copy
    of the first clip) if the xfade chain fails.
    """
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_exe()
    clips = [str(p) for p in scene_clips if p and Path(p).exists() and Path(p).stat().st_size > 1000]

    if not ffmpeg or not clips:
        # Last-resort placeholder so the web layer still has a file to serve.
        Path(out_path).write_bytes(b"")
        return out_path

    if len(clips) == 1:
        Path(out_path).write_bytes(Path(clips[0]).read_bytes())
        return out_path

    # Try crossfade first; if it errors, fall back to a hard-cut concat.
    if _concat_xfade(ffmpeg, clips, out_path, transition_duration=transition_duration):
        return out_path
    if _concat_hardcut(ffmpeg, clips, out_path):
        return out_path

    Path(out_path).write_bytes(Path(clips[0]).read_bytes())
    return out_path


def _concat_xfade(ffmpeg: str, clips: list[str], out_path: str, *, transition_duration: float) -> bool:
    """Crossfade chain between N clips."""
    durations: list[float] = []
    for c in clips:
        d = _probe_clip_duration(ffmpeg, c)
        if d is None or d <= 0:
            return False
        durations.append(d)

    inputs: list[str] = []
    for c in clips:
        inputs.extend(["-i", c])

    n = len(clips)
    # Build the xfade chain offsets.
    cumulative = 0.0
    v_chain: list[str] = []
    a_chain: list[str] = []
    last_v, last_a = "0:v", "0:a"
    for i in range(1, n):
        cumulative += durations[i - 1] - transition_duration
        if cumulative <= 0:
            return False  # transition longer than the prior clip
        v_label = f"v{i}"
        a_label = f"a{i}"
        v_chain.append(
            f"[{last_v}][{i}:v]xfade=transition=fade:"
            f"duration={transition_duration:.3f}:offset={cumulative:.3f}[{v_label}]"
        )
        a_chain.append(
            f"[{last_a}][{i}:a]acrossfade=d={transition_duration:.3f}[{a_label}]"
        )
        last_v, last_a = v_label, a_label

    filter_complex = ";".join(v_chain + a_chain)
    cmd = [ffmpeg, "-y", *inputs,
           "-filter_complex", filter_complex,
           "-map", f"[{last_v}]", "-map", f"[{last_a}]",
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-c:a", "aac",
           out_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=300)
        return Path(out_path).exists() and Path(out_path).stat().st_size > 1000
    except Exception:
        return False


def _concat_hardcut(ffmpeg: str, clips: list[str], out_path: str) -> bool:
    """Plain concat-demuxer fallback."""
    list_path = str(Path(out_path).with_suffix(".concat.txt"))
    try:
        with open(list_path, "w", encoding="utf-8") as f:
            for c in clips:
                p = os.path.abspath(c).replace("\\", "/").replace("'", r"'\''")
                f.write(f"file '{p}'\n")
    except Exception:
        return False
    cmd = [
        ffmpeg, "-y", "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-vf", f"scale={WIDTH}:{HEIGHT},setsar=1",
        "-r", str(FPS),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=300)
        return Path(out_path).exists() and Path(out_path).stat().st_size > 1000
    except Exception:
        return False
    finally:
        try:
            os.unlink(list_path)
        except Exception:
            pass


__all__ = ["compose_scene", "concat_with_transitions"]
