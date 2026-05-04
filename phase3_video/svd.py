"""
Stable Video Diffusion client.

Sends a still image to the Kaggle FastAPI server's `/svd` endpoint and gets
back a short clip (~3 seconds, 25 frames at 6 fps). When the endpoint isn't
reachable — `KAGGLE_ENDPOINT` unset, network failure, server error — we fall
back to a Ken Burns loop on the still so the pipeline always emits a valid
clip of the requested duration.

The output clip is retimed via FFmpeg to match the requested `duration_s`
exactly so the per-line timing in the manifest stays accurate.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from phase3_video import _http, animation


def generate_clip(
    image_path: str | Path,
    duration_s: float,
    out_path: str | Path,
    *,
    motion_strength: int = 127,
    num_frames: int = 25,
) -> str:
    """Image → ~duration_s ambient-motion MP4."""
    image_path = str(image_path)
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    if _try_remote(image_path, duration_s, out_path,
                   motion_strength=motion_strength, num_frames=num_frames):
        return out_path
    return _passthrough(image_path, duration_s, out_path)


def _try_remote(
    image_path: str,
    duration_s: float,
    out_path: str,
    *,
    motion_strength: int,
    num_frames: int,
) -> bool:
    if not _http.have_endpoint():
        return False
    payload = {
        "image_b64": _http.file_to_b64(image_path),
        "motion_strength": int(motion_strength),
        "num_frames": int(num_frames),
    }
    # First call has to download SVD-XT (~9 GB) + load to GPU + run 25 steps —
    # that can take 3-5 minutes on Kaggle's free tier. Subsequent calls are ~10 s.
    resp = _http.post_endpoint("svd", payload, timeout=600.0)
    if not resp or "clip_mp4_b64" not in resp:
        return False
    raw_path = out_path + ".raw.mp4"
    try:
        _http.b64_to_file(resp["clip_mp4_b64"], raw_path)
    except Exception:
        return False

    # The model outputs ~3s at 6fps. Retime to match the audio line duration.
    fps = int(resp.get("fps") or 6)
    src_duration = float(num_frames) / max(1, fps)
    if not _retime_clip(raw_path, out_path, src_duration=src_duration, target_duration=duration_s):
        return False
    try:
        Path(raw_path).unlink(missing_ok=True)
    except Exception:
        pass
    return True


def _passthrough(image_path: str, duration_s: float, out_path: str) -> str:
    """No remote model — emit a Ken Burns loop on the still image."""
    return animation.ken_burns(image_path, duration_s, out_path, motion="zoom_in")


def _retime_clip(src: str, dst: str, *, src_duration: float, target_duration: float) -> bool:
    """
    Re-encode `src` to land at exactly `target_duration` seconds in `dst`.

    We always loop the source and trim to length — this is the simplest path
    and guarantees clean PTS / a single fps in the output, which Wav2Lip's
    frame counter on the Kaggle side is sensitive to. (An earlier `setpts`
    speed-change path produced clips with non-monotonic timestamps that broke
    downstream face-detection.)
    """
    ffmpeg = animation.ffmpeg_exe()
    if not ffmpeg:
        return False
    target_duration = max(0.5, float(target_duration))
    cmd = [
        ffmpeg, "-y", "-stream_loop", "-1", "-i", src,
        "-vf", f"scale={animation.WIDTH}:{animation.HEIGHT},setsar=1",
        "-an",
        "-r", str(animation.FPS),
        "-fps_mode", "cfr",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-t", f"{target_duration:.3f}",
        dst,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=120)
        return Path(dst).exists() and Path(dst).stat().st_size > 1000
    except Exception:
        return False


__all__ = ["generate_clip"]
