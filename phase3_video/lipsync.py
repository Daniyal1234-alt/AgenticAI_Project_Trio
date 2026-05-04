"""
Wav2Lip client.

Sends a still image OR an MP4 clip (typically the SVD output) plus a
dialogue audio track to the Kaggle FastAPI server's `/lipsync` endpoint.
Gets back an MP4 where the speaker's mouth moves with the audio.

When the endpoint isn't reachable, we fall back to a passthrough that just
muxes the still/clip with the audio at the right duration — no mouth
motion, but a valid MP4 the compositor can stitch in.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from phase3_video import _http, animation


def lipsync_line(
    face_input_path: str | Path,
    audio_path: str | Path,
    out_path: str | Path,
) -> str:
    """
    Render one dialogue line as a video.

    `face_input_path` may be a still image (.png/.jpg) or a video (.mp4).
    """
    face_in = str(face_input_path)
    audio_in = str(audio_path)
    out = str(out_path)
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    # Refuse to call the remote model on garbage audio (saves a long timeout).
    audio_ok = Path(audio_in).exists() and Path(audio_in).stat().st_size > 200
    if audio_ok and _try_remote(face_in, audio_in, out):
        return out
    return _passthrough(face_in, audio_in, out)


def _try_remote(face_in: str, audio_in: str, out_path: str) -> bool:
    if not _http.have_endpoint():
        return False

    # Wav2Lip wants a 16kHz mono WAV — resample once and cache next to the source.
    wav_path = _to_16k_wav(audio_in)
    if wav_path is None:
        return False

    face_kind = "video" if face_in.lower().endswith((".mp4", ".mov", ".webm")) else "image"
    payload = {
        "face_b64": _http.file_to_b64(face_in),
        "face_kind": face_kind,
        "audio_b64": _http.file_to_b64(wav_path),
    }
    # Wav2Lip first call has to load the GAN checkpoint + face detector
    # in a fresh subprocess (~50 s of fixed overhead). For longer dialogue
    # lines that's plus ~50 s per ~3 s of audio. 600 s gives generous margin.
    resp = _http.post_endpoint("lipsync", payload, timeout=600.0)
    if not resp or "mp4_b64" not in resp:
        return False
    try:
        _http.b64_to_file(resp["mp4_b64"], out_path)
    except Exception:
        return False
    return Path(out_path).exists() and Path(out_path).stat().st_size > 1000


def _passthrough(face_in: str, audio_in: str, out_path: str) -> str:
    """No remote model — mux face (image or video) with audio at target duration."""
    ffmpeg = animation.ffmpeg_exe()
    if not ffmpeg:
        Path(out_path).write_bytes(b"")
        return out_path

    duration = _probe_audio_duration(audio_in) or 2.0
    is_video = face_in.lower().endswith((".mp4", ".mov", ".webm"))

    if is_video:
        cmd = [
            ffmpeg, "-y",
            "-stream_loop", "-1", "-i", face_in,   # loop the face clip
            "-i", audio_in,
            "-t", f"{duration:.3f}",
            "-vf", f"scale={animation.WIDTH}:{animation.HEIGHT}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(animation.FPS),
            "-c:a", "aac", "-shortest",
            "-map", "0:v:0", "-map", "1:a:0",
            out_path,
        ]
    else:
        cmd = [
            ffmpeg, "-y",
            "-loop", "1", "-t", f"{duration:.3f}", "-i", face_in,
            "-i", audio_in,
            "-vf", f"scale={animation.WIDTH}:{animation.HEIGHT},setsar=1",
            "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
            "-r", str(animation.FPS),
            "-c:a", "aac", "-shortest",
            out_path,
        ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=120)
    except Exception:
        Path(out_path).write_bytes(b"")
    return out_path


def _to_16k_wav(audio_in: str) -> str | None:
    """Resample to 16kHz mono WAV (Wav2Lip's expected input format)."""
    ffmpeg = animation.ffmpeg_exe()
    if not ffmpeg:
        return None
    out = str(Path(audio_in).with_suffix("")) + ".wav16k.wav"
    try:
        subprocess.run(
            [
                ffmpeg, "-y", "-i", audio_in,
                "-ar", "16000", "-ac", "1",
                "-c:a", "pcm_s16le",
                out,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except Exception:
        return None
    return out if os.path.isfile(out) else None


def _probe_audio_duration(path: str) -> float | None:
    """
    Best-effort audio duration probe via ffprobe.

    We deliberately avoid MoviePy here — its `AudioFileClip.__del__` leaks
    a child ffmpeg process under pytest, which races with the next ffmpeg
    invocation in the same test session.
    """
    ffmpeg = animation.ffmpeg_exe()
    if not ffmpeg:
        return None
    # imageio-ffmpeg ships ffmpeg.exe; ffprobe lives next to it on most setups.
    candidates = [
        ffmpeg.replace("ffmpeg.exe", "ffprobe.exe").replace("ffmpeg", "ffprobe"),
        shutil.which("ffprobe"),
    ]
    ffprobe = next((p for p in candidates if p and os.path.isfile(p)), None)
    if ffprobe:
        try:
            result = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                check=True, capture_output=True, timeout=10,
            )
            return float(result.stdout.decode().strip())
        except Exception:
            pass
    # Fallback: ask ffmpeg itself by running an info-only pass to stderr.
    try:
        result = subprocess.run(
            [ffmpeg, "-i", path, "-f", "null", "-"],
            capture_output=True, timeout=15,
        )
        # ffmpeg prints "Duration: HH:MM:SS.cc" on stderr.
        import re
        m = re.search(rb"Duration: (\d+):(\d+):(\d+\.\d+)", result.stderr)
        if m:
            h, mn, s = m.groups()
            return int(h) * 3600 + int(mn) * 60 + float(s)
    except Exception:
        pass
    return None


__all__ = ["lipsync_line"]
