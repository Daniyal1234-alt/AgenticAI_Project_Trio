"""
TTS layer.

Primary: edge-tts (Microsoft Edge's free SSML endpoint, no API key).
Fallback: a silent WAV of estimated length, so the pipeline still produces
files when edge-tts is unreachable (offline lab, no internet).

Public:
    voice_for(character) -> str   # picks an Edge-TTS voice based on style
    synthesize(text, voice, out_path) -> ms duration
"""

from __future__ import annotations

import asyncio
import math
import os
import struct
import wave
from typing import Optional


# Sensible Edge-TTS voice picks per style hint.
# IMPORTANT: only use voices that currently exist on Microsoft's TTS service.
# Retired voices return `NoAudioReceived: No audio was received` and silently
# fall through to the silent-WAV path. Verify with `edge_tts.list_voices()`
# before adding entries.
_STYLE_VOICE_MAP = {
    "warm": "en-US-AriaNeural",
    "narrator": "en-GB-RyanNeural",
    "wise": "en-GB-RyanNeural",
    "gravelly": "en-US-GuyNeural",
    "stern": "en-US-GuyNeural",
    "whispered": "en-US-JennyNeural",
    "cheerful": "en-US-JennyNeural",
    "youthful": "en-US-AnaNeural",
    "determined": "en-US-GuyNeural",
    "neutral": "en-US-JennyNeural",
}

_ROLE_VOICE_MAP = {
    "narrator": "en-GB-RyanNeural",
    "protagonist": "en-US-AriaNeural",
    "antagonist": "en-US-GuyNeural",
    "supporting": "en-US-JennyNeural",
}

# Known-good voice used when the chosen voice is rejected by edge-tts.
_FALLBACK_VOICE = "en-US-AriaNeural"


# Emotion → edge-tts prosody overrides (rate / pitch). Keeps the audio in
# step with the script's emotional arc — a tense line speeds up and lowers
# pitch, a melancholy line slows down and drops pitch, etc. The buckets
# match `phase1_story.tools.analyze_emotions`.
EMOTION_PROSODY: dict[str, dict[str, str]] = {
    "calm":       {"rate": "-5%",  "pitch": "+0Hz"},
    "tense":      {"rate": "+15%", "pitch": "-10Hz"},
    "urgent":     {"rate": "+25%", "pitch": "+5Hz"},
    "joyful":     {"rate": "+10%", "pitch": "+15Hz"},
    "melancholy": {"rate": "-15%", "pitch": "-15Hz"},
    "curious":    {"rate": "+0%",  "pitch": "+10Hz"},
    "determined": {"rate": "+5%",  "pitch": "-5Hz"},
}

_NEUTRAL_PROSODY = {"rate": "+0%", "pitch": "+0Hz"}


def prosody_for(emotion: str) -> dict[str, str]:
    """Look up rate/pitch overrides for an emotion tag; unknowns return neutral."""
    return EMOTION_PROSODY.get((emotion or "").lower(), dict(_NEUTRAL_PROSODY))


def voice_for(character) -> str:
    """Resolve a TTS voice ID from a Character (or dict-like)."""
    if isinstance(character, dict):
        style = (character.get("voice_style") or "").lower()
        role = (character.get("role") or "").lower()
    else:
        style = (getattr(character, "voice_style", "") or "").lower()
        role = (getattr(character, "role", "") or "").lower()
    return _STYLE_VOICE_MAP.get(style) or _ROLE_VOICE_MAP.get(role) or "en-US-JennyNeural"


def estimate_ms(text: str, wpm: int = 165) -> int:
    """Rough duration of `text` at `wpm` words per minute."""
    words = max(1, len(text.split()))
    return int(words / wpm * 60_000)


async def synthesize(
    text: str,
    voice: str,
    out_path: str,
    *,
    rate: str = "+0%",
    pitch: str = "+0Hz",
) -> int:
    """
    Synthesize `text` into `out_path` (MP3) and return the duration in ms.

    Uses edge-tts with up to 3 retries (Microsoft's free endpoint occasionally
    drops a connection mid-stream). If every attempt fails, falls back to a
    silent WAV at the estimated duration so downstream phases keep working.

    `rate` and `pitch` follow edge-tts's syntax (e.g. "+15%", "-10Hz").
    Defaults are no-ops so existing callers behave identically.
    """
    text = (text or "").strip()
    if not text:
        text = "(silence)"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Try edge-tts up to 3 times.
    try:
        import edge_tts  # type: ignore
    except ImportError:
        edge_tts = None  # type: ignore

    if edge_tts is not None:
        import asyncio  # local import to avoid pollution at module level
        import sys

        # Try the requested voice first; on a NoAudioReceived (e.g. the voice
        # has been retired by MS) fall back to a known-good voice so the line
        # still gets real speech instead of silence.
        voices_to_try = [voice]
        if voice != _FALLBACK_VOICE:
            voices_to_try.append(_FALLBACK_VOICE)

        last_error: Exception | None = None
        for v in voices_to_try:
            for attempt in range(3):
                try:
                    comm = edge_tts.Communicate(text, v, rate=rate, pitch=pitch)
                    await comm.save(out_path)
                    if os.path.getsize(out_path) > 200:  # got real audio
                        if v != voice:
                            print(
                                f"[Phase2] voice {voice!r} rejected — used "
                                f"fallback {v!r} instead",
                                file=sys.stderr, flush=True,
                            )
                        return _probe_duration_ms(out_path) or estimate_ms(text)
                    last_error = RuntimeError(
                        f"edge-tts wrote {os.path.getsize(out_path)} bytes (<200)"
                    )
                except Exception as exc:
                    last_error = exc
                # Linear backoff between attempts.
                await asyncio.sleep(0.4 * (attempt + 1))

            # Don't waste backoff on the fallback voice if the first voice
            # got `NoAudioReceived` (definite voice rejection — not transient).
            if last_error and "NoAudio" in type(last_error).__name__:
                continue
            break  # transient errors won't be helped by switching voices

        if last_error is not None:
            print(
                f"[Phase2] edge-tts failed after retries for "
                f"voice={voice!r} text={text[:60]!r}: "
                f"{type(last_error).__name__}: {last_error}",
                file=sys.stderr, flush=True,
            )

    # All attempts failed — clean up any partial/empty MP3 so the pipeline's
    # file-picker chooses the WAV fallback instead.
    if os.path.exists(out_path):
        try:
            os.unlink(out_path)
        except OSError:
            pass

    # Fallback — write a silent WAV at the estimated duration.
    fallback_path = os.path.splitext(out_path)[0] + ".wav"
    duration_ms = estimate_ms(text)
    _write_silent_wav(fallback_path, duration_ms)
    return duration_ms


def _write_silent_wav(path: str, duration_ms: int, sample_rate: int = 22_050) -> None:
    n_frames = max(1, int(sample_rate * duration_ms / 1000))
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<" + "h" * n_frames, *([0] * n_frames)))


def _probe_duration_ms(path: str) -> Optional[int]:
    """
    Read duration from an audio/video file via ffprobe.

    We avoid MoviePy here on purpose — its `AudioFileClip.__del__` can leak a
    child ffmpeg subprocess under pytest, which races with subsequent ffmpeg
    invocations. ffprobe ships next to ffmpeg in `imageio-ffmpeg`'s binaries
    on every install.
    """
    import shutil
    import subprocess

    ffprobe_path = None
    try:
        import imageio_ffmpeg  # type: ignore
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        candidate = ffmpeg_path.replace("ffmpeg.exe", "ffprobe.exe").replace("ffmpeg", "ffprobe")
        if os.path.isfile(candidate):
            ffprobe_path = candidate
    except Exception:
        pass
    ffprobe_path = ffprobe_path or shutil.which("ffprobe")
    if not ffprobe_path:
        return None
    try:
        result = subprocess.run(
            [ffprobe_path, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            check=True, capture_output=True, timeout=10,
        )
        return int(float(result.stdout.decode().strip()) * 1000)
    except Exception:
        return None
