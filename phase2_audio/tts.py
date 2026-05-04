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
_STYLE_VOICE_MAP = {
    "warm": "en-US-AriaNeural",
    "narrator": "en-GB-RyanNeural",
    "wise": "en-GB-RyanNeural",
    "gravelly": "en-US-GuyNeural",
    "stern": "en-US-GuyNeural",
    "whispered": "en-US-JennyNeural",
    "cheerful": "en-US-JennyNeural",
    "youthful": "en-US-AnaNeural",
    "determined": "en-US-DavisNeural",
    "neutral": "en-US-JennyNeural",
}

_ROLE_VOICE_MAP = {
    "narrator": "en-GB-RyanNeural",
    "protagonist": "en-US-AriaNeural",
    "antagonist": "en-US-GuyNeural",
    "supporting": "en-US-JennyNeural",
}


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

    Uses edge-tts; falls back to a silent WAV if anything goes wrong.
    `out_path` decides extension: edge-tts writes MP3, fallback writes WAV.

    `rate` and `pitch` follow edge-tts's syntax (e.g. "+15%", "-10Hz").
    Defaults are no-ops so existing callers behave identically.
    """
    text = (text or "").strip()
    if not text:
        text = "(silence)"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Try edge-tts first.
    try:
        import edge_tts  # type: ignore

        # edge-tts always writes an MP3 stream regardless of out_path extension.
        comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        await comm.save(out_path)
        if os.path.getsize(out_path) > 200:  # got real audio
            return _probe_duration_ms(out_path) or estimate_ms(text)
    except Exception:
        # fall through to silent fallback
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
    Try to read duration from a media file. Best-effort — return None if we
    can't tell. moviepy/ffprobe both work; we use moviepy because it's already
    a dependency.
    """
    # MoviePy 2.x: top-level import. MoviePy 1.x: moviepy.editor.
    AudioFileClip = None
    try:
        from moviepy import AudioFileClip  # type: ignore
    except ImportError:
        try:
            from moviepy.editor import AudioFileClip  # type: ignore
        except ImportError:
            return None
    try:
        with AudioFileClip(path) as clip:
            return int(clip.duration * 1000)
    except Exception:
        return None
