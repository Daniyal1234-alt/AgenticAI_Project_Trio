"""
Background-music layer.

Two sources, in priority order:
1. A user-provided file in `assets/bgm/<mood>.mp3` (e.g. `tense.mp3`, `hopeful.mp3`).
2. Procedurally generated tone — a slow, mood-tinted sine pad we can write with
   nothing but the standard library + numpy. Not great music, but a valid
   timing-manifest entry that demonstrates the layer is wired in.

Public: `pick_bgm(mood, total_seconds, out_path)` returns the chosen path.
"""

from __future__ import annotations

import math
import os
import struct
import wave

# Mood → fundamental frequency hint for procedural BGM.
_MOOD_FREQ = {
    "hopeful": 261.63,    # C4
    "tense": 196.00,      # G3
    "melancholy": 220.00, # A3
    "reflective": 246.94, # B3
    "joyful": 293.66,     # D4
    "neutral": 220.00,
}


def pick_bgm(mood: str, total_seconds: float, out_path: str) -> str:
    """Return path to a BGM track, generating one if no asset exists."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    asset = os.path.join(here, "assets", "bgm", f"{(mood or 'neutral').lower()}.mp3")
    if os.path.isfile(asset):
        return asset

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    _generate_pad(out_path, mood, max(2.0, total_seconds))
    return out_path


def _generate_pad(out_path: str, mood: str, seconds: float, sample_rate: int = 22_050) -> None:
    """Write a simple two-tone sine pad at a low volume — non-distracting BGM."""
    f1 = _MOOD_FREQ.get((mood or "neutral").lower(), 220.0)
    f2 = f1 * 1.5  # perfect fifth above
    n = int(sample_rate * seconds)
    amp = 0.18

    samples = bytearray()
    for i in range(n):
        t = i / sample_rate
        # Slow amplitude envelope so the pad breathes.
        env = 0.5 * (1.0 + math.sin(2 * math.pi * 0.07 * t))
        v = amp * env * (math.sin(2 * math.pi * f1 * t) + 0.6 * math.sin(2 * math.pi * f2 * t))
        s = max(-1.0, min(1.0, v))
        samples += struct.pack("<h", int(s * 32_000))

    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(samples))
