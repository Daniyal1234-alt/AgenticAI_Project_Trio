"""
Shared LLM helper.

Why this exists: every phase that uses an LLM needs the same fallback story.
If OPENAI_API_KEY is set we call ChatGPT; if it isn't, we return a deterministic
stub so tests, demos, and offline development still produce a complete pipeline
artifact (just less creative).
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional


def have_openai() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def chat_json(prompt: str, *, model: str = "gpt-4o-mini", temperature: float = 0.7) -> Any:
    """
    Call the chat model and parse its response as JSON.

    Strips markdown fences and falls back to {} on parse failure rather than raising —
    pipeline phases handle empty results explicitly.
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=temperature,
    )
    raw = llm.invoke(prompt).content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to recover the first {...} or [...] block.
        for opener, closer in (("{", "}"), ("[", "]")):
            i, j = raw.find(opener), raw.rfind(closer)
            if i != -1 and j != -1 and j > i:
                try:
                    return json.loads(raw[i : j + 1])
                except json.JSONDecodeError:
                    continue
        return {}


def stub_story(prompt: str, num_scenes: int = 3) -> dict:
    """
    Deterministic offline fallback used when no OPENAI_API_KEY is set.
    Produces a small, schema-valid story so the rest of the pipeline can run.
    """
    short = (prompt or "an unspecified adventure").strip()[:80]
    title = f"Echoes of {short.split()[0].title() if short else 'Tomorrow'}"
    return {
        "title": title,
        "logline": f"A short film inspired by: {short}.",
        "style": "cinematic",
        "characters": [
            {
                "name": "Narrator",
                "role": "narrator",
                "voice_style": "warm",
                "appearance": "Unseen voice — no on-screen presence",
                "personality": "Reflective, calm, observant",
            },
            {
                "name": "Protagonist",
                "role": "protagonist",
                "voice_style": "determined",
                "appearance": "A figure in their late twenties, traveler's clothes, weathered face",
                "personality": "Curious, brave, principled",
            },
        ],
        "scenes": [
            {
                "scene_number": i + 1,
                "heading": f"SCENE {i + 1} — ESTABLISHING SHOT",
                "location": "An open landscape that frames the moment",
                "time_of_day": ["DAWN", "DAY", "DUSK"][i % 3],
                "mood": ["hopeful", "tense", "reflective"][i % 3],
                "duration_seconds": 8.0,
                "characters": ["Narrator", "Protagonist"],
                "action": f"Beat {i + 1}: the story advances another step.",
                "visual_prompt": (
                    f"Cinematic still, {short}, moody lighting, ultra detailed, 16:9, "
                    f"beat {i + 1} of {num_scenes}"
                ),
                "dialogue": [
                    {
                        "character": "Narrator",
                        "line": f"It was the {['first', 'middle', 'final'][i % 3]} chapter of {title}.",
                        "direction": "(softly)",
                    },
                    {
                        "character": "Protagonist",
                        "line": "I have to keep going. There's no other way.",
                        "direction": "(determined)",
                    },
                ],
            }
            for i in range(num_scenes)
        ],
    }


def stub_outline(prompt: str, num_scenes: int = 3, style: str = "cinematic") -> dict:
    """Deterministic StoryOutline used when no OPENAI_API_KEY is set or the LLM fails."""
    short = (prompt or "an unspecified adventure").strip()[:80]
    title = f"Echoes of {short.split()[0].title() if short else 'Tomorrow'}"
    moods = ["hopeful", "tense", "reflective", "joyful", "melancholy"]
    return {
        "title": title,
        "logline": f"A short film inspired by: {short}.",
        "style": style,
        "themes": ["discovery", "resolve"],
        "arc": {
            "intro": f"Establish the world hinted at by '{short}'.",
            "rising_action": "The protagonist commits to a course of action.",
            "climax": "A defining choice or revelation.",
            "resolution": "Quiet aftermath, with the world subtly changed.",
        },
        "scene_outlines": [
            {
                "scene_number": i + 1,
                "heading": f"SCENE {i + 1} — ESTABLISHING SHOT",
                "mood": moods[i % len(moods)],
                "target_duration_s": 8.0,
                "summary": f"Beat {i + 1}: the story advances another step.",
            }
            for i in range(num_scenes)
        ],
    }


def stub_roster(outline: dict | None = None) -> dict:
    """Deterministic CharacterRoster — a narrator + a protagonist (always ≥2)."""
    return {
        "characters": [
            {
                "name": "Narrator",
                "role": "narrator",
                "voice_style": "warm",
                "appearance": "Unseen voice — no on-screen presence",
                "personality": "Reflective, calm, observant",
            },
            {
                "name": "Protagonist",
                "role": "protagonist",
                "voice_style": "youthful",
                "appearance": "A figure in their late twenties, traveler's clothes, weathered face",
                "personality": "Curious, brave, principled",
            },
        ],
    }


def stub_script(outline: dict, roster: dict, prompt: str = "") -> dict:
    """Deterministic ScriptOutput — full scenes with dialogue + visuals."""
    style = (outline or {}).get("style") or "cinematic"
    title = (outline or {}).get("title") or "Untitled"
    short = (prompt or title).strip()[:80]
    scene_outlines = (outline or {}).get("scene_outlines") or []
    num_scenes = max(1, len(scene_outlines))
    times = ["DAWN", "DAY", "DUSK"]
    return {
        "scenes": [
            {
                "scene_number": (so.get("scene_number") or i + 1),
                "heading": so.get("heading") or f"SCENE {i + 1} — ESTABLISHING SHOT",
                "location": "An open landscape that frames the moment",
                "time_of_day": times[i % len(times)],
                "mood": so.get("mood") or "neutral",
                "duration_seconds": float(so.get("target_duration_s") or 8.0),
                "characters": ["Narrator", "Protagonist"],
                "action": so.get("summary") or f"Beat {i + 1}: the story advances another step.",
                "visual_prompt": (
                    f"{style} still, {short}, moody lighting, ultra detailed, 16:9, "
                    f"beat {i + 1} of {num_scenes}"
                ),
                "dialogue": [
                    {
                        "character": "Narrator",
                        "line": f"It was the {['first', 'middle', 'final'][i % 3]} chapter of {title}.",
                        "direction": "(softly)",
                    },
                    {
                        "character": "Protagonist",
                        "line": "I have to keep going. There's no other way.",
                        "direction": "(determined)",
                    },
                ],
            }
            for i, so in enumerate(scene_outlines or [{}] * num_scenes)
        ],
    }


def stub_intent(query: str) -> dict:
    """Tiny rule-based intent classifier used when no LLM key is set."""
    q = (query or "").lower()
    target = "video"
    intent = "recompose_video"
    parameters: dict[str, Any] = {}
    scope = "all"

    if any(k in q for k in ("voice", "tts", "tone", "narrator", "speak")):
        target = "audio"
        intent = "change_voice_tone"
    elif any(k in q for k in ("music", "bgm", "soundtrack", "score")):
        target = "audio"
        intent = "change_bgm"
    elif any(k in q for k in ("subtitle", "caption")):
        target = "video"
        intent = "remove_subtitle" if "remove" in q else "add_subtitle"
    elif any(k in q for k in ("speed", "faster", "slower")):
        target = "video"
        intent = "adjust_speed"
    elif any(k in q for k in ("darker", "brighter", "sepia", "noir", "color", "filter")):
        target = "video_frame"
        intent = "apply_filter"
        if "darker" in q:
            parameters["filter"] = "darken"
        elif "brighter" in q:
            parameters["filter"] = "brighten"
        elif "sepia" in q:
            parameters["filter"] = "sepia"
        elif "noir" in q:
            parameters["filter"] = "noir"
    elif any(k in q for k in ("script", "story", "regenerate", "rewrite")):
        target = "script"
        intent = "regenerate_script"
    elif any(k in q for k in ("character", "design", "appearance", "look")):
        target = "video_frame"
        intent = "regenerate_character_frames"

    # Scope detection: scene N
    import re

    m = re.search(r"scene\s+(\d+)", q)
    if m:
        scope = f"scene:{m.group(1)}"

    return {
        "intent": intent,
        "target": target,
        "scope": scope,
        "parameters": parameters,
        "confidence": 0.6,
        "explanation": "Offline rule-based classifier (no OPENAI_API_KEY set).",
    }
