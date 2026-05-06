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
    pipeline phases handle empty results explicitly. Auth/quota/rate-limit errors
    propagate up so callers can fall through to stubs; we also log them to stderr
    once per process so out-of-quota keys become visible immediately instead of
    silently producing low-quality stub output.
    """
    import sys

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=temperature,
    )
    try:
        raw = llm.invoke(prompt).content.strip()
    except Exception as exc:
        # Surface the actual API error once per kind so the user sees
        # quota / auth / network failures without having to read the WS log.
        global _LOGGED_LLM_ERRORS
        try:
            _LOGGED_LLM_ERRORS
        except NameError:
            _LOGGED_LLM_ERRORS = set()
        kind = type(exc).__name__
        if kind not in _LOGGED_LLM_ERRORS:
            _LOGGED_LLM_ERRORS.add(kind)
            print(f"[LLM] {kind}: {str(exc)[:300]}", file=sys.stderr, flush=True)
        raise
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


# --------------------------------------------------------------------------- #
# Context-aware stub helpers — these run when OPENAI_API_KEY isn't set or when
# the LLM keeps producing non-conforming JSON. They try to extract enough
# structure from the user's prompt that the resulting story doesn't feel
# like Mad-Libs filler. None of this is a substitute for a real LLM, but it's
# a much better last resort than "It was the first chapter of Echoes of A".
# --------------------------------------------------------------------------- #


_GENRE_PROFILES: list[tuple[tuple[str, ...], dict]] = [
    (
        ("astronaut", "space", "spaceship", "station", "mars", "rocket", "orbital"),
        {
            "themes": ["isolation", "duty"],
            "title_suffix": "the Last Signal",
            "location": "Space station corridor with a viewport to the stars",
            "protagonist": {
                "name": "Captain Ana",
                "role": "protagonist",
                "voice_style": "determined",
                "appearance": "Astronaut in a worn flight suit, helmet visor open, weathered face",
                "personality": "Steady under pressure, quietly methodical",
            },
        },
    ),
    (
        ("ocean", "sea", "submarine", "diver", "wreck", "abyss", "deep"),
        {
            "themes": ["mystery", "depth"],
            "title_suffix": "Beneath the Surface",
            "location": "Cold blue water, the surface visible far above",
            "protagonist": {
                "name": "Lyra Cole",
                "role": "protagonist",
                "voice_style": "wise",
                "appearance": "Diver with silvering hair and patient eyes, no helmet, face visible",
                "personality": "Curious, fearless, comfortable with silence",
            },
        },
    ),
    (
        ("detective", "noir", "investigator", "crime", "case"),
        {
            "themes": ["truth", "consequence"],
            "title_suffix": "the 3:47 AM Case",
            "location": "City street under sodium-yellow lights, rain on asphalt",
            "protagonist": {
                "name": "Inspector Hale",
                "role": "protagonist",
                "voice_style": "gravelly",
                "appearance": "Trench coat, sharp eyes that miss nothing, no hat covering face",
                "personality": "Cynical but principled, three steps ahead",
            },
        },
    ),
    (
        ("library", "book", "scribe", "scholar", "archive"),
        {
            "themes": ["knowledge", "memory"],
            "title_suffix": "the Forgotten Page",
            "location": "Quiet library aisle between tall wooden shelves",
            "protagonist": {
                "name": "Mira",
                "role": "protagonist",
                "voice_style": "youthful",
                "appearance": "Young scholar with ink-stained fingers, wide curious eyes",
                "personality": "Patient, attentive, drawn to small details",
            },
        },
    ),
    (
        ("forest", "tree", "wood", "wild", "creature"),
        {
            "themes": ["balance", "wonder"],
            "title_suffix": "What the Forest Knows",
            "location": "Forest clearing dappled with light through tall trees",
            "protagonist": {
                "name": "Wren",
                "role": "protagonist",
                "voice_style": "serene",
                "appearance": "Forest-dweller in earthy clothes, calm watchful gaze",
                "personality": "Quiet, observant, protective of small things",
            },
        },
    ),
    (
        ("dragon", "wizard", "castle", "tower", "kingdom", "magic"),
        {
            "themes": ["legacy", "courage"],
            "title_suffix": "the Last Spell",
            "location": "Stone tower at the edge of a wind-swept ridge",
            "protagonist": {
                "name": "Ash",
                "role": "protagonist",
                "voice_style": "wise",
                "appearance": "Robed scholar-mage with silver-streaked hair, kind eyes",
                "personality": "Patient, slow to anger, fond of riddles",
            },
        },
    ),
    (
        ("cafe", "barista", "shop", "regular", "morning"),
        {
            "themes": ["connection", "ritual"],
            "title_suffix": "the Tuesday Regular",
            "location": "Small coffee shop, rain on the windows",
            "protagonist": {
                "name": "Sam",
                "role": "protagonist",
                "voice_style": "warm",
                "appearance": "Barista with a knowing smile, apron worn comfortable",
                "personality": "Quietly observant, kind without making a show of it",
            },
        },
    ),
]


def _genre_for(prompt: str) -> dict:
    """Pick the first genre profile whose keywords match the prompt."""
    p = (prompt or "").lower()
    for keywords, profile in _GENRE_PROFILES:
        if any(k in p for k in keywords):
            return profile
    # Generic fallback
    return {
        "themes": ["discovery", "resolve"],
        "title_suffix": "a Quiet Reckoning",
        "location": "An evocative setting that frames the moment",
        "protagonist": {
            "name": "Avery",
            "role": "protagonist",
            "voice_style": "youthful",
            "appearance": "Late twenties, alert posture, expressive eyes, face fully visible",
            "personality": "Curious, brave, principled",
        },
    }


def _title_from_prompt(prompt: str, suffix: str) -> str:
    """First-three-significant-words of the prompt + a genre suffix."""
    words = [w for w in (prompt or "").split() if len(w) > 3]
    head = " ".join(words[:3]).title() if words else "An Unspecified Tale"
    return f"{head}: {suffix}"


def stub_outline(prompt: str, num_scenes: int = 3, style: str = "cinematic") -> dict:
    """Deterministic StoryOutline used when no OPENAI_API_KEY is set or the LLM fails."""
    short = (prompt or "an unspecified adventure").strip()[:120]
    profile = _genre_for(short)
    title = _title_from_prompt(short, profile["title_suffix"])
    moods = ["hopeful", "tense", "reflective", "joyful", "melancholy"]
    return {
        "title": title,
        "logline": f"A {style} short inspired by: {short}.",
        "style": style,
        "themes": profile["themes"],
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
                "summary": f"Beat {i + 1}: the story advances toward its turn.",
            }
            for i in range(num_scenes)
        ],
    }


def stub_roster(outline: dict | None = None, prompt: str = "") -> dict:
    """Deterministic CharacterRoster — derives a thematic cast from the prompt."""
    profile = _genre_for(prompt)
    return {
        "characters": [
            {
                "name": "Narrator",
                "role": "narrator",
                "voice_style": "warm",
                "appearance": "Unseen voice — no on-screen presence",
                "personality": "Reflective, calm, observant",
            },
            profile["protagonist"],
        ],
    }


# Mood → (paired-direction, paired-line) tuples used by stub_script.
_MOOD_DIALOGUE: dict[str, list[tuple[str, str]]] = {
    "hopeful": [
        ("(softly, watching)",   "Maybe — maybe this time it's different."),
        ("(quietly resolved)",   "There's a way through. There always is."),
    ],
    "tense": [
        ("(low, urgent)",        "Something's wrong. I can feel it."),
        ("(eyes on the door)",   "We have to move. Now."),
    ],
    "urgent": [
        ("(calling out)",        "Move! There's no time left!"),
        ("(shouting back)",      "Now or never — go!"),
    ],
    "melancholy": [
        ("(quiet)",              "I never thought it would end like this."),
        ("(half-smile)",         "Some things you don't get to take with you."),
    ],
    "reflective": [
        ("(thinking aloud)",     "Looking back, I see what we missed."),
        ("(slowly)",             "It all led here. Every step."),
    ],
    "joyful": [
        ("(laughing)",           "I can't believe we made it!"),
        ("(grinning)",           "Look at all of this — really look."),
    ],
    "neutral": [
        ("(observing)",          "That's how it goes, I suppose."),
        ("(noting)",             "And so the moment passed."),
    ],
}


def _stub_dialogue_for_scene(
    mood: str,
    speaker_a: str,
    speaker_b: str,
    scene_index: int,
    num_scenes: int,
) -> list[dict]:
    """Build 2 dialogue lines for a scene from the mood-driven template table.

    The protagonist gets the first line (more dramatic / leading the beat);
    the narrator-or-other gets the second (commenting / closing).
    """
    pair = _MOOD_DIALOGUE.get((mood or "neutral").lower(), _MOOD_DIALOGUE["neutral"])
    direction_a, line_a = pair[0]
    direction_b, line_b = pair[1] if len(pair) > 1 else pair[0]
    return [
        {"character": speaker_a, "line": line_a, "direction": direction_a},
        {"character": speaker_b, "line": line_b, "direction": direction_b},
    ]


def stub_script(outline: dict, roster: dict, prompt: str = "") -> dict:
    """Deterministic ScriptOutput — full scenes with dialogue + visuals.

    Uses the actual roster character names (no longer hardcodes
    Narrator/Protagonist) and picks dialogue keyed off the per-scene mood
    so the script reads coherently even without an LLM.
    """
    style = (outline or {}).get("style") or "cinematic"
    title = (outline or {}).get("title") or "Untitled"
    short = (prompt or title).strip()[:120]
    scene_outlines = (outline or {}).get("scene_outlines") or []
    num_scenes = max(1, len(scene_outlines))
    profile = _genre_for(short)
    location = profile["location"]

    # Resolve real character names from the roster — no more hardcoded pair.
    roster_chars = (roster or {}).get("characters") or []
    char_names = [c.get("name") for c in roster_chars if c.get("name")]
    if not char_names:
        char_names = ["Narrator", "Protagonist"]
    # Speaker A = first non-narrator (typically the protagonist) if available;
    # else the first listed. Speaker B = the narrator (or second listed).
    non_narrator = [c for c in roster_chars if c.get("role") != "narrator"]
    narrator     = next((c for c in roster_chars if c.get("role") == "narrator"), None)
    speaker_a = (non_narrator[0]["name"] if non_narrator else char_names[0])
    speaker_b = (narrator["name"] if narrator else (char_names[1] if len(char_names) > 1 else speaker_a))

    times = ["DAWN", "DAY", "DUSK", "NIGHT"]
    return {
        "scenes": [
            {
                "scene_number": (so.get("scene_number") or i + 1),
                "heading": so.get("heading") or f"SCENE {i + 1} — ESTABLISHING SHOT",
                "location": location,
                "time_of_day": times[i % len(times)],
                "mood": (so.get("mood") or "neutral"),
                "duration_seconds": float(so.get("target_duration_s") or 8.0),
                "characters": list(char_names),
                "action": (
                    so.get("summary")
                    or f"Beat {i + 1}: {short[:60]} — the story advances another step."
                ),
                "visual_prompt": (
                    f"{style} still, {short}, "
                    f"{(so.get('mood') or 'neutral')} mood, "
                    f"ultra detailed, 16:9, beat {i + 1} of {num_scenes}"
                ),
                "dialogue": _stub_dialogue_for_scene(
                    so.get("mood") or "neutral",
                    speaker_a, speaker_b, i, num_scenes,
                ),
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
