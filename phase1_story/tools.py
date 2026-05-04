"""
Phase 1 — named tool functions invoked by the Story / Character / Script agents.

Each tool is a deterministic, pure-Python function (no LLM calls) returning a
ToolResult dict::

    {"ok": bool, "issues": list[str], "value": Any}

Routing in `agent.py` reads `ok` to decide whether to retry. `value` is the
tool's primary output (e.g., the estimated duration in seconds, the enriched
visual prompt, the per-line emotion tags). `issues` is appended to the run
log + summary.json so the grader can see exactly which tool flagged what.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from schemas.pipeline import (
    Character,
    DialogueLine,
    Scene,
    SceneOutline,
    StoryOutline,
)


ToolResult = dict[str, Any]


VALID_VOICE_STYLES = {
    "warm",
    "gravelly",
    "whispered",
    "cheerful",
    "stern",
    "youthful",
    "wise",
    "neutral",
    "determined",
    "anxious",
    "serene",
}

VALID_ROLES = {"protagonist", "antagonist", "narrator", "supporting"}

WORDS_PER_MIN = 165.0  # matches phase2_audio.tts.estimate_ms baseline


# --------------------------------------------------------------------------- #
# Story-agent tools                                                            #
# --------------------------------------------------------------------------- #


def validate_story_arc(outline: StoryOutline | dict) -> ToolResult:
    """
    Verifies a StoryOutline has a complete intro → rising_action → climax →
    resolution arc and that scene_outlines exist.
    """
    if isinstance(outline, dict):
        outline = StoryOutline.model_validate(outline)
    issues: list[str] = []
    arc = outline.arc
    for field in ("intro", "rising_action", "climax", "resolution"):
        if not getattr(arc, field, "").strip():
            issues.append(f"arc.{field} is empty")
    if len(outline.scene_outlines) < 1:
        issues.append("scene_outlines is empty")
    if not outline.title.strip():
        issues.append("title is empty")
    return {"ok": not issues, "issues": issues, "value": outline}


def estimate_duration(thing: Any) -> ToolResult:
    """
    Estimates total duration in seconds for one of:
      - a Scene / dict scene  (uses dialogue word count or scene.duration_seconds)
      - a SceneOutline        (uses target_duration_s)
      - a list of any of the above
      - a StoryOutline        (sums all scene_outlines)

    Returns the estimate in `value` (float seconds).
    """
    seconds = 0.0
    issues: list[str] = []

    def _from_scene(sc: dict | Scene) -> float:
        if isinstance(sc, Scene):
            d = sc.duration_seconds
            words = sum(len((line.line or "").split()) for line in sc.dialogue)
        else:
            d = float(sc.get("duration_seconds") or 0.0)
            words = sum(len((dl.get("line") or "").split()) for dl in sc.get("dialogue", []))
        if d:
            return d
        if words:
            return max(2.0, (words / WORDS_PER_MIN) * 60.0)
        return 8.0

    def _from_outline(so: dict | SceneOutline) -> float:
        if isinstance(so, SceneOutline):
            return so.target_duration_s
        return float(so.get("target_duration_s") or 8.0)

    if isinstance(thing, StoryOutline):
        thing = list(thing.scene_outlines)
    elif isinstance(thing, dict) and "scene_outlines" in thing:
        thing = list(thing["scene_outlines"])

    if isinstance(thing, list):
        for item in thing:
            if isinstance(item, (Scene, dict)) and (
                isinstance(item, Scene) or "dialogue" in item
            ):
                seconds += _from_scene(item)
            elif isinstance(item, (SceneOutline, dict)):
                seconds += _from_outline(item)
            else:
                issues.append(f"unknown list item type: {type(item).__name__}")
    elif isinstance(thing, (Scene,)):
        seconds = _from_scene(thing)
    elif isinstance(thing, (SceneOutline,)):
        seconds = _from_outline(thing)
    elif isinstance(thing, dict):
        seconds = _from_scene(thing) if "dialogue" in thing else _from_outline(thing)
    else:
        issues.append(f"unsupported argument type: {type(thing).__name__}")

    return {"ok": seconds > 0 and not issues, "issues": issues, "value": float(seconds)}


# --------------------------------------------------------------------------- #
# Character-agent tools                                                        #
# --------------------------------------------------------------------------- #


def check_consistency(
    roster: list[Character] | dict,
    outline: StoryOutline | dict | None = None,
) -> ToolResult:
    """
    Validates a character roster:
      - ≥1 protagonist
      - no duplicate names (case-insensitive)
      - voice_style in the allowed set (warns, doesn't fail)
      - role in the allowed set
      - appearance non-empty (so Phase 3 has something to draw)
    """
    if isinstance(roster, dict):
        roster = [Character.model_validate(c) for c in roster.get("characters", [])]
    issues: list[str] = []

    if not roster:
        return {"ok": False, "issues": ["roster is empty"], "value": []}

    names_lower = [c.name.strip().lower() for c in roster]
    if len(set(names_lower)) != len(names_lower):
        dups = {n for n in names_lower if names_lower.count(n) > 1}
        issues.append(f"duplicate character names: {sorted(dups)}")

    roles = [c.role.lower() for c in roster]
    if "protagonist" not in roles:
        issues.append("no protagonist in roster")

    for c in roster:
        if c.role.lower() not in VALID_ROLES:
            issues.append(f"{c.name}: role '{c.role}' not in {sorted(VALID_ROLES)}")
        if c.voice_style.lower() not in VALID_VOICE_STYLES:
            issues.append(
                f"{c.name}: voice_style '{c.voice_style}' not in {sorted(VALID_VOICE_STYLES)}"
            )
        if not c.appearance.strip():
            issues.append(f"{c.name}: appearance is empty (Phase 3 needs it)")

    return {"ok": not issues, "issues": issues, "value": roster}


# --------------------------------------------------------------------------- #
# Script-agent tools                                                           #
# --------------------------------------------------------------------------- #


def build_visual_prompt(scene: Scene | dict, style: str = "cinematic") -> ToolResult:
    """
    Synthesises a rich image-generator prompt from a scene's metadata.
    Used both to enrich missing visual_prompts and to overwrite weak ones.
    """
    if isinstance(scene, dict):
        sc = Scene.model_validate(scene)
    else:
        sc = scene
    prompt = (
        f"{style} still, {sc.location or 'an evocative location'}, "
        f"{(sc.time_of_day or 'day').lower()} lighting, mood: {sc.mood}, "
        f"action: {(sc.action or 'a quiet moment')[:120]}, "
        "ultra detailed, 16:9, dramatic composition"
    )
    return {"ok": True, "issues": [], "value": prompt}


def validate_duration(scene: Scene | dict) -> ToolResult:
    """
    Confirms a scene's duration is in 2..60 s and that it has at least one
    dialogue line (otherwise Phase 2 has nothing to synthesise).
    """
    if isinstance(scene, dict):
        sc = Scene.model_validate(scene)
    else:
        sc = scene
    issues: list[str] = []
    if sc.duration_seconds < 2.0:
        issues.append(f"scene {sc.scene_number}: duration {sc.duration_seconds}s < 2s")
    if sc.duration_seconds > 60.0:
        issues.append(f"scene {sc.scene_number}: duration {sc.duration_seconds}s > 60s")
    if not sc.dialogue:
        issues.append(f"scene {sc.scene_number}: no dialogue lines")
    return {"ok": not issues, "issues": issues, "value": sc.duration_seconds}


_EMOTION_KEYWORDS = {
    "tense":      ("danger", "warning", "afraid", "fear", "hide", "run", "no!", "stop", "panic"),
    "joyful":     ("yay", "wonderful", "amazing", "love", "smile", "laugh", "haha", "great"),
    "melancholy": ("alone", "lost", "miss", "gone", "sigh", "regret", "tear", "goodbye"),
    "urgent":     ("now", "hurry", "quick", "must", "immediately", "fast", "right away"),
    "curious":    ("what", "why", "how", "wonder", "perhaps", "maybe", "could it"),
    "determined": ("will", "won't stop", "no other way", "have to", "must keep"),
}


def analyze_emotions(dialogue: Iterable[DialogueLine | dict]) -> ToolResult:
    """
    Tags each dialogue line with an emotion via simple keyword matching.

    Returns a list of {character, line, emotion} dicts in `value`.
    Used by the serializer to enrich phase2_audio_handoff.json so Phase 2
    can pick voice tones without re-running an LLM.
    """
    tagged: list[dict[str, str]] = []
    for d in dialogue:
        if isinstance(d, dict):
            character = d.get("character", "")
            line = d.get("line", "")
            direction = d.get("direction", "") or ""
        else:
            character = d.character
            line = d.line
            direction = d.direction or ""
        haystack = f"{line} {direction}".lower()
        emotion = "calm"
        for label, keywords in _EMOTION_KEYWORDS.items():
            if any(kw in haystack for kw in keywords):
                emotion = label
                break
        # stage directions like "(whispered)" or "(angry)" override
        m = re.match(r"\s*\(([^)]+)\)", direction)
        if m:
            tag = m.group(1).lower().strip()
            for label in _EMOTION_KEYWORDS:
                if tag.startswith(label) or label in tag:
                    emotion = label
                    break
            if tag in {"whispered", "soft", "softly"}:
                emotion = "calm"
            elif tag in {"angry", "shouted", "yelling"}:
                emotion = "tense"
        tagged.append({"character": character, "line": line, "emotion": emotion})
    return {"ok": True, "issues": [], "value": tagged}


__all__ = [
    "validate_story_arc",
    "estimate_duration",
    "check_consistency",
    "build_visual_prompt",
    "validate_duration",
    "analyze_emotions",
    "VALID_VOICE_STYLES",
    "VALID_ROLES",
]
