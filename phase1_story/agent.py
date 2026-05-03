"""
Phase 1 LangGraph agent.

Pipeline (StateGraph):
    [START]
       │
       ▼
    generate_story  ──►  validate_story  ──►  enrich_visuals  ──►  [END]
                            │ invalid
                            ▼
                        (retry once, then accept best-effort)

`run_phase1(prompt, num_scenes, style)` returns a validated `Story` Pydantic model.
"""

from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from phase1_story.prompts import STORY_PROMPT, VALIDATE_PROMPT
from schemas.llm import chat_json, have_openai, stub_story
from schemas.pipeline import Story


class Phase1State(TypedDict, total=False):
    prompt: str
    num_scenes: int
    style: str
    story_dict: dict[str, Any]
    validation: dict[str, Any]
    retries: int
    story: Story | None
    log: list[str]


def _log(state: Phase1State, msg: str) -> list[str]:
    arr = list(state.get("log") or [])
    arr.append(msg)
    return arr


# --------------------------------------------------------------------------- #
# Nodes                                                                        #
# --------------------------------------------------------------------------- #


def generate_story_node(state: Phase1State) -> Phase1State:
    """Ask the LLM (or fallback stub) for a JSON story matching our schema."""
    prompt = state.get("prompt", "")
    num_scenes = int(state.get("num_scenes") or 3)
    style = state.get("style") or "cinematic"
    log = _log(state, f"[Phase1] generate_story (scenes={num_scenes}, style={style})")

    if have_openai():
        rendered = STORY_PROMPT.format(
            prompt=prompt,
            style=style,
            num_scenes=num_scenes,
            total_duration=num_scenes * 8.0,
        )
        try:
            data = chat_json(rendered, temperature=0.85)
            if not isinstance(data, dict) or "scenes" not in data:
                log.append("[Phase1] LLM returned non-conforming JSON — using stub")
                data = stub_story(prompt, num_scenes)
            else:
                log.append(f"[Phase1] LLM returned story '{data.get('title', '?')}'")
        except Exception as exc:  # network, auth, etc.
            log.append(f"[Phase1] LLM error: {exc!r} — using stub")
            data = stub_story(prompt, num_scenes)
    else:
        log.append("[Phase1] No OPENAI_API_KEY — using deterministic stub story")
        data = stub_story(prompt, num_scenes)

    return {**state, "story_dict": data, "log": log}


def validate_story_node(state: Phase1State) -> Phase1State:
    """Schema-validate via Pydantic, then run a lightweight LLM consistency check."""
    log = _log(state, "[Phase1] validate_story")
    data = state.get("story_dict") or {}
    errors: list[str] = []

    # Pydantic structural validation
    try:
        story = Story.model_validate(data)
    except ValidationError as exc:
        log.append(f"[Phase1] Pydantic errors: {exc.error_count()}")
        errors.append(str(exc))
        return {**state, "validation": {"is_valid": False, "errors": errors}, "log": log}

    # Consistency rules (purely local — no LLM needed)
    char_names = {c.name for c in story.characters}
    for sc in story.scenes:
        for d in sc.dialogue:
            if d.character not in char_names:
                errors.append(
                    f"Scene {sc.scene_number}: dialogue speaker '{d.character}' "
                    "missing from characters[]"
                )
        if not sc.dialogue:
            errors.append(f"Scene {sc.scene_number}: no dialogue lines")
        if not sc.characters:
            errors.append(f"Scene {sc.scene_number}: no characters listed")

    nums = [sc.scene_number for sc in story.scenes]
    if sorted(nums) != list(range(1, len(nums) + 1)):
        errors.append(f"Scene numbers not contiguous: {nums}")

    is_valid = len(errors) == 0
    log.append(f"[Phase1] Validation {'OK' if is_valid else 'FAIL'} ({len(errors)} errors)")

    return {
        **state,
        "validation": {"is_valid": is_valid, "errors": errors, "suggestions": []},
        "story": story if is_valid else None,
        "log": log,
    }


def enrich_visuals_node(state: Phase1State) -> Phase1State:
    """
    Make sure every scene has a visual_prompt suitable for image generation.
    The LLM usually gives one; if not, we synthesize a basic one from the action.
    """
    log = _log(state, "[Phase1] enrich_visuals")
    story = state.get("story")
    if story is None:
        return {**state, "log": log}

    style = story.style or "cinematic"
    for sc in story.scenes:
        if not sc.visual_prompt or len(sc.visual_prompt) < 20:
            sc.visual_prompt = (
                f"{style} still, {sc.location or 'an evocative location'}, "
                f"{sc.time_of_day.lower()} lighting, mood: {sc.mood}, "
                f"action: {sc.action[:120] if sc.action else 'a quiet moment'}, "
                "ultra detailed, 16:9, dramatic composition"
            )
            log.append(f"[Phase1] synthesised visual_prompt for scene {sc.scene_number}")

    return {**state, "story": story, "log": log}


# --------------------------------------------------------------------------- #
# Routing                                                                      #
# --------------------------------------------------------------------------- #


def route_after_validation(state: Phase1State) -> str:
    if state.get("validation", {}).get("is_valid"):
        return "enrich_visuals"
    if int(state.get("retries") or 0) >= 1:
        # Already retried — accept the structurally-broken result as best-effort.
        return "enrich_visuals"
    return "retry"


def retry_node(state: Phase1State) -> Phase1State:
    log = _log(state, "[Phase1] validation failed — regenerating story")
    return {**state, "retries": int(state.get("retries") or 0) + 1, "log": log}


# --------------------------------------------------------------------------- #
# Graph                                                                        #
# --------------------------------------------------------------------------- #


def build_graph():
    g = StateGraph(Phase1State)
    g.add_node("generate_story", generate_story_node)
    g.add_node("validate_story", validate_story_node)
    g.add_node("retry", retry_node)
    g.add_node("enrich_visuals", enrich_visuals_node)

    g.set_entry_point("generate_story")
    g.add_edge("generate_story", "validate_story")
    g.add_conditional_edges(
        "validate_story",
        route_after_validation,
        {"enrich_visuals": "enrich_visuals", "retry": "retry"},
    )
    g.add_edge("retry", "generate_story")
    g.add_edge("enrich_visuals", END)
    return g.compile()


def run_phase1(prompt: str, num_scenes: int = 3, style: str = "cinematic") -> tuple[Story, list[str]]:
    """
    Run Phase 1 end-to-end.

    Returns the (Story, log) tuple. If validation fails twice and the LLM never
    produced a parseable story, we still return a stub Story so downstream phases
    can run — Phase 1 never raises, it just records errors in the log.
    """
    app = build_graph()
    init: Phase1State = {
        "prompt": prompt,
        "num_scenes": num_scenes,
        "style": style,
        "retries": 0,
    }
    final = app.invoke(init)
    story = final.get("story")
    if story is None:
        # Last-ditch fallback so the pipeline always has a Story.
        story = Story.model_validate(stub_story(prompt, num_scenes))
        final.setdefault("log", []).append("[Phase1] FINAL fallback to stub")
    return story, final.get("log", [])


__all__ = ["run_phase1", "build_graph"]
