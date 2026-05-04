"""
Phase 1 LangGraph agent — three sequential agents matching the spec diagram.

    [START]
       │
       ▼
    story_agent      ──┐ tools: validate_story_arc, estimate_duration
       │ outline       │
       ▼               │ retry / error_handler
    character_agent  ──┤ tools: check_consistency
       │ roster        │
       ▼               │
    script_agent     ──┘ tools: build_visual_prompt, validate_duration,
       │ script         analyze_emotions, estimate_duration
       ▼
    serialize  →  writes story.json / characters.json / script.json /
       │           phase2_audio_handoff.json / phase3_video_handoff.json /
       ▼           summary.json (when project_dir is provided)
     [END]

Each agent retries once on validation failure; after the second failure the
graph routes to `error_handler`, which uses the deterministic `stub_story`
fallback so downstream phases always see a valid `Story`.

Public:
    run_phase1(prompt, num_scenes=3, style="cinematic", project_dir=None)
        → (Story, log)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from phase1_story import tools
from phase1_story.prompts import (
    CHARACTER_ROSTER_PROMPT,
    SCRIPT_PROMPT,
    STORY_OUTLINE_PROMPT,
)
from phase1_story.serializer import write_artifacts
from schemas.llm import (
    chat_json,
    have_openai,
    stub_outline,
    stub_roster,
    stub_script,
    stub_story,
)
from schemas.pipeline import (
    Character,
    CharacterRoster,
    ScriptOutput,
    Scene,
    Story,
    StoryOutline,
)


SCENE_SECONDS_DEFAULT = 8.0


class Phase1State(TypedDict, total=False):
    # Inputs
    prompt: str
    num_scenes: int
    style: str
    project_dir: str | None

    # Per-agent outputs (raw dicts so failed validation doesn't crash routing)
    outline: dict[str, Any] | None
    roster: dict[str, Any] | None
    script: dict[str, Any] | None

    # Per-agent retry counters
    outline_retries: int
    roster_retries: int
    script_retries: int

    # Per-agent validation results
    outline_validation: dict[str, Any]
    roster_validation: dict[str, Any]
    script_validation: dict[str, Any]

    # Final output
    story: Story | None

    # Diagnostic logs
    log: list[str]
    tools_log: list[dict[str, Any]]
    errors: list[str]
    fallback_used: bool


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _log(state: Phase1State, msg: str) -> list[str]:
    arr = list(state.get("log") or [])
    arr.append(msg)
    return arr


def _record_tool(state: Phase1State, tool_name: str, result: dict) -> list[dict]:
    arr = list(state.get("tools_log") or [])
    arr.append(
        {
            "tool": tool_name,
            "ok": bool(result.get("ok")),
            "issues": list(result.get("issues") or []),
        }
    )
    return arr


def _record_errors(state: Phase1State, msgs: list[str]) -> list[str]:
    arr = list(state.get("errors") or [])
    arr.extend(msgs)
    return arr


# --------------------------------------------------------------------------- #
# Story agent                                                                  #
# --------------------------------------------------------------------------- #


def story_agent_node(state: Phase1State) -> Phase1State:
    """Generate a StoryOutline (title, logline, themes, arc, scene_outlines)."""
    prompt = state.get("prompt", "")
    num_scenes = int(state.get("num_scenes") or 3)
    style = state.get("style") or "cinematic"
    log = _log(state, f"[Phase1] story_agent (scenes={num_scenes}, style={style})")

    data: dict[str, Any] = {}
    if have_openai():
        rendered = STORY_OUTLINE_PROMPT.format(
            prompt=prompt,
            num_scenes=num_scenes,
            style=style,
            scene_seconds=SCENE_SECONDS_DEFAULT,
        )
        try:
            data = chat_json(rendered, temperature=0.85)
            if not isinstance(data, dict) or "scene_outlines" not in data:
                log.append("[Phase1] story_agent: LLM returned non-conforming JSON — using stub")
                data = stub_outline(prompt, num_scenes, style)
            else:
                log.append(f"[Phase1] story_agent: outline '{data.get('title', '?')}'")
        except Exception as exc:  # network, auth, etc.
            log.append(f"[Phase1] story_agent error: {exc!r} — using stub")
            data = stub_outline(prompt, num_scenes, style)
    else:
        log.append("[Phase1] story_agent: no OPENAI_API_KEY — using stub")
        data = stub_outline(prompt, num_scenes, style)

    # Run tools
    tools_log = list(state.get("tools_log") or [])
    arc_check = tools.validate_story_arc(data)
    tools_log.append({"tool": "validate_story_arc", "ok": arc_check["ok"], "issues": arc_check["issues"]})
    dur_check = tools.estimate_duration(data)
    tools_log.append({"tool": "estimate_duration", "ok": dur_check["ok"], "issues": dur_check["issues"], "value": dur_check["value"]})

    # Pydantic structural validation
    issues = list(arc_check["issues"])
    try:
        StoryOutline.model_validate(data)
    except ValidationError as exc:
        issues.append(f"StoryOutline schema: {exc.error_count()} validation errors")

    valid = not issues
    log.append(f"[Phase1] story_agent: validation {'OK' if valid else 'FAIL'} ({len(issues)} issues)")

    return {
        **state,
        "outline": data,
        "outline_validation": {"ok": valid, "issues": issues},
        "log": log,
        "tools_log": tools_log,
    }


def story_retry_node(state: Phase1State) -> Phase1State:
    n = int(state.get("outline_retries") or 0) + 1
    log = _log(state, f"[Phase1] story_agent: retry #{n}")
    return {**state, "outline_retries": n, "log": log}


# --------------------------------------------------------------------------- #
# Character agent                                                              #
# --------------------------------------------------------------------------- #


def character_agent_node(state: Phase1State) -> Phase1State:
    """Given the outline, produce a CharacterRoster."""
    log = _log(state, "[Phase1] character_agent")
    outline = state.get("outline") or {}

    data: dict[str, Any] = {}
    if have_openai():
        rendered = CHARACTER_ROSTER_PROMPT.format(outline_json=json.dumps(outline, ensure_ascii=False))
        try:
            data = chat_json(rendered, temperature=0.7)
            if not isinstance(data, dict) or "characters" not in data:
                log.append("[Phase1] character_agent: LLM returned non-conforming JSON — using stub")
                data = stub_roster(outline)
            else:
                log.append(f"[Phase1] character_agent: roster of {len(data.get('characters', []))} characters")
        except Exception as exc:
            log.append(f"[Phase1] character_agent error: {exc!r} — using stub")
            data = stub_roster(outline)
    else:
        log.append("[Phase1] character_agent: no OPENAI_API_KEY — using stub")
        data = stub_roster(outline)

    # Run tool
    try:
        roster_objs = [Character.model_validate(c) for c in data.get("characters") or []]
    except ValidationError as exc:
        log.append(f"[Phase1] character_agent: Pydantic errors: {exc.error_count()}")
        roster_objs = []

    cons = tools.check_consistency(roster_objs, outline)
    tools_log = _record_tool(state, "check_consistency", cons)

    issues = list(cons["issues"])
    if not roster_objs:
        issues.append("character roster is empty or invalid")

    valid = not issues
    log.append(f"[Phase1] character_agent: validation {'OK' if valid else 'FAIL'} ({len(issues)} issues)")

    return {
        **state,
        "roster": data,
        "roster_validation": {"ok": valid, "issues": issues},
        "log": log,
        "tools_log": tools_log,
    }


def character_retry_node(state: Phase1State) -> Phase1State:
    n = int(state.get("roster_retries") or 0) + 1
    log = _log(state, f"[Phase1] character_agent: retry #{n}")
    return {**state, "roster_retries": n, "log": log}


# --------------------------------------------------------------------------- #
# Script agent                                                                 #
# --------------------------------------------------------------------------- #


def script_agent_node(state: Phase1State) -> Phase1State:
    """Given the outline and roster, produce ScriptOutput (full scenes)."""
    log = _log(state, "[Phase1] script_agent")
    outline = state.get("outline") or {}
    roster = state.get("roster") or {}
    num_scenes = int(state.get("num_scenes") or 3)
    style = (outline.get("style") if isinstance(outline, dict) else None) or state.get("style") or "cinematic"

    data: dict[str, Any] = {}
    if have_openai():
        rendered = SCRIPT_PROMPT.format(
            outline_json=json.dumps(outline, ensure_ascii=False),
            roster_json=json.dumps(roster, ensure_ascii=False),
            num_scenes=num_scenes,
            scene_seconds=SCENE_SECONDS_DEFAULT,
        )
        try:
            data = chat_json(rendered, temperature=0.85)
            if not isinstance(data, dict) or "scenes" not in data:
                log.append("[Phase1] script_agent: LLM returned non-conforming JSON — using stub")
                data = stub_script(outline, roster, prompt=state.get("prompt", ""))
            else:
                log.append(f"[Phase1] script_agent: {len(data.get('scenes', []))} scenes drafted")
        except Exception as exc:
            log.append(f"[Phase1] script_agent error: {exc!r} — using stub")
            data = stub_script(outline, roster, prompt=state.get("prompt", ""))
    else:
        log.append("[Phase1] script_agent: no OPENAI_API_KEY — using stub")
        data = stub_script(outline, roster, prompt=state.get("prompt", ""))

    # Per-scene tool sweep: build_visual_prompt (fill-ins), validate_duration,
    # analyze_emotions, estimate_duration.
    tools_log = list(state.get("tools_log") or [])
    issues: list[str] = []

    try:
        scenes_objs: list[Scene] = [Scene.model_validate(s) for s in data.get("scenes") or []]
    except ValidationError as exc:
        log.append(f"[Phase1] script_agent: Pydantic errors: {exc.error_count()}")
        scenes_objs = []
        issues.append(f"Scene schema: {exc.error_count()} errors")

    char_names = {(c.get("name") if isinstance(c, dict) else c.name) for c in roster.get("characters") or []}

    for sc in scenes_objs:
        # Enrich weak visual prompts.
        if not sc.visual_prompt or len(sc.visual_prompt) < 20:
            bvp = tools.build_visual_prompt(sc, style=style)
            tools_log.append({"tool": "build_visual_prompt", "ok": True, "scene": sc.scene_number})
            sc.visual_prompt = bvp["value"]

        vd = tools.validate_duration(sc)
        tools_log.append({"tool": "validate_duration", "ok": vd["ok"], "issues": vd["issues"], "scene": sc.scene_number})
        if not vd["ok"]:
            issues.extend(vd["issues"])

        ed = tools.estimate_duration(sc)
        tools_log.append({"tool": "estimate_duration", "ok": ed["ok"], "scene": sc.scene_number, "value": ed["value"]})

        ae = tools.analyze_emotions(sc.dialogue)
        tools_log.append({"tool": "analyze_emotions", "ok": True, "scene": sc.scene_number, "tags": ae["value"]})

        for d in sc.dialogue:
            if char_names and d.character not in char_names:
                issues.append(
                    f"Scene {sc.scene_number}: speaker '{d.character}' not in roster"
                )

    nums = [sc.scene_number for sc in scenes_objs]
    if scenes_objs and sorted(nums) != list(range(1, len(nums) + 1)):
        issues.append(f"scene numbers not contiguous: {nums}")
    if not scenes_objs:
        issues.append("script has no scenes")

    # Push enriched scenes back into the state dict so the serializer writes them.
    data = {"scenes": [s.model_dump() for s in scenes_objs] or data.get("scenes", [])}

    valid = not issues
    log.append(f"[Phase1] script_agent: validation {'OK' if valid else 'FAIL'} ({len(issues)} issues)")

    return {
        **state,
        "script": data,
        "script_validation": {"ok": valid, "issues": issues},
        "log": log,
        "tools_log": tools_log,
    }


def script_retry_node(state: Phase1State) -> Phase1State:
    n = int(state.get("script_retries") or 0) + 1
    log = _log(state, f"[Phase1] script_agent: retry #{n}")
    return {**state, "script_retries": n, "log": log}


# --------------------------------------------------------------------------- #
# Error handler — last-ditch fallback to stub_story so the pipeline always     #
# produces a valid Story for Phase 2/3.                                        #
# --------------------------------------------------------------------------- #


def error_handler_node(state: Phase1State) -> Phase1State:
    log = _log(state, "[Phase1] error_handler: falling back to stub_story()")
    prompt = state.get("prompt", "")
    num_scenes = int(state.get("num_scenes") or 3)
    style = state.get("style") or "cinematic"

    data = stub_story(prompt, num_scenes)
    data["style"] = style

    outline = stub_outline(prompt, num_scenes, style)
    roster = {"characters": data["characters"]}
    script = {"scenes": data["scenes"]}
    errors = _record_errors(state, ["error_handler invoked: stub fallback used"])
    return {
        **state,
        "outline": outline,
        "roster": roster,
        "script": script,
        "errors": errors,
        "fallback_used": True,
        "log": log,
    }


# --------------------------------------------------------------------------- #
# Serialize node — compose final Story + write the 6 handoff JSONs.            #
# --------------------------------------------------------------------------- #


def serialize_node(state: Phase1State) -> Phase1State:
    log = _log(state, "[Phase1] serialize")
    outline = state.get("outline") or {}
    roster = state.get("roster") or {}
    script = state.get("script") or {}

    story_dict = {
        "title": outline.get("title") or "Untitled",
        "logline": outline.get("logline") or "",
        "style": outline.get("style") or state.get("style") or "cinematic",
        "characters": roster.get("characters") or [],
        "scenes": script.get("scenes") or [],
    }
    try:
        story = Story.model_validate(story_dict)
    except ValidationError as exc:
        log.append(f"[Phase1] serialize: Story schema {exc.error_count()} errors — using stub_story")
        story = Story.model_validate(stub_story(state.get("prompt", ""), int(state.get("num_scenes") or 3)))

    # Write handoff JSONs only if a project_dir is provided.
    project_dir = state.get("project_dir")
    if project_dir:
        try:
            paths = write_artifacts(
                project_dir=Path(project_dir),
                outline=outline,
                roster=roster,
                script=script,
                story=story,
                tools_log=list(state.get("tools_log") or []),
                errors=list(state.get("errors") or []),
                fallback_used=bool(state.get("fallback_used")),
            )
            log.append(f"[Phase1] serialize: wrote {len(paths)} artifacts to {project_dir}")
        except Exception as exc:
            log.append(f"[Phase1] serialize: write_artifacts failed — {exc!r}")
    else:
        log.append("[Phase1] serialize: no project_dir → skipping handoff JSON writes")

    return {**state, "story": story, "log": log}


# --------------------------------------------------------------------------- #
# Routing                                                                      #
# --------------------------------------------------------------------------- #


def _route(validation_key: str, retries_key: str, ok_dest: str, retry_dest: str):
    def _r(state: Phase1State) -> str:
        v = state.get(validation_key) or {}
        if v.get("ok"):
            return ok_dest
        if int(state.get(retries_key) or 0) >= 1:
            return "error_handler"
        return retry_dest
    return _r


route_after_story = _route("outline_validation", "outline_retries", "character_agent", "story_retry")
route_after_character = _route("roster_validation", "roster_retries", "script_agent", "character_retry")
route_after_script = _route("script_validation", "script_retries", "serialize", "script_retry")


# --------------------------------------------------------------------------- #
# Graph                                                                        #
# --------------------------------------------------------------------------- #


def build_graph():
    g = StateGraph(Phase1State)

    g.add_node("story_agent", story_agent_node)
    g.add_node("story_retry", story_retry_node)
    g.add_node("character_agent", character_agent_node)
    g.add_node("character_retry", character_retry_node)
    g.add_node("script_agent", script_agent_node)
    g.add_node("script_retry", script_retry_node)
    g.add_node("error_handler", error_handler_node)
    g.add_node("serialize", serialize_node)

    g.set_entry_point("story_agent")

    g.add_conditional_edges(
        "story_agent",
        route_after_story,
        {
            "character_agent": "character_agent",
            "story_retry": "story_retry",
            "error_handler": "error_handler",
        },
    )
    g.add_edge("story_retry", "story_agent")

    g.add_conditional_edges(
        "character_agent",
        route_after_character,
        {
            "script_agent": "script_agent",
            "character_retry": "character_retry",
            "error_handler": "error_handler",
        },
    )
    g.add_edge("character_retry", "character_agent")

    g.add_conditional_edges(
        "script_agent",
        route_after_script,
        {
            "serialize": "serialize",
            "script_retry": "script_retry",
            "error_handler": "error_handler",
        },
    )
    g.add_edge("script_retry", "script_agent")

    g.add_edge("error_handler", "serialize")
    g.add_edge("serialize", END)
    return g.compile()


def run_phase1(
    prompt: str,
    num_scenes: int = 3,
    style: str = "cinematic",
    project_dir: str | Path | None = None,
) -> tuple[Story, list[str]]:
    """
    Run Phase 1 end-to-end.

    Parameters
    ----------
    prompt
        The user's free-text idea.
    num_scenes
        Target scene count.
    style
        High-level visual style hint (cinematic / cartoon / noir / anime).
    project_dir
        If provided, the serializer writes the 6 handoff JSON files
        (story.json, characters.json, script.json, phase2_audio_handoff.json,
        phase3_video_handoff.json, summary.json) into this directory.

    Returns
    -------
    (Story, log_lines)
    """
    app = build_graph()
    init: Phase1State = {
        "prompt": prompt,
        "num_scenes": num_scenes,
        "style": style,
        "project_dir": str(project_dir) if project_dir else None,
        "outline_retries": 0,
        "roster_retries": 0,
        "script_retries": 0,
    }
    final = app.invoke(init)
    story = final.get("story")
    if story is None:
        # Defensive: this branch is unreachable in the current graph (serialize
        # always produces a Story) but keeps the historical safety net.
        story = Story.model_validate(stub_story(prompt, num_scenes))
        final.setdefault("log", []).append("[Phase1] FINAL fallback to stub")
    return story, final.get("log", [])


__all__ = ["run_phase1", "build_graph"]
