"""
Edit-intent classification agent (LangGraph).

Free-text edit query  →  validated EditIntent.

The graph is a single node — the value here is the LangGraph wrapping +
checkpointer (so multi-turn editing conversations could be plugged in later)
and the strict Pydantic-validated structured output.
"""

from __future__ import annotations

import json
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from schemas.edit import EditIntent
from schemas.llm import chat_json, have_openai, stub_intent


# Curated few-shot examples — ten distinct query types as required by the spec.
FEW_SHOTS = [
    # (query, intent, target, scope, parameters)
    ("Change voice tone for the narrator",          "change_voice_tone",         "audio",       "character:Narrator",  {"tone": "softer"}),
    ("Make scene 2 darker",                          "apply_filter",              "video_frame", "scene:2",             {"filter": "darken"}),
    ("Add background music to the whole video",      "change_bgm",                "audio",       "all",                 {"mood": "epic"}),
    ("Remove the subtitles",                         "remove_subtitle",           "video",       "all",                 {}),
    ("Change character design for the protagonist",  "regenerate_character_frames","video_frame","character:Protagonist",{}),
    ("Speed up scene 3",                             "adjust_speed",              "video",       "scene:3",             {"speed": 1.5}),
    ("Regenerate the entire script",                 "regenerate_script",         "script",      "all",                 {}),
    ("Apply a sepia filter to scene 1",              "apply_filter",              "video_frame", "scene:1",             {"filter": "sepia"}),
    ("Make the narrator sound more cheerful",        "change_voice_tone",         "audio",       "character:Narrator",  {"tone": "cheerful"}),
    ("Add a noir filter and slow down the video",    "apply_filter",              "video_frame", "all",                 {"filter": "noir"}),
]


CLASSIFY_PROMPT = """You are an edit-intent classifier for a video-editing AI agent.
The user has just generated an animated short film and now describes an edit they want.

Classify the request into a structured EditIntent.

VALID TARGETS (pick exactly one):
- audio        → change voices, BGM, mute, re-synthesise dialogue
- video_frame  → change still images, character looks, colour filters, scene aesthetics
- video        → modify the final composite — speed, subtitles, transitions
- script       → rewrite the story or scene script (cascades to all phases)

VALID INTENT LABELS (snake_case action you would call):
- change_voice_tone, change_bgm, mute_audio
- apply_filter (parameters.filter ∈ darken|brighten|sepia|noir|blur|saturate)
- regenerate_character_frames, regenerate_scene_image
- adjust_speed (parameters.speed: float), add_subtitle, remove_subtitle
- regenerate_script

SCOPE format examples:
- "all"              — affects every scene
- "scene:2"          — only scene 2
- "character:Aria"   — only frames or audio for character Aria

Few-shot examples:
{examples}

Now classify this query:
"{query}"

Return ONLY a JSON object:
{{"intent": "...", "target": "...", "scope": "...", "parameters": {{}},
  "confidence": 0.0-1.0, "explanation": "one short sentence"}}
"""


class _State(TypedDict, total=False):
    query: str
    raw: dict
    intent: EditIntent | None
    log: list[str]


def _format_examples() -> str:
    return "\n".join(
        f'- "{q}"  →  {{"intent":"{i}","target":"{t}","scope":"{s}","parameters":{json.dumps(p)}}}'
        for q, i, t, s, p in FEW_SHOTS
    )


def classify_node(state: _State) -> _State:
    query = state.get("query", "")
    log = list(state.get("log") or [])

    if have_openai():
        prompt = CLASSIFY_PROMPT.format(examples=_format_examples(), query=query)
        try:
            raw = chat_json(prompt, temperature=0.0)
            log.append("[EditAgent] LLM classification OK")
        except Exception as exc:
            log.append(f"[EditAgent] LLM error {exc!r} — using rule-based")
            raw = stub_intent(query)
    else:
        log.append("[EditAgent] No OPENAI_API_KEY — using rule-based classifier")
        raw = stub_intent(query)

    if not isinstance(raw, dict):
        raw = stub_intent(query)

    try:
        intent = EditIntent.model_validate(raw)
    except ValidationError as exc:
        log.append(f"[EditAgent] Schema error: {exc.error_count()} — fallback to rule-based")
        intent = EditIntent.model_validate(stub_intent(query))

    return {**state, "raw": raw, "intent": intent, "log": log}


def build_intent_graph():
    g = StateGraph(_State)
    g.add_node("classify", classify_node)
    g.set_entry_point("classify")
    g.add_edge("classify", END)
    # Checkpointer enables multi-turn editing later if we add a follow-up node.
    return g.compile(checkpointer=MemorySaver())


_INTENT_APP = None


def classify_edit_intent(query: str, *, thread_id: str = "default") -> tuple[EditIntent, list[str]]:
    """Public entry — run the classifier for a single edit query."""
    global _INTENT_APP
    if _INTENT_APP is None:
        _INTENT_APP = build_intent_graph()
    config = {"configurable": {"thread_id": thread_id}}
    final = _INTENT_APP.invoke({"query": query, "log": []}, config=config)
    intent = final.get("intent") or EditIntent.model_validate(stub_intent(query))
    return intent, final.get("log", [])
