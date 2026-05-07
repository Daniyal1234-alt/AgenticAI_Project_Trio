"""Prompt templates used by the three Phase-1 agents.

The pipeline used to live in a single mega-prompt. We split it into three
because the spec calls for three independent agents — Story, Character,
and Script — each owning a distinct part of the schema.
"""

# --------------------------------------------------------------------------- #
# Story agent — produces a StoryOutline (no characters, no dialogue yet).      #
# --------------------------------------------------------------------------- #

STORY_OUTLINE_PROMPT = """You are a senior screenwriter for short-form animated films.
Plan a {num_scenes}-scene story (each scene about {scene_seconds:.0f}s of screen time).

USER PROMPT:
{prompt}

REQUIRED STYLE: {style}

Return ONLY a single JSON object — no prose, no markdown — matching this schema exactly:
{{
  "title": "Short evocative title",
  "logline": "One-sentence pitch",
  "style": "{style}",
  "themes": ["one or two themes — single words"],
  "arc": {{
    "intro": "How the world is set up",
    "rising_action": "What pressure builds",
    "climax": "The turning moment",
    "resolution": "How things land"
  }},
  "scene_outlines": [
    {{
      "scene_number": 1,
      "heading": "INT./EXT. LOCATION — TIME",
      "mood": "single mood word: tense, hopeful, melancholy, joyful, reflective",
      "target_duration_s": {scene_seconds:.1f},
      "summary": "One sentence describing what happens in the scene"
    }}
  ]
}}

Hard rules:
- EXACTLY {num_scenes} scene_outlines.
- The arc must touch all four beats — intro, rising_action, climax, resolution.
- Do NOT invent characters or dialogue here. That is the next agent's job.
"""


# --------------------------------------------------------------------------- #
# Character agent — given a StoryOutline, produces a CharacterRoster.          #
# --------------------------------------------------------------------------- #

CHARACTER_ROSTER_PROMPT = """You are a casting director for an animated short film.
Given the story outline below, design the cast.

STORY OUTLINE:
{outline_json}

Return ONLY a single JSON object — no prose, no markdown — matching this schema:
{{
  "characters": [
    {{
      "name": "Character Name",
      "role": "protagonist | antagonist | narrator | supporting",
      "voice_style": "warm | gravelly | whispered | cheerful | stern | youthful | wise | determined | anxious | serene | neutral",
      "appearance": "1-2 sentence visual description (Phase 3 needs this to draw the character)",
      "personality": "1 sentence personality summary"
    }}
  ]
}}

Hard rules:
- AT LEAST 2 characters.
- EXACTLY one protagonist.
- Pick a `voice_style` from the list above (Phase 2 maps these to TTS voices).
- Names must be unique.
- Appearance is REQUIRED — Phase 3 uses it for image generation.
"""


# --------------------------------------------------------------------------- #
# Script agent — given outline + roster, produces full per-scene scripts.      #
# --------------------------------------------------------------------------- #

SCRIPT_PROMPT = """You are a senior screenwriter writing the final shooting script.
Given the outline and cast below, write each scene in full.

STORY OUTLINE:
{outline_json}

CAST:
{roster_json}

Return ONLY a single JSON object — no prose, no markdown — matching this schema:
{{
  "scenes": [
    {{
      "scene_number": 1,
      "heading": "INT./EXT. LOCATION — TIME",
      "location": "Specific setting",
      "time_of_day": "DAWN | DAY | DUSK | NIGHT",
      "mood": "single mood word matching the outline",
      "duration_seconds": 8.0,
      "characters": ["Name1", "Name2"],
      "action": "What physically happens in the scene",
      "visual_prompt": "Detailed image-generator prompt: subject, setting, lighting, lens, art style",
      "dialogue": [
        {{"character": "Name1", "line": "Opening beat — sets up what's about to happen.", "direction": "(softly)"}},
        {{"character": "Name2", "line": "Reactive beat — pushes back, agrees, or escalates.", "direction": "(determined)"}},
        {{"character": "Name1", "line": "Closing beat — lands the moment for the cut.", "direction": "(quiet)"}}
      ]
    }}
  ]
}}

Hard rules:
- EXACTLY {num_scenes} scenes, in scene_number order 1..N.
- Every dialogue.character MUST be a name from the CAST above (no new characters).
- AIM FOR 2-3 dialogue lines per scene — at least 2 unless the scene is purely
  visual/atmospheric. Use back-and-forth between characters, not monologue.
- Each line should be short (1 sentence, max ~15 words) — short-film pacing,
  not a stage play.
- Vary which character speaks first across scenes; don't make one character
  lead every scene.
- duration_seconds must be between 2 and 60. Aim for {scene_seconds:.0f}s per scene.
- Every scene needs a vivid visual_prompt for the establishing shot.
"""
