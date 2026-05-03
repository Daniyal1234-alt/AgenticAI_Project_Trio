"""Prompt templates used by Phase 1's nodes."""

STORY_PROMPT = """You are a senior screenwriter for short-form animated films.
The user wants a {num_scenes}-scene story (each scene ~8 seconds of screen time).

USER PROMPT:
{prompt}

REQUIRED STYLE: {style}

Return ONLY a single JSON object — no prose, no markdown — matching this schema exactly:
{{
  "title": "Short evocative title",
  "logline": "One-sentence pitch",
  "style": "{style}",
  "characters": [
    {{
      "name": "Character Name",
      "role": "protagonist | antagonist | narrator | supporting",
      "voice_style": "warm | gravelly | whispered | cheerful | stern | youthful | wise",
      "appearance": "1-2 sentence visual description",
      "personality": "1 sentence personality summary"
    }}
  ],
  "scenes": [
    {{
      "scene_number": 1,
      "heading": "INT./EXT. LOCATION — TIME",
      "location": "Specific setting",
      "time_of_day": "DAWN | DAY | DUSK | NIGHT",
      "mood": "single mood word — tense, hopeful, melancholy, etc.",
      "duration_seconds": 8.0,
      "characters": ["Name1"],
      "action": "What physically happens in the scene",
      "visual_prompt": "Detailed image-generator prompt: subject, setting, lighting, lens, art style",
      "dialogue": [
        {{"character": "Name1", "line": "What they say", "direction": "(softly)"}}
      ]
    }}
  ]
}}

Hard rules:
- Generate EXACTLY {num_scenes} scenes.
- The same characters appear consistently across scenes (do not invent new names mid-story).
- Each scene needs at least 1 dialogue line and a vivid `visual_prompt`.
- Total story duration should land near {total_duration:.0f} seconds.
- Keep dialogue tight — short film pacing, not a stage play.
"""


VALIDATE_PROMPT = """You are a JSON-schema validator. Below is a Story JSON.
Check for:
1. Every dialogue.character appears in story.characters[].name
2. Every scene has at least one character and one dialogue line
3. scene_number is contiguous from 1..N

Return ONLY {{"is_valid": true|false, "errors": ["..."], "suggestions": ["..."]}}
No prose, no markdown.

STORY:
{story_json}
"""
