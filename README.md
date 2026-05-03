# AgenticAI_Project — AI-Powered Animated Video Generation System

> **Course:** Agentic AI · Spring 2026 · National University of Computer & Emerging Sciences, Islamabad
> **Project Type:** Full-stack multi-agent AI system

From a single natural-language prompt, this system writes a story, voices the
characters, generates the visuals, composites a final MP4, and lets you edit
the result by typing **plain-English commands** like *"make scene 2 darker"*
or *"change the narrator's voice to whispered"* — with full version history
and one-click undo at any granularity.

---

## Group

| Member | Roll # | Primary phases |
| --- | --- | --- |
| **Hussain Ali Zaidi** | 22i-0902 | Phase 1 — Story / Script / Character generation |
| **Hamza Ahmed** | 22i-1339 | Phase 2 — Audio (TTS + BGM + timing manifest) · Phase 3 — Video composition |
| **Muhammad Daniyal Aziz** | 22i-0753 | Phase 4 — FastAPI backend + frontend · Phase 5 — Edit Agent + Undo |

The shared JSON schema, integration testing, and the final report are joint
work of all three members.

---

## What it does

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   prompt          Phase 1            Phase 2          Phase 3       │
│   "An astronaut   Story+Script+      TTS + BGM        Per-scene     │
│   discovers a   ─►Characters     ──► + timing      ──► images +     │
│   hidden ocean    (LangGraph)        manifest          MoviePy MP4  │
│   on Mars"                                                          │
│                       │                                             │
│                       ▼                                             │
│                 ┌───────────────────────────────────────┐           │
│                 │ Shared ProjectState (Pydantic)        │           │
│                 │ snapshotted by Phase 5 StateManager   │           │
│                 │ on every change → v1, v2, v3, …       │           │
│                 └───────────────────────────────────────┘           │
│                       │                                             │
│   user: "make         ▼                                             │
│   scene 2       Phase 5: Edit-intent agent                          │
│   darker"   ──► (LangGraph + Pydantic structured output)            │
│                       │                                             │
│                       ▼                                             │
│                 Phase 5 executor → re-runs only what's needed       │
│                       │                                             │
│                       ▼                                             │
│                 New snapshot vN+1 — undoable forever                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick start

```bash
# 1. Install
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. (Optional) Set API keys for richer stories — see .env.example
copy .env.example .env

# 3. Run the web UI
uvicorn phase4_web.api:app --host 127.0.0.1 --port 8000 --reload
# Open http://127.0.0.1:8000/

# 4. Or run the CLI end-to-end
python main.py --prompt "A young astronaut discovers a hidden ocean on Mars"
```

The system **runs without any API keys** — Phase 1 falls back to a deterministic
stub story, Phase 2 uses free Microsoft `edge-tts` voices (no key needed),
Phase 3 ships a stylised PIL placeholder image generator, and Phase 5 uses a
rule-based intent classifier when no LLM is available. Setting `OPENAI_API_KEY`
turns on creative LLM-driven story generation and intelligent edit classification.

---

## CLI

```bash
# End-to-end generation (3 scenes, cinematic style)
python main.py --prompt "A robot tutor on Mars"

# Custom scene count + style
python main.py --prompt "A noir detective" --scenes 5 --style noir

# Apply an edit to an existing project
python main.py --edit "Apply a sepia filter to scene 1" --project 20260503-XXXXXXXX

# Revert to a previous version
python main.py --revert 1 --project 20260503-XXXXXXXX
```

---

## Web UI (Phase 4)

The web interface (`phase4_web/`) walks through the pipeline with live
WebSocket progress:

1. **Create a project** — prompt + scene count + style.
2. **Pipeline progress** — phase strip + streaming log over WebSocket.
3. **Preview** — embedded `<video>` player, plus tabs for story / characters / scenes / audio manifest JSON.
4. **Version history** — every generation and every edit is listed with timestamps and a *revert* button.
5. **Edit agent** — free-text input + one-click chips for common edits (e.g. *"scene 1 darker"*, *"sepia all"*, *"1.5× speed"*, *"remove subs"*).

---

## Project layout

```
.
├── main.py                          # CLI entry point
├── README.md  ·  requirements.txt  ·  .env.example
│
├── schemas/                         # The contract every phase reads/writes
│   ├── pipeline.py                    – Story, Scene, Character, AudioSegment,
│   │                                    TimingManifest, SceneVideo, ProjectState
│   ├── edit.py                        – EditIntent, EditTarget, VersionEntry
│   └── llm.py                         – Shared LLM helper + offline stubs
│
├── phase1_story/                    # Member 1
│   ├── agent.py                       – LangGraph: generate → validate → enrich
│   └── prompts.py                     – Few-shot prompt templates
│
├── phase2_audio/                    # Member 2
│   ├── tts.py                         – Edge-TTS (free) + silent-WAV fallback
│   ├── bgm.py                         – Procedural mood-tinted BGM (numpy/wave)
│   └── pipeline.py                    – Walks Story → AudioSegment[] + TimingManifest
│
├── phase3_video/                    # Member 2 (shared)
│   ├── image_gen.py                   – DALL-E 3 (optional) or PIL placeholder
│   ├── compositor.py                  – MoviePy v1/v2 + ffmpeg + slideshow fallback
│   └── pipeline.py                    – Per-scene image + final MP4
│
├── phase4_web/                      # Member 3
│   ├── api.py                         – FastAPI: REST + WebSocket progress
│   ├── orchestrator.py                – Wires phases 1→2→3 sequentially
│   └── static/  index.html · styles.css · app.js
│
├── phase5_edit/                     # Member 3
│   ├── intent_agent.py                – LangGraph + checkpointer + 10 few-shots
│   ├── executor.py                    – Maps EditIntent → targeted phase re-runs
│   └── state_manager.py               – Append-only versioned snapshots, full undo
│
├── tests/                           # 29 tests, all offline-runnable
│   ├── conftest.py
│   ├── test_phase1_story.py            (3 tests)
│   ├── test_phase2_audio.py            (5 tests)
│   ├── test_phase3_video.py            (3 tests)
│   ├── test_phase4_api.py              (3 tests)
│   └── test_phase5_edit.py             (15 tests — incl. 11 distinct query types)
│
├── outputs/projects/<project_id>/   # Generated artefacts (one folder per project)
│   ├── current/                       – live state (state.json + audio/ + images/ + final_output.mp4)
│   ├── v1/  v2/  v3/  …               – snapshots, copy of current/ at the time
│   └── history.json                   – list[VersionEntry]
│
├── assets/bgm/                      # Drop *.mp3 named tense.mp3 / hopeful.mp3 / …
│                                      to override the procedural BGM.
└── docs/architecture.md             # Detailed phase-by-phase write-up
```

---

## The shared JSON schema (the most important file)

`schemas/pipeline.py` defines the single source of truth. Every phase consumes
it, validates it via Pydantic v2, and hands it to the next phase:

```python
ProjectState
├── project_id : str           # ULID-ish
├── version    : int           # bumped on every snapshot
├── phase      : "created" | "phase1" | "phase2" | "phase3" | "complete" | "edited"
├── story      : Story
│   ├── title, logline, style
│   ├── characters[]: Character (name, role, voice_style, voice_id, appearance)
│   └── scenes[]    : Scene (number, heading, location, time_of_day, mood,
│                            duration_seconds, characters[], action,
│                            visual_prompt, dialogue[])
├── timing      : TimingManifest
│   └── segments[]: AudioSegment (scene_id, kind=dialogue|bgm,
│                                  character?, audio_file,
│                                  start_ms, end_ms, text)
├── scene_videos[]: SceneVideo (scene_id, image_file, duration_seconds, animation)
├── final_video : str | None
├── edit_log    : list[dict]   # every accepted EditIntent
└── notes       : list[str]
```

This is the contract. Every phase boundary tests round-trip through this schema.

---

## Phase-by-phase highlights

### Phase 1 — Story, Script, Character (Member 1)
- **LangGraph** StateGraph: `generate_story → validate_story → (retry once if invalid) → enrich_visuals`.
- Pydantic-validated structured output: structural errors caught locally, semantic errors (speaker not in roster, non-contiguous scene numbers) flagged per scene.
- Every scene leaves with a `visual_prompt` ready for Phase 3.
- Graceful offline fallback: `schemas/llm.py:stub_story` produces a complete schema-valid story when no API key is set.

### Phase 2 — Audio (Member 2)
- **Free TTS:** `edge-tts` ships zero-config Microsoft Edge voices over the network, no key required.
- Per-character voice resolution from `voice_style` and `role` heuristics.
- Per-scene **procedural BGM** generated with `numpy` + `wave` (mood → fundamental frequency); drop your own `.mp3` into `assets/bgm/<mood>.mp3` to override.
- **Timing manifest** with `start_ms` / `end_ms` per segment is the contract Phase 3 consumes.

### Phase 3 — Video (Member 2)
- **PIL placeholder renderer** ships out-of-the-box: vertical mood gradient + character silhouettes + vignette + scene-heading title card. Zero API calls, deterministic per scene.
- **DALL-E 3** path activated by `IMAGE_BACKEND=openai` in `.env`.
- **MoviePy** compositor supports both v1.x (`moviepy.editor`) and v2.x (top-level imports) — the compositor auto-detects.
- Falls back to direct `ffmpeg` subprocess if MoviePy fails, then to a slideshow PNG sidecar if both are missing — the pipeline always produces *something*.
- Per-scene Ken-Burns zoom, optional caption burn-in, BGM ducking under dialogue.

### Phase 4 — Web Interface (Member 3)
- `FastAPI` with REST + a per-project **WebSocket** (`/api/projects/{id}/ws`) for streaming progress.
- Single-page frontend (`phase4_web/static/`) with a four-card layout: create / progress / edit / preview / version history.
- Past projects sidebar with one-click reopen.
- Safe path resolution on every file route (no traversal escape).

### Phase 5 — Edit Agent + Undo (Member 3)
- **LangGraph** classifier with `MemorySaver` checkpointer keyed by `project_id` (multi-turn editing conversations are one node away).
- 10+ few-shot examples covering every target type: `audio`, `video_frame`, `video`, `script`.
- Structured Pydantic `EditIntent` with `intent` / `target` / `scope` / `parameters` / `confidence`.
- **StateManager** = append-only filesystem snapshots:
  - `snapshot()` copies `current/` → `vN/` and appends `VersionEntry`.
  - `revert(N)` restores `current/` from `vN/` *and* records the revert as a new version (so undo is itself undoable).
- **Smart re-runs:** an `apply_filter` edit only mutates the affected images and recomposites; an `audio` edit only re-runs Phase 2 + recomposites; a `script` edit cascades through every phase.

---

## Testing

```bash
pytest                    # full suite — 29 tests, all offline
pytest tests/test_phase5_edit.py -v   # 15 tests, includes 11 distinct edit-query types
```

The test suite **deliberately deletes `OPENAI_API_KEY` from the environment**
in `tests/conftest.py` so the offline / fallback paths are what gets graded.

| Phase | Tests | What's covered |
| --- | --- | --- |
| 1 | 3 | Story shape · contiguous scene numbers · speaker-in-roster invariant |
| 2 | 5 | TTS voice mapping · BGM file generation · manifest segments per scene · duration realignment |
| 3 | 3 | PNG generation · darken filter changes pixels · `final_output.mp4` always exists |
| 4 | 3 | `/api/health` · empty-project list · 404 for unknown project |
| 5 | 15 | 11 query-type classifications · scope+param extraction · snapshot+revert+history |

---

## Sample run

A real run produced this layout (commit-tracked under `outputs/projects/`):

```
20260503-162725-35e62336/
├── current/
│   ├── state.json
│   ├── audio/   scene01_line01.mp3 · scene01_bgm.wav · …
│   ├── images/  scene01.png · scene02.png · scene03.png
│   └── final_output.mp4
├── v1/  (Initial generation)
├── v2/  (Edit: Make scene 2 darker)
├── v3/  (Edit: Apply a sepia filter to scene 1)
├── v4/  (Reverted to v1)
└── history.json
```

---

## Tech stack

| Layer | Tool | Why |
| --- | --- | --- |
| Agent runtime | **LangGraph** | StateGraph + checkpointer for both Phase 1 and Phase 5 |
| LLM | OpenAI `gpt-4o-mini` (optional) | Phase 1 stories + Phase 5 intent classification |
| Schema | **Pydantic v2** | Validates every phase boundary |
| TTS | `edge-tts` | Free, no API key, async, decent voice quality |
| BGM | `numpy` + `wave` | Procedural pads — no licensing concerns |
| Image gen | `Pillow` (default) / DALL-E 3 (optional) | Works offline, works in CI |
| Video | `MoviePy` 1.x **or** 2.x → `ffmpeg` → slideshow | Three-tier fallback |
| Backend | **FastAPI** + `uvicorn` | Fast, type-safe, native WebSocket |
| Frontend | Hand-written HTML/CSS/JS | No build step, no Node dependency |
| State store | Append-only filesystem snapshots | Simple, durable, transparent in `outputs/` |
| Tests | `pytest` + `pytest-asyncio` | 29 tests, all offline |

---

## Design decisions worth flagging

- **Why filesystem snapshots instead of SQLite?** Every snapshot has to include
  binary assets (audio, images, MP4). Storing those as blobs in SQLite buys
  nothing — the filesystem is already the simplest, most transparent
  versioned store. `history.json` is the only metadata file, and it's an
  append-only list so concurrent writes are trivial.
- **Why graceful degradation everywhere?** A grader running this on a fresh
  laptop with no API keys, no ffmpeg in PATH, and no internet should still
  see *something* end-to-end. Every phase has a documented fallback that
  lets the pipeline complete and produces schema-valid artefacts.
- **Why per-character `voice_id` resolution lives in Phase 2?** Keeping it
  there means Phase 1 stays purely creative — it picks `voice_style: "warm"`
  and Phase 2 maps that to a concrete TTS voice. An edit that changes the
  voice style (Phase 5) just clears `voice_id` and re-runs Phase 2.
- **Why edits with `scope_filter=set()` reuse images?** A filter edit (e.g.
  "make scene 2 darker") mutates the image *in place* — re-running the
  generator afterwards would overwrite the filter. `scope_filter=set()` tells
  Phase 3 to recomposite the MP4 without regenerating any images.

---

## License

Educational / academic use, FAST-NUCES. All generated assets are produced
locally; no proprietary content is shipped in this repository.
