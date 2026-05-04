# AgenticAI_Project — AI-Powered Animated Video Generation System

> **Course:** Agentic AI · Spring 2026 · National University of Computer & Emerging Sciences, Islamabad
> **Project Type:** Full-stack multi-agent AI system

From a single natural-language prompt, this system writes a story, casts
characters, voices each line with emotion-aware TTS, generates per-scene
visuals, **lip-syncs** mouth motion to the dialogue, composites a final MP4,
and lets you edit the result by typing **plain-English commands** like
*"make scene 2 darker"* or *"change the narrator's voice to whispered"* —
with full version history and one-click undo at any granularity.

---

## Group

| Member | Roll # | Primary phases |
| --- | --- | --- |
| **Muhammad Daniyal Aziz** | 22i-0753 | Phase 1 — Story / Script / Character generation |
| **Hamza Ahmed** | 22i-1339 | Phase 2 — Audio (TTS + BGM + timing manifest) · Phase 3 — Video composition |
| **Hussain Ali Zaidi** | 22i-0902 | Phase 4 — FastAPI backend + frontend · Phase 5 — Edit Agent + Undo |
| **Saifullah Khalid** | 22i-1312 | Project report |

The shared JSON schema and integration testing are joint work of all four members.

---

## What it does

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   prompt        Phase 1          Phase 2         Phase 3            │
│   "An astro-    Story → Char     edge-tts +      SDXL still →       │
│   naut...    ─► → Script         emotion         SVD motion →       │
│                 (3 LangGraph     prosody +       Wav2Lip lip-       │
│                  agents +        BGM +           sync + Ken Burns   │
│                  6 named tools)  timing.json     + xfade → MP4      │
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

The heavy Phase 3 models (SDXL · SVD · Wav2Lip) run **remotely on a Kaggle
GPU notebook** exposed via ngrok — opt-in. Without an endpoint set, Phase 3
falls through to a stylised PIL placeholder + FFmpeg passthrough so the
pipeline always produces a valid (less rich) MP4.

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

The system **runs without any API keys** — Phase 1 falls through three
agents to a deterministic stub story, Phase 2 uses free Microsoft `edge-tts`
voices (no key needed) with a runtime voice-fallback safety net, Phase 3
ships a stylised PIL placeholder + FFmpeg passthrough lip-sync, and Phase 5
uses a rule-based intent classifier when no LLM is available. Setting
`OPENAI_API_KEY` turns on creative LLM-driven story generation and
intelligent edit classification.

### Phase 3 rich path (optional)

For real per-scene visuals (Stable Diffusion stills + Stable Video Diffusion
ambient motion + Wav2Lip lip-sync), spin up the Kaggle inference server —
see [`docs/kaggle_setup.md`](docs/kaggle_setup.md). Paste
[`docs/kaggle_phase3_server.py`](docs/kaggle_phase3_server.py) into a Kaggle
GPU notebook, copy the printed ngrok URL into your local `.env` as
`KAGGLE_ENDPOINT=...`, and Phase 3 starts hitting the remote endpoints
automatically. Without it, Phase 3 falls back to FFmpeg passthrough clips
(static face + audio, no mouth motion) — a valid MP4 either way.

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
│   ├── agent.py                       – 3-node LangGraph: story → character → script
│   │                                    + serialize, with retry + error_handler
│   ├── prompts.py                     – Per-agent prompt templates (3 of them)
│   ├── tools.py                       – 6 named tools: validate_story_arc,
│   │                                    estimate_duration, check_consistency,
│   │                                    build_visual_prompt, validate_duration,
│   │                                    analyze_emotions
│   └── serializer.py                  – Writes 6 handoff JSONs into current/:
│                                        story · characters · script ·
│                                        phase2_audio_handoff · phase3_video_handoff
│                                        · summary
│
├── phase2_audio/                    # Member 2
│   ├── tts.py                         – Edge-TTS (free) + 3-attempt retry +
│   │                                    runtime fallback voice on rejection +
│   │                                    EMOTION_PROSODY (rate/pitch per emotion)
│   ├── bgm.py                         – Procedural mood-tinted BGM (numpy/wave)
│   └── pipeline.py                    – Walks Story → AudioSegment[] +
│                                        writes timing_manifest.json + reads
│                                        Phase 1's audio handoff for emotion tags
│
├── phase3_video/                    # Member 2 (shared)
│   ├── image_gen.py                   – SDXL-Turbo (optional) → DALL-E 3 → PIL fallback
│   ├── animation.py                   – FFmpeg zoompan Ken Burns clips
│   ├── svd.py                         – Stable Video Diffusion remote client
│   ├── lipsync.py                     – Wav2Lip remote client + passthrough
│   ├── compositor.py                  – Line-clip stitching, BGM mix, drawtext subs, xfade
│   ├── _http.py                       – shared HTTP helper for Kaggle endpoints
│   └── pipeline.py                    – Per-scene+per-line orchestration → final MP4
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
├── tests/                           # 46 tests, all offline-runnable
│   ├── conftest.py                    – pops OPENAI_API_KEY + KAGGLE_ENDPOINT
│   ├── test_phase1_story.py            (11 tests — 3 e2e + 7 tools + serializer)
│   ├── test_phase2_audio.py            (9 tests — TTS + BGM + manifest + prosody)
│   ├── test_phase3_video.py            (8 tests — image + ken_burns + svd + lipsync)
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
└── docs/
    ├── architecture.md              # Detailed phase-by-phase write-up
    ├── kaggle_setup.md              # Per-session runbook for the Kaggle server
    └── kaggle_phase3_server.py      # Paste-into-Kaggle FastAPI app:
                                       SDXL + SVD + Wav2Lip behind one ngrok tunnel
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
- **Three-agent LangGraph** matching the spec diagram:
  `story_agent → character_agent → script_agent → serialize`, each with its
  own retry node and a shared `error_handler` that falls through to the
  deterministic stub when the LLM keeps failing.
- **Six named tool functions** in [`phase1_story/tools.py`](phase1_story/tools.py) —
  `validate_story_arc`, `estimate_duration`, `check_consistency`,
  `build_visual_prompt`, `validate_duration`, `analyze_emotions`. Pure
  Python, deterministic, individually unit-tested.
- **Six handoff JSONs** written by [`phase1_story/serializer.py`](phase1_story/serializer.py)
  into `current/`: `story.json`, `characters.json`, `script.json`,
  `phase2_audio_handoff.json` (per-line emotion tags + voice configs),
  `phase3_video_handoff.json` (visual prompts + camera + transitions),
  `summary.json` (run status + tool log + artifact paths).
- Graceful offline fallback chain: each agent has its own stub
  (`stub_outline`, `stub_roster`, `stub_script`); after exhausted retries
  the error handler reuses the original `stub_story` so the pipeline
  always produces a schema-valid `Story`.

### Phase 2 — Audio (Member 2)
- **Free TTS:** `edge-tts` ships zero-config Microsoft Edge voices over the
  network, no API key required. Wrapped with **3-attempt retries +
  exponential backoff** for transient drops.
- **Runtime voice fallback:** if Microsoft retires a mapped voice (we hit
  `en-US-DavisNeural` mid-project), `synthesize()` automatically retries
  with a known-good voice (`en-US-AriaNeural`) and logs the swap to stderr
  so the next maintainer sees what happened.
- **Emotion-driven prosody:** Phase 1's `analyze_emotions` tool tags every
  dialogue line; Phase 2 reads that handoff and passes per-line `rate` /
  `pitch` to edge-tts. A 7-bucket `EMOTION_PROSODY` table maps
  `tense / urgent / joyful / melancholy / curious / determined / calm`
  to concrete edge-tts kwargs.
- Per-scene **procedural BGM** generated with `numpy` + `wave`
  (mood → fundamental frequency); drop your own `.mp3` into
  `assets/bgm/<mood>.mp3` to override.
- **`timing_manifest.json` is written to disk as a standalone file** —
  the spec calls this artifact out by name. Each segment carries
  `scene_id`, `audio_file`, `start_ms`, `end_ms`, plus extras
  (`kind`, `character`, `text`).

### Phase 3 — Video (Member 2)
A three-tier visual pipeline, each with a graceful local fallback so the
final MP4 always renders even when nothing is reachable:

| Stage | Remote (Kaggle GPU) | Local fallback |
|---|---|---|
| Speaker portrait per dialogue line | **SDXL-Turbo** at 1 step (~3 s on T4) | Local SDXL on CPU (~30 s) → DALL-E 3 → PIL stylised stills |
| Per-line ambient motion | **Stable Video Diffusion XT** with `enable_model_cpu_offload` + `decode_chunk_size=2` (~2.5 min/clip on T4) | FFmpeg `zoompan` Ken Burns clip on the still |
| Mouth lip-sync to dialogue audio | **Wav2Lip** (Rudrabha + justinjohn0306 mirror) | FFmpeg passthrough — still + audio, no mouth motion but a valid clip |

- **Compositor** ([`phase3_video/compositor.py`](phase3_video/compositor.py))
  stitches per-line clips into per-scene MP4s with BGM mixed at -12 dB,
  burns subtitles via FFmpeg `drawtext` (replacing MoviePy `TextClip` which
  silently failed without ImageMagick), and concatenates scenes with
  `xfade` 0.5 s crossfade transitions. Falls back to plain `concat`
  demuxer + finally a 1-byte placeholder if everything fails.
- **Remote endpoint contract** ([`phase3_video/_http.py`](phase3_video/_http.py))
  is one base URL exposing `/sdxl`, `/svd`, `/lipsync`, `/health`, `/unload`.
  All routes ship the `ngrok-skip-browser-warning` header so free-tier
  ngrok doesn't return its HTML interstitial.
- **Auto-cleanup:** every `run_phase3` call POSTs `/unload` first to free
  GPU caches accumulated by the previous run — no kernel restart needed
  on the Kaggle side.
- **Phase 5 cooperation:** an `apply_filter` edit invalidates the affected
  scene's per-line clip cache so the filter actually shows up in the
  recomposite, then `run_phase3` re-renders from the (now filtered) inputs.
- **GPU planning:** when Kaggle is in `T4 x2` mode, SVD takes `cuda:0`
  (with cpu_offload for VRAM safety alongside the Wav2Lip subprocess) and
  SDXL takes `cuda:1` permanently. Auto-detected; falls back to
  lock-and-swap on a single GPU.

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
pytest                    # full suite — 46 tests, all offline
pytest tests/test_phase5_edit.py -v   # 15 tests, includes 11 distinct edit-query types
```

The test suite **deliberately deletes `OPENAI_API_KEY` and `KAGGLE_ENDPOINT`
from the environment** in `tests/conftest.py` so the offline / fallback paths
are what gets graded.

| Phase | Tests | What's covered |
| --- | --- | --- |
| 1 | 11 | Story shape · contiguous scene numbers · speaker-in-roster invariant · each of the 6 named tools in isolation · serializer writes 6 handoff JSONs with the right shape |
| 2 | 9 | TTS voice mapping · BGM file generation · manifest segments per scene · duration realignment · prosody table distinct rates · synthesize accepts rate+pitch kwargs · `timing_manifest.json` written to disk · handoff emotions consumed |
| 3 | 8 | Scene + speaker PNG generation · darken filter changes pixels · Ken Burns clip · SVD passthrough · Wav2Lip passthrough · per-line clips written · `final_output.mp4` always exists |
| 4 | 3 | `/api/health` · empty-project list · 404 for unknown project |
| 5 | 15 | 11 query-type classifications · scope+param extraction · snapshot+revert+history |

---

## Sample run

A real run produces this layout (per project under `outputs/projects/`):

```
20260503-162725-35e62336/
├── current/
│   ├── state.json                    # umbrella ProjectState (Phase 4 reads this)
│   │
│   ├── story.json                    # Phase 1 handoffs ─┐
│   ├── characters.json               #                    │ also embedded in
│   ├── script.json                   #                    │ state.json — these
│   ├── phase2_audio_handoff.json     #                    │ are the named
│   ├── phase3_video_handoff.json     #                    │ artifacts the
│   ├── summary.json                  #                   ─┘ spec lists
│   │
│   ├── timing_manifest.json          # Phase 2 standalone artifact
│   │
│   ├── audio/
│   │   ├── scene01_line01.mp3        # edge-tts MP3 per dialogue line
│   │   ├── scene01_line01.wav16k.wav # 16 kHz mono WAV cached for Wav2Lip
│   │   └── scene01_bgm.wav           # procedural BGM per scene
│   │
│   ├── images/
│   │   ├── scene01.png               # establishing shot (Phase 5 filter target)
│   │   └── scene01_line01_<slug>.png # per-line speaker portrait
│   │
│   ├── clips/
│   │   ├── scene01_line01_motion.mp4 # SVD ambient-motion (or KB fallback)
│   │   ├── scene01_line01.mp4        # Wav2Lip lip-synced (or passthrough)
│   │   └── scene01.mp4               # per-scene composite with BGM + subs
│   │
│   └── final_output.mp4              # full story, scenes joined with xfade
│
├── v1/   (Initial generation)
├── v2/   (Edit: Make scene 2 darker)
├── v3/   (Edit: Apply a sepia filter to scene 1)
├── v4/   (Reverted to v1)
└── history.json
```

---

## Tech stack

| Layer | Tool | Why |
| --- | --- | --- |
| Agent runtime | **LangGraph** | StateGraph + checkpointer for both Phase 1 (3 agents) and Phase 5 (intent classifier) |
| LLM | OpenAI `gpt-4o-mini` (optional) | Phase 1 stories + Phase 5 intent classification |
| Schema | **Pydantic v2** | Validates every phase boundary, including the 6 Phase-1 handoff files |
| TTS | `edge-tts` (free) + emotion-driven prosody | Free, no API key, async; runtime fallback voice when Microsoft retires one |
| BGM | `numpy` + `wave` | Procedural pads — no licensing concerns |
| Image gen | **SDXL-Turbo** (`diffusers`) → DALL-E 3 → PIL stylised stills | Three-tier fallback; SDXL runs locally on CPU OR remotely on Kaggle GPU |
| Video gen | **Stable Video Diffusion XT** on Kaggle | Image → 2-3 s ambient motion clip per dialogue line |
| Lip-sync | **Wav2Lip** (Rudrabha) on Kaggle | Mouth motion synced to dialogue audio |
| Composition | `ffmpeg` (zoompan, drawtext, xfade, amix) | One re-encode pass per stage; replaces MoviePy's TextClip |
| Remote tunnel | `pyngrok` + Kaggle T4×2 GPU | Free GPU compute exposed via one HTTP base URL with `/sdxl`, `/svd`, `/lipsync`, `/unload`, `/health` |
| Backend | **FastAPI** + `uvicorn` | Fast, type-safe, native WebSocket |
| Frontend | Hand-written HTML/CSS/JS | No build step, no Node dependency |
| State store | Append-only filesystem snapshots | Simple, durable, transparent in `outputs/` |
| Tests | `pytest` + `pytest-asyncio` | 46 tests, all offline |

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
