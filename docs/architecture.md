# Architecture

This document explains the dataflow and the reasoning behind each phase. The
README has the user-facing summary; this is the deeper explanation a grader or
future maintainer would want.

---

## 1. The single contract — `schemas/pipeline.ProjectState`

Everything in this system flows through **one** Pydantic model. Phase 1 fills
in `story`. Phase 2 fills in `timing` and updates `story.scenes[*].duration_seconds`
based on actual audio length. Phase 3 fills in `scene_videos[]` and `final_video`.
Phase 5 mutates whichever subtree the edit targets. The web layer serialises
the whole thing to the frontend.

Why this matters: when phase boundaries are tightly typed, integration bugs
are caught at the earliest possible moment. We `model_validate()` at every
ingress point, so a malformed Phase 1 output can never silently corrupt
Phase 2.

---

## 2. Phase 1 — LangGraph story generation

```
generate_story ──► validate_story ──┬─► enrich_visuals ──► END
                                    │
                                    ▼ (invalid AND retries < 1)
                                  retry → generate_story
```

- **`generate_story`** — calls `gpt-4o-mini` if `OPENAI_API_KEY` is set,
  otherwise calls `schemas/llm.py:stub_story()`. Both produce a dict that
  matches the `Story` schema.
- **`validate_story`** — Pydantic structural check + three semantic
  invariants: every dialogue speaker exists in `characters[]`, every scene
  has at least one dialogue line, scene numbers are contiguous from 1.
- **`enrich_visuals`** — synthesises `visual_prompt` for any scene that
  doesn't have one. This way Phase 3 always has a usable image-generator
  prompt.
- The retry loop runs at most once. On a second failure the agent accepts
  the best-effort story rather than blocking the rest of the pipeline.

---

## 3. Phase 2 — Audio with deterministic timing

The trick here is that audio length determines visual length. Workflow:

1. Resolve a TTS voice for every character once (`tts.voice_for`).
2. For each scene, synthesise dialogue lines sequentially and accumulate a
   monotonically increasing `cursor_ms`.
3. If the actual dialogue is shorter than the scene's declared
   `duration_seconds`, pad up to that target so visuals don't end early.
4. Write a per-scene BGM file and register it as a scene-level `AudioSegment`
   spanning the same time window.
5. Call `realign_scene_durations(story, manifest)` — Phase 3 will then hold
   each image on screen for exactly as long as the audio plays.

The fallback path matters: when `edge-tts` can't reach Microsoft (no internet,
firewall, locked-down lab), `synthesize()` writes a silent WAV at the
estimated duration. The pipeline still produces a valid manifest and a valid
final MP4 — just without spoken voice.

---

## 4. Phase 3 — Three-tier video composition

```
                ┌──── MoviePy v1 / v2 (preferred) ────┐
                │                                     │
generate ──►   ├──── ffmpeg subprocess (mid-tier) ───┼──► final_output.mp4
images          │                                     │
                └──── slideshow PNG sidecar (fallback)┘
```

The placeholder image renderer was a deliberate choice: a deterministic,
mood-driven gradient + character silhouette + vignette + scene-heading title
card produces output that looks intentional. When you set
`IMAGE_BACKEND=openai` in `.env`, DALL-E 3 takes over.

The compositor's MoviePy path supports both v1.x (`from moviepy.editor import …`)
and v2.x (top-level imports + `with_*` instead of `set_*`). The version is
detected at import time and the right method shim is applied. This was
necessary because pip's resolver doesn't pin MoviePy and university lab
machines often have a mix.

---

## 5. Phase 4 — Web layer

REST + WebSocket — no SSR, no build step, no Node dependency. The frontend
is a single `index.html` + a hand-written `app.js` that talks to:

- `POST /api/projects` — fires off the full pipeline, returns a
  `project_id` immediately.
- `WS /api/projects/{id}/ws` — streams progress messages from
  `phase4_web/orchestrator.py:_emit()` until a `__DONE__` sentinel.
- `GET /api/projects/{id}` — current `ProjectState` + `history`.
- `GET /api/projects/{id}/file/{rel}` — safe path resolution under
  `current/`, used by the `<video>` tag and any debug downloads.
- `GET /api/projects/{id}/v/{n}/file/{rel}` — same, but scoped to a
  historical snapshot (useful for diffing v1 vs v3 visually).
- `POST /api/projects/{id}/edit` — delegates to Phase 5.
- `POST /api/projects/{id}/revert` — delegates to `StateManager.revert()`.

A single `ProgressBus` fans `_emit()` calls out to all subscribed WebSockets
for a given project. This means a long-running edit can be observed from
multiple browser tabs.

---

## 6. Phase 5 — Edit agent

```
free-text query
      │
      ▼
LangGraph(checkpointer=MemorySaver)   ← keyed by project_id
      │
classify_node:
  - LLM (if OPENAI_API_KEY) with 10 few-shot examples
  - else: rule-based stub_intent()
      │
      ▼
EditIntent (Pydantic)
      │
      ▼
executor.apply_edit(state, intent, project_dir)
      │
      ▼ depending on intent.target …
audio        → re-run Phase 2 → recomposite (reuse images)
video_frame  → mutate images (filter) OR regenerate scoped images → recomposite
video        → recomposite only (different speed / subtitles)
script       → re-run Phase 1 → 2 → 3 (full cascade)
      │
      ▼
StateManager.snapshot() → vN+1
```

### The `scope_filter=set()` trick

The cleanest part of the executor is how it decides what to re-render. The
function signature of `run_phase3` is:

```python
async def run_phase3(..., scope_filter: Optional[set[int]] = None, ...)
```

- `scope_filter=None` (default) → render every scene's image. Used by full
  generation.
- `scope_filter={2, 3}` → render only scenes 2 and 3, reuse the rest.
- `scope_filter=set()` (empty set) → render nothing, reuse everything.

The third case is what makes filter edits idempotent: when you've already
mutated `images/scene02.png` in place via `apply_filter`, you want
`run_phase3` to compose the MP4 *without* overwriting your work. An empty
set says "trust the disk, don't regenerate".

### Append-only snapshots

`StateManager` writes every change to `vN/` and never deletes. This means:

- Storage grows linearly with edit count — fine for academic-scale projects.
- Undo is "copy `vN/` over `current/`" — no diffs, no merges, no compute.
- A revert to `vN` becomes a *new* version `vN+1`. So you can undo a revert
  by reverting again. No version is ever lost.
- The on-disk layout is the audit log. You can `git log -- v3/state.json`
  if you check the project into git.

---

## 7. Things deliberately not implemented

- **SQLite for state.** Asset blobs (audio, images, MP4) want to live on the
  filesystem regardless. Adding SQLite for the `history.json` metadata alone
  is over-engineering — a JSON list updated atomically is more transparent.
- **A diff visualiser between versions.** The data is on disk; a future
  PR can render it. The contract is in place.
- **Voice cloning.** Out of scope per the spec ("consistent character voices"
  is satisfied by deterministic style→voice mapping).
- **Server-side authentication.** This is a single-user lab demo.
