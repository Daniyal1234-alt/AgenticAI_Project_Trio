"""
FastAPI application — the user-facing surface for the whole project.

Routes:
    GET  /                          → static frontend (index.html)
    GET  /api/health                → liveness + capability flags
    POST /api/projects              → start a new generation; returns project_id
    WS   /api/projects/{id}/ws      → stream phase-by-phase progress messages
    GET  /api/projects/{id}         → current ProjectState + history
    GET  /api/projects/{id}/file/*  → safe access to artifacts in current/
    GET  /api/projects/{id}/v/{n}/file/*  → safe access to a specific snapshot
    POST /api/projects/{id}/edit    → run Phase 5 edit agent + executor
    POST /api/projects/{id}/revert  → undo to a given version
    GET  /api/projects              → list known projects (most-recent first)
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from phase4_web.orchestrator import (
    PROJECTS_ROOT,
    generate_full_pipeline,
    new_project_id,
)
from phase5_edit.executor import apply_edit
from phase5_edit.intent_agent import classify_edit_intent
from phase5_edit.state_manager import StateManager
from schemas.pipeline import ProjectState

load_dotenv()


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


app = FastAPI(title="Agentic AI — Animated Video Generator", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Per-project progress fan-out                                                 #
# --------------------------------------------------------------------------- #


class ProgressBus:
    """One queue per WebSocket; many consumers per project_id."""

    def __init__(self) -> None:
        self._consumers: dict[str, list[asyncio.Queue[str]]] = {}

    def subscribe(self, pid: str) -> asyncio.Queue[str]:
        q: asyncio.Queue[str] = asyncio.Queue()
        self._consumers.setdefault(pid, []).append(q)
        return q

    def unsubscribe(self, pid: str, q: asyncio.Queue[str]) -> None:
        if pid in self._consumers and q in self._consumers[pid]:
            self._consumers[pid].remove(q)

    async def publish(self, pid: str, msg: str) -> None:
        for q in self._consumers.get(pid, []):
            await q.put(msg)


bus = ProgressBus()


def _publisher(pid: str):
    async def cb(msg: str) -> None:
        await bus.publish(pid, msg)
    return cb


# --------------------------------------------------------------------------- #
# Models                                                                       #
# --------------------------------------------------------------------------- #


class CreateProjectBody(BaseModel):
    prompt: str = Field(..., min_length=3)
    num_scenes: int = Field(default=3, ge=2, le=8)
    style: str = Field(default="cinematic")


class EditBody(BaseModel):
    query: str = Field(..., min_length=2)


class RevertBody(BaseModel):
    version: int = Field(..., ge=1)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _project_dir(project_id: str) -> str:
    safe = os.path.basename(project_id)
    if not safe or safe in (".", ".."):
        raise HTTPException(404, "Invalid project id")
    pdir = os.path.join(PROJECTS_ROOT, safe)
    if not os.path.isdir(pdir):
        raise HTTPException(404, "Project not found")
    return pdir


def _safe_under(base: str, rel_path: str) -> str:
    base = os.path.realpath(base)
    candidate = os.path.realpath(os.path.join(base, rel_path))
    if not (candidate == base or candidate.startswith(base + os.sep)):
        raise HTTPException(404, "Invalid path")
    if not os.path.isfile(candidate):
        raise HTTPException(404, "File not found")
    return candidate


def _state_payload(sm: StateManager) -> dict:
    state = sm.load_state()
    return {
        "state": state.model_dump() if state else None,
        "history": [e.model_dump() for e in sm.history()],
    }


# --------------------------------------------------------------------------- #
# Routes — meta                                                                #
# --------------------------------------------------------------------------- #


@app.get("/api/health")
def health() -> dict:
    import shutil

    return {
        "status": "ok",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
        "moviepy_available": _moviepy_present(),
        "projects_root": PROJECTS_ROOT,
    }


def _moviepy_present() -> bool:
    try:
        import moviepy.editor  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Routes — projects                                                            #
# --------------------------------------------------------------------------- #


@app.get("/api/projects")
def list_projects() -> dict:
    if not os.path.isdir(PROJECTS_ROOT):
        return {"projects": []}
    items = []
    for name in os.listdir(PROJECTS_ROOT):
        sm = StateManager(PROJECTS_ROOT, name)
        if not os.path.isfile(sm.state_path):
            continue
        st = sm.load_state()
        items.append(
            {
                "project_id": name,
                "title": (st.story.title if st and st.story else "(untitled)"),
                "phase": st.phase if st else "unknown",
                "version": st.version if st else 0,
                "history_count": len(sm.history()),
            }
        )
    items.sort(key=lambda x: x["project_id"], reverse=True)
    return {"projects": items}


@app.post("/api/projects")
async def create_project(body: CreateProjectBody) -> dict:
    pid = new_project_id()
    # Ensure the project dir exists immediately so the UI can subscribe.
    StateManager(PROJECTS_ROOT, pid)

    async def runner():
        try:
            await generate_full_pipeline(
                body.prompt.strip(),
                project_id=pid,
                num_scenes=body.num_scenes,
                style=body.style,
                progress=_publisher(pid),
            )
            await bus.publish(pid, "__DONE__")
        except Exception as exc:
            await bus.publish(pid, f"[ERROR] {exc!r}")
            await bus.publish(pid, "__DONE__")

    asyncio.create_task(runner())
    return {"project_id": pid}


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict:
    _project_dir(project_id)
    sm = StateManager(PROJECTS_ROOT, project_id)
    return _state_payload(sm)


@app.websocket("/api/projects/{project_id}/ws")
async def project_ws(ws: WebSocket, project_id: str) -> None:
    """
    Stream pipeline progress to the browser.

    Single-step Phase 3 stages (SDXL → SVD → Wav2Lip per dialogue line) can
    each take 30-300 s on Kaggle. Without a periodic heartbeat the browser
    drops the WebSocket on idle, and a single Wav2Lip face-detect failure
    that runs ~3 min silent is enough to kill the stream. We send a `:` ping
    every 15 s alongside whatever the pipeline emits — `app.js` can ignore
    pings (single-character messages) when rendering the log.
    """
    await ws.accept()
    q = bus.subscribe(project_id)
    HEARTBEAT_SECONDS = 15.0
    try:
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=HEARTBEAT_SECONDS)
            except asyncio.TimeoutError:
                # No pipeline output for HEARTBEAT_SECONDS — keep the WS alive.
                await ws.send_text(":")
                continue
            await ws.send_text(msg)
            if msg == "__DONE__":
                break
    except WebSocketDisconnect:
        pass
    finally:
        bus.unsubscribe(project_id, q)


# --------------------------------------------------------------------------- #
# Routes — files                                                               #
# --------------------------------------------------------------------------- #


@app.get("/api/projects/{project_id}/file/{rel:path}")
def project_file(project_id: str, rel: str):
    pdir = _project_dir(project_id)
    full = _safe_under(os.path.join(pdir, "current"), rel)
    mime, _ = mimetypes.guess_type(full)
    return FileResponse(full, media_type=mime or "application/octet-stream")


@app.get("/api/projects/{project_id}/v/{version}/file/{rel:path}")
def project_version_file(project_id: str, version: int, rel: str):
    pdir = _project_dir(project_id)
    full = _safe_under(os.path.join(pdir, f"v{version}"), rel)
    mime, _ = mimetypes.guess_type(full)
    return FileResponse(full, media_type=mime or "application/octet-stream")


# --------------------------------------------------------------------------- #
# Routes — Phase 5 edit + revert                                               #
# --------------------------------------------------------------------------- #


@app.post("/api/projects/{project_id}/edit")
async def edit_project(project_id: str, body: EditBody) -> dict:
    pdir = _project_dir(project_id)
    sm = StateManager(PROJECTS_ROOT, project_id)
    state = sm.load_state()
    if state is None or state.story is None:
        raise HTTPException(400, "Project not ready — run generation first")

    intent, log = classify_edit_intent(body.query, thread_id=project_id)

    async def runner():
        try:
            for entry in log:
                await bus.publish(project_id, entry)
            await bus.publish(
                project_id,
                f"[Edit] Intent: {intent.intent} target={intent.target} scope={intent.scope}",
            )
            new_state = await apply_edit(
                state, intent, os.path.join(pdir, "current"),
                progress=_publisher(project_id),
            )
            new_state.edit_log.append(
                {"query": body.query, "intent": intent.model_dump()}
            )
            sm.save_state(new_state)
            sm.snapshot(summary=f"Edit: {body.query[:60]}", intent=intent)
            await bus.publish(project_id, "__DONE__")
        except Exception as exc:
            await bus.publish(project_id, f"[ERROR] {exc!r}")
            await bus.publish(project_id, "__DONE__")

    asyncio.create_task(runner())
    return {"intent": intent.model_dump(), "log": log}


@app.post("/api/projects/{project_id}/revert")
def revert_project(project_id: str, body: RevertBody) -> dict:
    _project_dir(project_id)
    sm = StateManager(PROJECTS_ROOT, project_id)
    try:
        entry = sm.revert(body.version)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    return {"reverted_to": body.version, "new_version": entry.version, "history": [e.model_dump() for e in sm.history()]}


# --------------------------------------------------------------------------- #
# Static frontend                                                              #
# --------------------------------------------------------------------------- #


if os.path.isdir(STATIC_DIR):
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


@app.get("/")
def index():
    p = os.path.join(STATIC_DIR, "index.html")
    if not os.path.isfile(p):
        raise HTTPException(500, "Frontend missing — phase4_web/static/index.html")
    return FileResponse(p)
