"""
CLI entry point for the Agentic AI Animated Video Generation System.

Usage:
    # Run the full pipeline end-to-end and print the project_id:
    python main.py --prompt "A young astronaut discovers a hidden ocean on Mars"

    # Apply an edit to an existing project:
    python main.py --edit "Make scene 2 darker" --project 20260505-XXXXXXXX

    # Revert to a prior version:
    python main.py --revert 1 --project 20260505-XXXXXXXX

    # Launch the web UI:
    uvicorn phase4_web.api:app --host 127.0.0.1 --port 8000 --reload
    # then open http://127.0.0.1:8000/
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Ensure UTF-8 stdout on Windows.
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


async def cmd_generate(args: argparse.Namespace) -> int:
    from phase4_web.orchestrator import generate_full_pipeline

    async def progress(msg: str) -> None:
        print(msg, flush=True)

    state = await generate_full_pipeline(
        args.prompt,
        num_scenes=args.scenes,
        style=args.style,
        progress=progress,
    )
    print()
    print("=" * 60)
    print(f" Project ID:    {state.project_id}")
    print(f" Title:         {state.story.title if state.story else '(none)'}")
    print(f" Final video:   outputs/projects/{state.project_id}/current/{state.final_video}")
    print(f" Phase:         {state.phase}  ·  Version: {state.version}")
    print("=" * 60)
    return 0


async def cmd_edit(args: argparse.Namespace) -> int:
    from phase4_web.orchestrator import PROJECTS_ROOT
    from phase5_edit.executor import apply_edit
    from phase5_edit.intent_agent import classify_edit_intent
    from phase5_edit.state_manager import StateManager

    sm = StateManager(PROJECTS_ROOT, args.project)
    state = sm.load_state()
    if state is None or state.story is None:
        print(f"[ERROR] No state for project {args.project}")
        return 1

    intent, log = classify_edit_intent(args.edit, thread_id=args.project)
    for line in log:
        print(line)
    print(f"[Edit] Intent: {intent.intent} target={intent.target} scope={intent.scope}")

    async def progress(msg: str) -> None:
        print(msg, flush=True)

    state = await apply_edit(state, intent, sm.current_dir, progress=progress)
    state.edit_log.append({"query": args.edit, "intent": intent.model_dump()})
    sm.save_state(state)
    entry = sm.snapshot(summary=f"Edit: {args.edit[:60]}", intent=intent)
    print(f"[Edit] Snapshot v{entry.version} saved")
    return 0


def cmd_revert(args: argparse.Namespace) -> int:
    from phase4_web.orchestrator import PROJECTS_ROOT
    from phase5_edit.state_manager import StateManager

    sm = StateManager(PROJECTS_ROOT, args.project)
    entry = sm.revert(args.revert)
    print(f"[Revert] Reverted to v{args.revert} → snapshot v{entry.version}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agentic AI animated video generator")
    p.add_argument("--prompt", help="Story prompt (runs full pipeline)")
    p.add_argument("--scenes", type=int, default=3, help="Number of scenes (2–8)")
    p.add_argument("--style", default="cinematic", help="cinematic / cartoon / noir / anime / watercolor")
    p.add_argument("--edit", help="Apply an edit query to an existing project")
    p.add_argument("--revert", type=int, help="Revert to a specific version of an existing project")
    p.add_argument("--project", help="Existing project_id (required for --edit and --revert)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.revert is not None:
        if not args.project:
            print("[ERROR] --revert requires --project")
            return 2
        return cmd_revert(args)
    if args.edit:
        if not args.project:
            print("[ERROR] --edit requires --project")
            return 2
        return asyncio.run(cmd_edit(args))
    if args.prompt:
        return asyncio.run(cmd_generate(args))
    print("[ERROR] one of --prompt / --edit / --revert is required")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
