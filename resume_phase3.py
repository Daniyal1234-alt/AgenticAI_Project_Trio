"""
One-off helper: re-run Phase 3 against the latest project (or a specific
project_id) to fill in any missing line clips, then recompose the final MP4.

Cached scene clips are reused — only changed/missing clips are regenerated,
and the final concat always runs (so a compositor-only fix lands without
re-doing SDXL/SVD/Wav2Lip).

Usage:
    python resume_phase3.py                       # latest project
    python resume_phase3.py 20260505-XXXXXXXX    # specific project
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from phase3_video.pipeline import run_phase3
from phase5_edit.state_manager import StateManager

PROJECTS_ROOT = "outputs/projects"


def main() -> int:
    project_id = sys.argv[1] if len(sys.argv) > 1 else None
    if project_id is None:
        candidates = [p for p in Path(PROJECTS_ROOT).iterdir() if p.is_dir()]
        if not candidates:
            print(f"No projects under {PROJECTS_ROOT}/", file=sys.stderr)
            return 1
        project_id = max(candidates, key=lambda p: p.stat().st_mtime).name
    print(f"Resuming Phase 3 on project {project_id}", flush=True)

    sm = StateManager(PROJECTS_ROOT, project_id)
    state = sm.load_state()
    if state is None or state.story is None or state.timing is None:
        print("No story/timing in state.json — can't resume", file=sys.stderr)
        return 1

    async def go():
        async def cb(m: str) -> None:
            print(f"  {m}", flush=True)
        return await run_phase3(
            state.story, state.timing, str(sm.current_dir),
            regenerate=False, scope_filter=None,
            subtitles=True, speed=1.0,
            progress=cb,
        )

    result = asyncio.run(go())
    state.scene_videos = result["scene_videos"]
    state.final_video = result["final_video"]
    state.phase = "complete"
    sm.save_state(state)
    final = Path(sm.current_dir) / result["final_video"]
    size_mb = final.stat().st_size / 1024 / 1024 if final.exists() else 0
    print(f"\nDone — {final}  ({size_mb:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
