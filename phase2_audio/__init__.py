"""Phase 2 — Audio generation (Member 2).

Public entry: `await run_phase2(story, project_dir)` writes audio files to
`{project_dir}/audio/` and returns a `TimingManifest`.
"""

from phase2_audio.pipeline import run_phase2

__all__ = ["run_phase2"]
