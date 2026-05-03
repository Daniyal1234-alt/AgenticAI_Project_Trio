"""
StateManager — append-only versioned state store.

On every snapshot we copy the project's `current/` directory to `vN/` and
record a `VersionEntry` in `history.json`. `revert(n)` copies `vN/` back
on top of `current/`. Nothing is ever deleted, so undo is always available
all the way back to v1.

Layout:
    outputs/projects/<project_id>/
        ├── history.json            ← list[VersionEntry]
        ├── current/                ← what the pipeline reads/writes today
        │   ├── state.json
        │   ├── audio/  images/
        │   └── final_output.mp4
        ├── v1/
        ├── v2/
        └── ...
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
from typing import Optional

from pydantic import ValidationError

from schemas.edit import EditIntent, VersionEntry
from schemas.pipeline import ProjectState


def _utcnow_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class StateManager:
    """Project-scoped version store. One instance per project_id."""

    def __init__(self, projects_root: str, project_id: str):
        self.project_id = project_id
        self.project_dir = os.path.join(projects_root, project_id)
        self.current_dir = os.path.join(self.project_dir, "current")
        self.history_path = os.path.join(self.project_dir, "history.json")
        os.makedirs(self.current_dir, exist_ok=True)

    # ----------------------------------------------------------------- #
    # Current state                                                     #
    # ----------------------------------------------------------------- #

    @property
    def state_path(self) -> str:
        return os.path.join(self.current_dir, "state.json")

    def load_state(self) -> Optional[ProjectState]:
        if not os.path.isfile(self.state_path):
            return None
        with open(self.state_path, encoding="utf-8") as f:
            data = json.load(f)
        try:
            return ProjectState.model_validate(data)
        except ValidationError:
            return None

    def save_state(self, state: ProjectState) -> None:
        with open(self.state_path, "w", encoding="utf-8") as f:
            f.write(state.model_dump_json(indent=2))

    # ----------------------------------------------------------------- #
    # Versioning                                                        #
    # ----------------------------------------------------------------- #

    def _next_version(self) -> int:
        history = self.history()
        return (max((v.version for v in history), default=0)) + 1

    def history(self) -> list[VersionEntry]:
        if not os.path.isfile(self.history_path):
            return []
        with open(self.history_path, encoding="utf-8") as f:
            data = json.load(f)
        out = []
        for entry in data:
            try:
                out.append(VersionEntry.model_validate(entry))
            except ValidationError:
                continue
        return out

    def _write_history(self, entries: list[VersionEntry]) -> None:
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump([e.model_dump() for e in entries], f, indent=2)

    def snapshot(
        self,
        summary: str,
        *,
        intent: Optional[EditIntent] = None,
    ) -> VersionEntry:
        """Copy `current/` into `vN/` and append a history entry."""
        version = self._next_version()
        snap_dir = os.path.join(self.project_dir, f"v{version}")
        if os.path.isdir(snap_dir):
            shutil.rmtree(snap_dir)
        shutil.copytree(self.current_dir, snap_dir)

        # Make sure the copied state.json reflects the version number we just assigned.
        st = self.load_state()
        if st is not None:
            st.version = version
            self.save_state(st)
            # Re-copy state.json so v{N}/state.json matches.
            shutil.copy2(self.state_path, os.path.join(snap_dir, "state.json"))

        entry = VersionEntry(
            version=version,
            created_at=_utcnow_iso(),
            summary=summary,
            snapshot_dir=os.path.relpath(snap_dir, self.project_dir).replace("\\", "/"),
            edit_intent=intent,
        )
        entries = self.history() + [entry]
        self._write_history(entries)
        return entry

    def revert(self, version: int) -> VersionEntry:
        """Restore `current/` to the contents of `vN/` and append a revert entry."""
        target_dir = os.path.join(self.project_dir, f"v{version}")
        if not os.path.isdir(target_dir):
            raise FileNotFoundError(f"Version v{version} does not exist")

        # Wipe current/ and copy from snapshot.
        if os.path.isdir(self.current_dir):
            shutil.rmtree(self.current_dir)
        shutil.copytree(target_dir, self.current_dir)

        # Mark the active version on the restored state.
        st = self.load_state()
        if st is not None:
            st.version = version
            st.phase = "edited"
            self.save_state(st)

        # Record the revert as a new version (so undo is itself undoable).
        new_version = self._next_version()
        snap_dir = os.path.join(self.project_dir, f"v{new_version}")
        shutil.copytree(self.current_dir, snap_dir)
        entry = VersionEntry(
            version=new_version,
            created_at=_utcnow_iso(),
            summary=f"Reverted to v{version}",
            snapshot_dir=os.path.relpath(snap_dir, self.project_dir).replace("\\", "/"),
            edit_intent=None,
        )
        entries = self.history() + [entry]
        self._write_history(entries)
        return entry

    # ----------------------------------------------------------------- #
    # Convenience                                                       #
    # ----------------------------------------------------------------- #

    def diff_summary(self) -> list[dict]:
        return [
            {
                "version": e.version,
                "created_at": e.created_at,
                "summary": e.summary,
                "intent": e.edit_intent.model_dump() if e.edit_intent else None,
            }
            for e in self.history()
        ]
