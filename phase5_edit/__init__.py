"""Phase 5 — Intelligent Edit & Undo (Member 4 / shared)."""

from phase5_edit.intent_agent import classify_edit_intent
from phase5_edit.state_manager import StateManager
from phase5_edit.executor import apply_edit

__all__ = ["classify_edit_intent", "StateManager", "apply_edit"]
